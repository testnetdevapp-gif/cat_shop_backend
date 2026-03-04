import os
import re
import json
import uuid
import logging
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import time

load_dotenv()

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "models/gemini-2.5-flash"

# ── Safety Settings ───────────────────────────────────────────
SAFETY_SETTINGS = [
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
]

# ── Prompt ────────────────────────────────────────────────────
CAT_ANALYSIS_PROMPT = """
You are a professional cat analysis AI specialized in pet body measurement and health assessment.

STRICT OUTPUT REQUIREMENTS:
- Return raw JSON only. No markdown. No explanation. No extra text.
- JSON must be complete and properly closed — never truncate.
- Every field must be present. Use null only where explicitly allowed.

If no cat is detected, return ONLY:
{"is_cat": false, "message": "ไม่พบแมวในภาพ"}

If a cat is detected, return this exact schema (all fields required):
{
  "is_cat": true,
  "cat_color": "describe main color(s) e.g. orange, black and white, grey tabby",
  "breed": "string or null",
  "age": 3,
  "gender": 0,

  "weight": 4.5,
  "chest_cm": 32.0,
  "neck_cm": 22.0,
  "waist_cm": 28.0,
  "body_length_cm": 45.0,
  "back_length_cm": 38.0,
  "leg_length_cm": 12.0,

  "body_condition_score": 5,
  "body_condition": "normal",
  "body_condition_description": "Healthy weight, ribs palpable with slight fat cover",

  "posture": "sitting",
  "size_recommendation": "M",
  "size_ranges": {
    "chest_min": 32.0,
    "chest_max": 36.0,
    "neck_min": 20.0,
    "neck_max": 24.0,
    "back_length_min": 35.0,
    "back_length_max": 42.0
  },

  "quality_flag": "good",
  "confidence": 0.87
}

DERIVATION RULES:

age (integer, NEVER null — use 0 if truly unknown):
  Estimate from face, body proportions, coat condition.

age_category (derived by backend — do NOT include in response)

gender: 0=unknown/female, 1=male

weight (kg, float): estimate from body volume vs typical domestic cat.

size_recommendation and size_ranges (derive from chest_cm):
  XS: chest < 28    neck_min=16 neck_max=20 back_min=28 back_max=34
  S : 28<=chest<32  neck_min=18 neck_max=22 back_min=32 back_max=38
  M : 32<=chest<36  neck_min=20 neck_max=24 back_min=36 back_max=42
  L : 36<=chest<40  neck_min=22 neck_max=26 back_min=40 back_max=46
  XL: chest>=40     neck_min=24 neck_max=28 back_min=44 back_max=50

body_condition_score: integer 1(emaciated) to 9(obese), 4-5=ideal
body_condition: underweight | normal | overweight | obese
posture: standing | sitting | lying | crouching | other
quality_flag: good | blurry | partial | dark | backlit | other
confidence: 0.0-1.0 reflecting image clarity and full body visibility
"""


# ── Pydantic Schema ───────────────────────────────────────────
class SizeRanges(BaseModel):
    chest_min: float
    chest_max: float
    neck_min: float
    neck_max: float
    back_length_min: float
    back_length_max: float


class CatAnalysisSchema(BaseModel):
    is_cat: bool
    cat_color: str
    breed: Optional[str] = None
    age: int = 0
    gender: int = 0
    weight: float
    chest_cm: float
    neck_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    body_length_cm: Optional[float] = None
    back_length_cm: Optional[float] = None
    leg_length_cm: Optional[float] = None
    body_condition_score: int = Field(..., ge=1, le=9)
    body_condition: str
    body_condition_description: Optional[str] = None
    posture: str = "other"
    size_recommendation: Optional[str] = None
    size_ranges: Optional[SizeRanges] = None
    quality_flag: str = "good"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("weight", "chest_cm", mode="before")
    @classmethod
    def cast_to_float(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert '{v}' to float")

    @field_validator("body_condition_score", mode="before")
    @classmethod
    def clamp_bcs(cls, v):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 5
        return max(1, min(9, v))

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v):
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def from_ai(cls, data: dict) -> "CatAnalysisSchema":
       
        return cls(**data)


# ── Pure Helpers ──────────────────────────────────────────────

def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calc_bmi(weight: Optional[float], body_length_cm: Optional[float]) -> Optional[float]:
    if not weight or not body_length_cm or body_length_cm <= 0:
        return None
    return round(weight / ((body_length_cm / 100.0) ** 2), 2)


def _calc_size(chest: Optional[float]) -> str:
    if chest is None: return "M"
    if chest < 28:    return "XS"
    if chest < 32:    return "S"
    if chest < 36:    return "M"
    if chest < 40:    return "L"
    return "XL"


def _calc_age_category(age: int) -> str:
    if age < 1:   return "kitten"
    if age <= 2:  return "junior"
    if age <= 10: return "adult"
    return "senior"


def _log_parse_error(raw_text: str, error, request_id: str = "") -> None:
    log_path = f"parse_error_{int(time.time())}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"RequestID: {request_id}\nError: {error}\n\nRaw response:\n{raw_text}")
        logger.error(f"[{request_id}] Parse error logged to: {log_path}")
    except Exception as e:
        logger.error(f"[{request_id}] Failed to write parse error log: {e}")


# ── Robust JSON Parser ────────────────────────────────────────

def _parse_json_robust(raw_text: str) -> dict:
    """
    5-step fallback — รองรับ truncated, markdown fence, trailing garbage
    ใช้เฉพาะกรณีที่ _call_gemini_with_retry ยัง return มาได้
    """
    text = raw_text.strip()

    # Step 1: parse ตรงๆ
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2: ลบ markdown fence
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: brace-depth counting (แม่นกว่า regex greedy)
    start = cleaned.find('{')
    if start != -1:
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError as e:
                logger.warning(f"Brace-counted block still invalid: {e}")

    # Step 4: truncated repair (last resort)
    fragment = cleaned[start:] if start != -1 else cleaned
    repaired = _repair_truncated_json(fragment)
    if repaired:
        try:
            result = json.loads(repaired)
            logger.warning("Used truncated JSON repair — result may be incomplete")
            return result
        except json.JSONDecodeError:
            pass

    # Step 5: is_cat false fallback
    if '"is_cat": false' in text or '"is_cat":false' in text:
        return {"is_cat": False, "message": "ไม่พบแมวในภาพ"}

    raise RuntimeError(
        f"Gemini returned invalid JSON. "
        f"Length={len(text)}, Preview: {text[:300]}"
    )


def _repair_truncated_json(text: str) -> Optional[str]:
    try:
        lines = text.split('\n')
        complete_lines = []
        for line in lines:
            s = line.strip()
            if s.endswith(',') or s.endswith('{') or s.endswith('[') or s == '{':
                complete_lines.append(line)
            else:
                break
        if not complete_lines:
            return None
        partial = '\n'.join(complete_lines).rstrip().rstrip(',')
        depth_brace   = partial.count('{') - partial.count('}')
        depth_bracket = partial.count('[') - partial.count(']')
        return partial + ']' * depth_bracket + '}' * depth_brace
    except Exception:
        return None


# ── 🔥 Bulletproof Gemini Wrapper ────────────────────────────

def _call_gemini_with_retry(image_bytes: bytes, mime_type: str) -> str:
    """
    Production-safe Gemini caller
    - Retry เฉพาะ transient errors (truncated / empty / rate limit / timeout)
    - ไม่ retry ถ้า schema พัง หรือ quota หมด
    - มี request_id สำหรับ trace log
    - มี latency log
    """
    request_id = str(uuid.uuid4())[:8]
    max_retries = 3
    base_wait   = 3

    for attempt in range(max_retries):
        start = time.time()
        try:
            print(f"[{request_id}] 🤖 Gemini attempt {attempt + 1}/{max_retries}")

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=CAT_ANALYSIS_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=4000,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),   # ลดให้พอดี = ลดโอกาส truncate
                    safety_settings=SAFETY_SETTINGS,
                    response_mime_type="application/json",
                ),
            )

            # ── Extract text ──────────────────────────────────
            raw_text = ""
            if hasattr(response, "text") and response.text:
                raw_text = response.text.strip()
            else:
                raw_text = response.candidates[0].content.parts[0].text.strip()

            latency = round(time.time() - start, 2)

            # ── Guard: empty ──────────────────────────────────
            if not raw_text:
                raise RuntimeError("Empty Gemini response")

            # ── Guard: truncated (JSON ไม่ปิด) ────────────────
            stripped = raw_text.rstrip()
            if not stripped.endswith("}"):
                print(f"[{request_id}] ⚠️  Truncated response ({len(raw_text)} chars), retrying...")
                raise RuntimeError("Truncated Gemini JSON")

            print(f"[{request_id}] ✅ OK | {len(raw_text)} chars | {latency}s")
            return raw_text

        except Exception as e:
            error_str = str(e)
            latency   = round(time.time() - start, 2)
            print(f"[{request_id}] ❌ Attempt {attempt + 1} failed ({latency}s): {error_str}")

            # ── Quota หมดทั้งวัน → อย่า retry ────────────────
            if "limit: 0" in error_str or "PerDay" in error_str:
                raise RuntimeError("วันนี้ใช้ quota หมดแล้ว กรุณาลองใหม่พรุ่งนี้")

            # ── Retry เฉพาะ transient ─────────────────────────
            transient = any(kw in error_str.lower() for kw in [
                "truncated", "empty", "429", "resource_exhausted",
                "deadline", "timeout", "unavailable",
            ])

            if transient and attempt < max_retries - 1:
                wait = base_wait * (attempt + 1)  # 3s → 6s → 9s
                print(f"[{request_id}] ⏳ Retrying in {wait}s...")
                time.sleep(wait)
                continue

            # ── Non-transient หรือ retry หมด → raise ──────────
            raise RuntimeError(f"Gemini failed [{request_id}]: {error_str}")

    raise RuntimeError(f"Gemini failed completely after {max_retries} retries [{request_id}]")


# ── Main ──────────────────────────────────────────────────────

def analyze_cat(image_cat: str) -> dict:
    # 1. Download ──────────────────────────────────────────────
    print(f"⬇️  Downloading: {image_cat}")
    try:
        resp = requests.get(image_cat, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot download image: {e}")

    image_bytes = resp.content
    mime_type   = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    print(f"✅ Downloaded ({len(image_bytes)/1024:.1f} KB) | mime={mime_type}")

    # 2. Call Gemini (bulletproof wrapper) ─────────────────────
    raw_text = _call_gemini_with_retry(image_bytes, mime_type)

    # 3. Robust JSON parse ─────────────────────────────────────
    try:
        ai_data: dict = _parse_json_robust(raw_text)
    except RuntimeError as e:
        _log_parse_error(raw_text, e)
        raise

    # 4. Not a cat ─────────────────────────────────────────────
    if not ai_data.get("is_cat", True):
        return {
            "is_cat": False,
            "message": ai_data.get("message", "ไม่พบแมวในภาพ"),
        }

    # 5. Pydantic validation ───────────────────────────────────
    try:
        validated = CatAnalysisSchema.from_ai(ai_data)
    except Exception as e:
        _log_parse_error(raw_text, e)
        raise RuntimeError(f"AI response failed schema validation: {e}")

    # 6. Business logic — deterministic ───────────────────────
    weight   = _to_float(validated.weight)
    chest_cm = _to_float(validated.chest_cm)
    body_len = _to_float(validated.body_length_cm)
    age: int = validated.age

    size_category = _calc_size(chest_cm)
    age_category  = _calc_age_category(age)
    bmi           = _calc_bmi(weight, body_len)
    confidence    = validated.confidence if validated.confidence is not None else 0.5

    # 7. Return flat — map ตรงกับ DB columns ──────────────────
    result = {
        # Basic
        "is_cat":    True,
        "message":   "ok",
        "cat_color": validated.cat_color,
        "breed":     validated.breed,
        "age":       age,
        "gender":    validated.gender,

        # Weight & Size
        "weight":              weight,
        "size_category":       size_category,
        "size_recommendation": validated.size_recommendation,
        "size_ranges":         validated.size_ranges.model_dump() if validated.size_ranges else None,

        # Measurements (flat — ตรงกับ DB columns)
        "chest_cm":       chest_cm,
        "neck_cm":        _to_float(validated.neck_cm),
        "waist_cm":       _to_float(validated.waist_cm),
        "body_length_cm": body_len,
        "back_length_cm": _to_float(validated.back_length_cm),
        "leg_length_cm":  _to_float(validated.leg_length_cm),

        # Body condition
        "age_category":               age_category,
        "body_condition_score":       validated.body_condition_score,
        "body_condition":             validated.body_condition,
        "body_condition_description": validated.body_condition_description,
        "bmi":                        bmi,
        "posture":                    validated.posture,

        # Meta
        "confidence":       confidence,
        "quality_flag":     validated.quality_flag,
        "analysis_version": "2.0",
        "analysis_method":  "gemini_2.5_flash_vision",
    }

    print(
        f"✅ Done: {result['cat_color']} | size={size_category} "
        f"| chest={chest_cm}cm | weight={weight}kg | bmi={bmi} "
        f"| age={age} | confidence={confidence}"
    )
    return result
