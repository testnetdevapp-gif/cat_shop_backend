import os
import re
import json
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
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
]

CAT_ANALYSIS_PROMPT = """
You are a professional cat analysis AI specialized in pet body measurement and health assessment.

STRICT OUTPUT REQUIREMENTS:
- Return raw JSON only.
- No markdown.
- No explanation.
- No extra text.
- Ensure valid JSON format.
- If invalid, internally correct before returning.

If no cat is detected, return:
{
  "is_cat": false,
  "message": "ไม่พบแมวในภาพ"
}

If a cat is detected, return this exact schema:

{
  "is_cat": true,
  "cat_color": string,
  "breed": string,
  "age": integer (MUST be a number, never null — use 0 if unknown),
  "gender": 0,
  "age_category": "kitten" | "junior" | "adult" | "senior",
  "weight_kg": float,
  "chest_cm": float,
  "neck_cm": float or null,
  "waist_cm": float or null,
  "body_length_cm": float or null,
  "back_length_cm": float or null,
  "leg_length_cm": float or null,
  "body_condition_score": integer (1-9),
  "body_condition": "underweight" | "ideal" | "overweight",
  "body_condition_description": string,
  "posture": "standing" | "sitting" | "lying" | "other",
  "size_recommendation": "XS" | "S" | "M" | "L" | "XL",
  "size_ranges": {
    "chest_min": float,
    "chest_max": float,
    "neck_min": float,
    "neck_max": float,
    "back_length_min": float,
    "back_length_max": float
  },
  "quality_flag": "good" | "blurry" | "partial" | "unclear",
  "confidence": float
}

DERIVATION RULES:

Age category (derive strictly from age):
- kitten: age >= 0 AND age < 1
- junior: age >= 1 AND age < 3
- adult : age >= 3 AND age <= 10
- senior: age > 10
- If age is unknown → set age = 0 and age_category = "kitten"

Size recommendation (derive strictly from chest_cm):
- XS: chest < 28
- S : 28 <= chest < 32
- M : 32 <= chest < 36
- L : 36 <= chest < 40
- XL: chest >= 40

- size_ranges must correspond to the selected size category.
- If a measurement cannot be estimated from image, use null.
- Estimates must reflect realistic domestic cat proportions.
- confidence should reflect image clarity and visibility of full body (0.0–1.0).
"""


# ── Pydantic schema ───────────────────────────────────────────
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
    age: int = 0  # ✅ NOT NULL — default 0 ถ้า model ไม่ส่ง
    gender: int = 0
    weight_kg: float
    chest_cm: float
    neck_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    body_length_cm: Optional[float] = None
    back_length_cm: Optional[float] = None
    leg_length_cm: Optional[float] = None
    body_condition_score: int = Field(..., ge=1, le=9)
    body_condition: str
    body_condition_description: Optional[str] = None
    posture: str = "unknown"
    size_recommendation: Optional[str] = None
    size_ranges: Optional[SizeRanges] = None
    quality_flag: str = "good"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # ✅ auto-cast string → float
    @field_validator("weight_kg", "chest_cm", mode="before")
    @classmethod
    def cast_to_float(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cannot convert '{v}' to float")

    # ✅ clamp body_condition_score ให้อยู่ใน 1–9 เสมอ
    @field_validator("body_condition_score", mode="before")
    @classmethod
    def clamp_bcs(cls, v):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 5  # fallback กลาง ๆ
        return max(1, min(9, v))

    # ✅ แปลง null/None → 0 สำหรับ age
    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v):
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0


# ── Pure helper functions ─────────────────────────────────────

def _to_float(value) -> Optional[float]:
    """Safe cast to float, returns None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calc_bmi(weight_kg: Optional[float], body_length_cm: Optional[float]) -> Optional[float]:
    if not weight_kg or not body_length_cm or body_length_cm <= 0:
        return None
    length_m = body_length_cm / 100.0
    return round(weight_kg / (length_m ** 2), 2)


def _calc_size(chest: Optional[float]) -> str:
    """Deterministic — backend ตัดสิน business rule ไม่ใช่ AI"""
    if chest is None:
        return "M"
    if chest < 28:   return "XS"
    elif chest < 32: return "S"
    elif chest < 36: return "M"
    elif chest < 40: return "L"
    return "XL"


def _calc_age_category(age: int) -> str:
    """Deterministic age_category จาก age — ไม่เชื่อ model"""
    if age < 1:      return "kitten"
    elif age <= 2:   return "junior"
    elif age <= 10:  return "adult"
    return "senior"


def _log_parse_error(raw_text: str, error) -> None:
    """Dump raw response ลง file เพื่อ debug"""
    log_path = f"parse_error_{int(time.time())}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Error: {error}\n\nRaw response:\n{raw_text}")
        logger.error(f"Parse error logged to: {log_path}")
    except Exception as write_err:
        logger.error(f"Failed to write parse error log: {write_err}")


# ── Main ──────────────────────────────────────────────────────

def analyze_cat(image_cat: str) -> dict:
    # ── 1. Download image ─────────────────────────────────────
    print(f"⬇️  Downloading image: {image_cat}")
    try:
        resp = requests.get(image_cat, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot download image: {e}")

    image_bytes = resp.content
    mime_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    print(f"✅ Downloaded ({len(image_bytes)/1024:.1f} KB) | mime={mime_type}")

    # ── 2. เรียก Gemini + Retry ───────────────────────────────
    print(f"🤖 Calling {MODEL}...")
    response = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=CAT_ANALYSIS_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                    safety_settings=SAFETY_SETTINGS,
                    response_mime_type="application/json",  # ✅ บังคับ pure JSON
                ),
            )
            break

        except Exception as e:
            error_str = str(e)
            print(f"❌ Gemini API error (attempt {attempt+1}): {e}")

            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if "limit: 0" in error_str or "PerDay" in error_str:
                    raise RuntimeError("วันนี้ใช้ quota หมดแล้ว กรุณาลองใหม่พรุ่งนี้")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # 5s → 10s → 20s
                    print(f"⏳ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise RuntimeError("Gemini rate limit exceeded. Please try again later.")

            raise RuntimeError(f"Gemini Vision failed: {e}")

    # ✅ guard ถ้า response ยัง None หลัง retry ทั้งหมด
    if response is None:
        raise RuntimeError("Gemini did not return a response after all retries")

    # ── 3. Extract text (with fallback) ──────────────────────
    raw_text = ""
    if hasattr(response, "text") and response.text:
        raw_text = response.text.strip()
    else:
        try:
            raw_text = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            raise RuntimeError("Gemini returned empty response")

    # ✅ guard empty text
    if not raw_text:
        raise RuntimeError("Gemini returned empty text")

    print(f"📝 Gemini raw response:\n{raw_text[:300]}...")

    # strip markdown code block แบบ robust (กัน 2 ชั้น ต่อให้ mime type ไม่ work)
    raw_text = re.sub(r"^```[a-zA-Z]*\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)
    raw_text = raw_text.strip()

    # ── 4. JSON parse (with fallback extract) ────────────────
    try:
        ai_data: dict = json.loads(raw_text)
    except json.JSONDecodeError:
        # fallback: หา JSON block ใน response
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            _log_parse_error(raw_text, "No JSON block found")
            raise RuntimeError("Gemini returned invalid JSON (no JSON block found)")
        try:
            ai_data = json.loads(match.group(0))
        except Exception as e:
            _log_parse_error(raw_text, e)
            raise RuntimeError(f"Gemini returned invalid JSON: {e}")

    # ── 5. ถ้าไม่ใช่แมว ──────────────────────────────────────
    if not ai_data.get("is_cat", True):
        return {
            "is_cat": False,
            "message": ai_data.get("message", "ไม่พบแมวในภาพ"),
        }

    # ── 6. Pydantic validation ────────────────────────────────
    try:
        validated = CatAnalysisSchema(**ai_data)
    except Exception as e:
        _log_parse_error(raw_text, e)
        raise RuntimeError(f"AI response failed schema validation: {e}")

    # ── 7. Business logic — backend ตัดสินใจเอง ──────────────
    chest_cm  = _to_float(validated.chest_cm)
    weight_kg = _to_float(validated.weight_kg)
    body_len  = _to_float(validated.body_length_cm)

    # age รับประกัน int ไม่มี null แล้ว (Pydantic coerce_age จัดการ)
    age: int = validated.age

    size_category = _calc_size(chest_cm)       # ✅ deterministic
    age_category  = _calc_age_category(age)    # ✅ deterministic

    bmi = _calc_bmi(weight_kg, body_len)

    # confidence default 0.5 ถ้า model ไม่ส่ง
    confidence = validated.confidence if validated.confidence is not None else 0.5

    # ── 8. Return ─────────────────────────────────────────────
    result = {
        "is_cat":        True,
        "message":       "ok",
        "cat_color":     validated.cat_color,
        "breed":         validated.breed,
        "age":           age,           # ✅ int เสมอ ไม่มี null
        "gender":        validated.gender,
        "weight_kg":     weight_kg,
        "size_category": size_category,
        "confidence":    confidence,
        "measurements": {
            "chest_cm":       chest_cm,
            "neck_cm":        _to_float(validated.neck_cm),
            "waist_cm":       _to_float(validated.waist_cm),
            "body_length_cm": body_len,
            "back_length_cm": _to_float(validated.back_length_cm),
            "leg_length_cm":  _to_float(validated.leg_length_cm),
        },
        "age_category":               age_category,
        "body_condition_score":       validated.body_condition_score,
        "body_condition":             validated.body_condition,
        "body_condition_description": validated.body_condition_description,
        "bmi":                        bmi,
        "posture":                    validated.posture,
        "size_recommendation":        validated.size_recommendation,
        "size_ranges":                validated.size_ranges.model_dump() if validated.size_ranges else None,
        "quality_flag":               validated.quality_flag,
        "analysis_version":           "2.0",
        "analysis_method":            "gemini_2.5_flash_vision",
    }

    print(
        f"✅ Done: {result['cat_color']} | size={size_category} "
        f"| chest={chest_cm}cm | bmi={bmi} | confidence={confidence} | age={age}"
    )
    return result