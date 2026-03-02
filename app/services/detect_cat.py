# app/services/detect_cat.py
# รับ base64 จาก Flutter — auto fallback ถ้า model quota หมด

import os
import re
import json
import uuid
import time
import base64
import logging

from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY_DETECT"])

# fallback order — แต่ละตัว quota แยกกัน
# ❌ ไม่ใส่ gemini-2.5-flash เพราะใช้ใน analysis แล้ว
# DETECT_MODELS = [
#     "models/gemini-2.5-flash-lite",  # ถูก เร็ว quota แยก
#     "models/gemini-2.0-flash",       # fallback
#     "models/gemini-2.0-flash-lite",  # fallback สุดท้าย
# ]

DETECT_MODELS = "models/gemini-2.5-flash"

# ── Safety Settings ───────────────────────────────────────────────────────────
SAFETY_SETTINGS = [
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
]

# ── Prompt ────────────────────────────────────────────────────────────────────
DETECT_PROMPT = """
You are a cat image validator. Analyze the image carefully and respond ONLY with raw JSON.
No markdown. No explanation. No extra text.

Schema (return exactly this):
{
  "is_cat": boolean,
  "is_single": boolean,
  "is_real_photo": boolean,
  "reason": string,
  "confidence": number
}

Rules:
- "is_cat": true ONLY if there is a real domestic cat visible (NOT lion/tiger/cheetah/wildcat)
- "is_single": true ONLY if exactly 1 cat is visible in the entire image
- "is_real_photo": true = real photograph | false = cartoon/anime/drawing/plush/figurine/3D render/toy
- "reason": MUST be one of these exact strings:
    "passed"         → single real cat, real photo ✅
    "no_cat"         → no cat found in image
    "multiple_cats"  → 2 or more cats detected
    "is_dog"         → dog detected (not a cat)
    "non_cat_animal" → other animal (rabbit, bird, hamster, etc.)
    "cartoon"        → cartoon / drawing / toy / not a real photo
    "other"          → cannot determine clearly
- "confidence": float 0.0-1.0 how confident you are in the result
"""


# ── JSON Parser ───────────────────────────────────────────────────────────────

def _parse_json_robust(raw_text: str) -> dict:
    text = raw_text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find('{')
    if start != -1:
        depth, end = 0, -1
        in_string, escape_next = False, False
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
            except json.JSONDecodeError:
                pass

    if '"is_cat": false' in text or '"is_cat":false' in text:
        return {
            "is_cat": False,
            "is_single": False,
            "is_real_photo": False,
            "reason": "no_cat",
            "confidence": 0.0,
        }

    raise RuntimeError(f"Cannot parse Gemini detect response. Preview: {text[:200]}")


# ── Quota error check ─────────────────────────────────────────────────────────

def _is_quota_error(error_str: str) -> bool:
    return (
        "limit: 0" in error_str
        or "PerDay" in error_str
        or "RESOURCE_EXHAUSTED" in error_str
        or "resource_exhausted" in error_str.lower()
        or ("429" in error_str and "quota" in error_str.lower())
    )


# ── Gemini Caller (auto fallback to next model on quota error) ────────────────

def _call_gemini_detect(image_bytes: bytes, mime_type: str) -> str:
    request_id = str(uuid.uuid4())[:8]

    for model in DETECT_MODELS:
        print(f"[detect/{request_id}] 🔍 Trying: {model}")
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=DETECT_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=150,
                    safety_settings=SAFETY_SETTINGS,
                    response_mime_type="application/json",
                ),
            )

            raw_text = ""
            if hasattr(response, "text") and response.text:
                raw_text = response.text.strip()
            else:
                raw_text = response.candidates[0].content.parts[0].text.strip()

            if not raw_text:
                raise RuntimeError("Empty response from Gemini detect")

            print(f"[detect/{request_id}] ✅ OK | model={model} | {len(raw_text)} chars")
            return raw_text

        except Exception as e:
            error_str = str(e)
            print(f"[detect/{request_id}] ❌ {model} failed: {error_str[:150]}")

            if _is_quota_error(error_str):
                # quota หมด → ลอง model ถัดไปทันที
                print(f"[detect/{request_id}] ⚠️ Quota exhausted → trying next model...")
                continue

            # error อื่น (network, auth, etc.) → raise ทันที
            raise RuntimeError(f"Gemini detect failed [{request_id}]: {error_str}")

    # ทุก model quota หมด
    raise RuntimeError("detect quota หมดทุก model กรุณาลองใหม่พรุ่งนี้")


# ── Helper: build result dict ─────────────────────────────────────────────────

def _build_result(raw_text: str) -> dict:
    try:
        result = _parse_json_robust(raw_text)
    except RuntimeError as e:
        logger.error(f"detect parse error: {e}")
        return {
            "is_cat": True, "is_single": True, "is_real_photo": True,
            "reason": "other", "confidence": 0.5, "passed": True,
        }

    is_cat     = bool(result.get("is_cat", False))
    is_single  = bool(result.get("is_single", True))
    is_real    = bool(result.get("is_real_photo", True))
    reason     = result.get("reason", "other")
    confidence = float(result.get("confidence", 0.0))
    passed     = is_cat and is_single and is_real

    print(
        f"🔍 detect result: passed={passed} | reason={reason} "
        f"| cat={is_cat} single={is_single} real={is_real} | conf={confidence:.2f}"
    )

    return {
        "is_cat":        is_cat,
        "is_single":     is_single,
        "is_real_photo": is_real,
        "reason":        reason,
        "confidence":    confidence,
        "passed":        passed,
    }


# ── Main Entry Points ─────────────────────────────────────────────────────────

def detect_cat_base64(image_base64: str, mime_type: str = "image/jpeg") -> dict:
    """รับ base64 จาก Flutter โดยตรง"""
    print(f"🔍 detect_cat_base64: mime={mime_type} | size={len(image_base64)} chars")

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as e:
        raise RuntimeError(f"Cannot decode base64 image: {e}")

    print(f"✅ Decoded ({len(image_bytes)/1024:.1f} KB)")
    raw_text = _call_gemini_detect(image_bytes, mime_type)
    return _build_result(raw_text)


def detect_cat(image_url: str) -> dict:
    """Legacy: รับ URL"""
    import requests
    print(f"🔍 detect_cat: downloading {image_url}")
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot download image for detect: {e}")

    image_bytes = resp.content
    mime_type   = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    print(f"✅ Downloaded ({len(image_bytes)/1024:.1f} KB) | mime={mime_type}")

    raw_text = _call_gemini_detect(image_bytes, mime_type)
    return _build_result(raw_text)