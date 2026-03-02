# app/services/detect_cat.py
# Gemini 2.0 Flash Lite — ตรวจจับแมว (ประตูกรองก่อนส่ง analysis)
# ใช้ gemini-2.0-flash-lite เพราะ RPD สูง (14,400/วัน) ไม่กิน quota gemini-2.5-flash

import os
import re
import json
import uuid
import time
import logging
import requests

from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Client (ใช้ key เดิมกับ analysis_cat.py) ─────────────────────────────────
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL  = "models/gemini-2.0-flash-lite"   # ← 14,400 RPD free tier

# ── Safety Settings (เหมือน analysis_cat.py) ─────────────────────────────────
SAFETY_SETTINGS = [
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
]

# ── Prompt (สั้น เน้น detect เพียงอย่างเดียว ประหยัด token) ─────────────────
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


# ── JSON Parser (reuse pattern จาก analysis_cat.py) ──────────────────────────

def _parse_json_robust(raw_text: str) -> dict:
    text = raw_text.strip()

    # Step 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 2: strip markdown fence
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: brace-depth extraction
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

    # Step 4: fallback — is_cat false
    if '"is_cat": false' in text or '"is_cat":false' in text:
        return {
            "is_cat": False,
            "is_single": False,
            "is_real_photo": False,
            "reason": "no_cat",
            "confidence": 0.0,
        }

    raise RuntimeError(f"Cannot parse Gemini detect response. Preview: {text[:200]}")


# ── Gemini Caller with Retry ──────────────────────────────────────────────────

def _call_gemini_detect(image_bytes: bytes, mime_type: str) -> str:
    request_id  = str(uuid.uuid4())[:8]
    max_retries = 2          # detect ไม่ต้อง retry เยอะ เน้นเร็ว
    base_wait   = 2

    for attempt in range(max_retries):
        start = time.time()
        try:
            print(f"[detect/{request_id}] 🔍 Attempt {attempt + 1}/{max_retries}")

            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=DETECT_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,          # ต้องการ deterministic
                    max_output_tokens=150,    # schema เล็ก ไม่ต้องการมาก
                    safety_settings=SAFETY_SETTINGS,
                    response_mime_type="application/json",
                ),
            )

            raw_text = ""
            if hasattr(response, "text") and response.text:
                raw_text = response.text.strip()
            else:
                raw_text = response.candidates[0].content.parts[0].text.strip()

            latency = round(time.time() - start, 2)

            if not raw_text:
                raise RuntimeError("Empty response from Gemini detect")

            print(f"[detect/{request_id}] ✅ OK | {len(raw_text)} chars | {latency}s")
            return raw_text

        except Exception as e:
            error_str = str(e)
            latency   = round(time.time() - start, 2)
            print(f"[detect/{request_id}] ❌ Attempt {attempt + 1} failed ({latency}s): {error_str}")

            # Quota หมด → หยุดทันที
            if "limit: 0" in error_str or "PerDay" in error_str:
                raise RuntimeError("detect quota หมดแล้ว กรุณาลองใหม่พรุ่งนี้")

            transient = any(kw in error_str.lower() for kw in [
                "429", "resource_exhausted", "deadline", "timeout", "unavailable",
            ])

            if transient and attempt < max_retries - 1:
                wait = base_wait * (attempt + 1)
                print(f"[detect/{request_id}] ⏳ Retrying in {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Gemini detect failed [{request_id}]: {error_str}")

    raise RuntimeError(f"Gemini detect failed after {max_retries} retries [{request_id}]")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def detect_cat(image_url: str) -> dict:
    """
    ตรวจจับแมวจาก URL
    Returns:
        {
            "is_cat": bool,
            "is_single": bool,
            "is_real_photo": bool,
            "reason": str,        # "passed" | "no_cat" | "multiple_cats" | "is_dog" | "non_cat_animal" | "cartoon" | "other"
            "confidence": float,
            "passed": bool        # True เฉพาะเมื่อ is_cat AND is_single AND is_real_photo
        }
    """
    print(f"🔍 detect_cat: downloading {image_url}")

    # 1. Download image
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cannot download image for detect: {e}")

    image_bytes = resp.content
    mime_type   = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    print(f"✅ Downloaded ({len(image_bytes)/1024:.1f} KB) | mime={mime_type}")

    # 2. Call Gemini Lite
    raw_text = _call_gemini_detect(image_bytes, mime_type)

    # 3. Parse JSON
    try:
        result = _parse_json_robust(raw_text)
    except RuntimeError as e:
        logger.error(f"detect_cat parse error: {e}")
        # fallback: ให้ผ่านไปแล้วให้ analysis_cat ตัดสิน
        return {
            "is_cat": True,
            "is_single": True,
            "is_real_photo": True,
            "reason": "other",
            "confidence": 0.5,
            "passed": True,
        }

    is_cat       = bool(result.get("is_cat", False))
    is_single    = bool(result.get("is_single", True))
    is_real      = bool(result.get("is_real_photo", True))
    reason       = result.get("reason", "other")
    confidence   = float(result.get("confidence", 0.0))

    passed = is_cat and is_single and is_real

    print(
        f"🔍 detect result: passed={passed} | reason={reason} "
        f"| cat={is_cat} single={is_single} real={is_real} | conf={confidence:.2f}"
    )

    return {
        "is_cat":       is_cat,
        "is_single":    is_single,
        "is_real_photo": is_real,
        "reason":       reason,
        "confidence":   confidence,
        "passed":       passed,
    }