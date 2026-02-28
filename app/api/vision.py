"""
app/api/vision.py
POST /vision/analyze-cat

Flow:
  Flutter (ML Kit detect cat ✅)
    → ส่ง image_cat มาที่ Backend
    → Gemini 1.5 Flash วิเคราะห์ขนาด/สี/สายพันธุ์
    → INSERT ครบทุก column ใน table cat
    → คืน JSON กลับ Flutter
"""

import json
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from app.auth.dependencies import verify_firebase_token
from app.services.analysis_cat import analyze_cat   # ← Gemini 1.5 Flash
from app.db.database import get_db_pool

router = APIRouter()


class AnalyzeCatRequest(BaseModel):
    image_cat: str


@router.post("/vision/analyze-cat", response_model=dict)
async def analyze_cat_endpoint(
    request: AnalyzeCatRequest,
    user: dict = Depends(verify_firebase_token),
):
    firebase_uid = user.get("firebase_uid")
    if not firebase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token",
        )

    print(f"\n🐱 analyze-cat | user={firebase_uid[:8]}*** | url={request.image_cat}")

    try:
        # ── STEP 1: Gemini 1.5 Flash วิเคราะห์โดยตรง ──────────
        # ML Kit detect ผ่านมาจาก Flutter แล้ว ไม่ต้อง detect ซ้ำ
        print("\n--- STEP 1: Gemini Vision Analysis ---")
        analysis = analyze_cat(image_cat=request.image_cat)

        # Gemini double-check ว่าไม่ใช่แมว
        if not analysis.get("is_cat", True):
            return {
                "is_cat": False,
                "message": analysis.get("message", "😿 ไม่พบแมวในภาพ"),
            }

        measurements = analysis.get("measurements", {})

        # ── STEP 2: INSERT ครบทุก column ──────────────────────
        print("\n--- STEP 2: Saving to Database ---")
        pool = await get_db_pool()

        async with pool.acquire() as conn:
            cat_id = await conn.fetchval(
                """
                INSERT INTO cat (
                    firebase_uid,
                    cat_color, breed, age, gender,
                    weight, size_category,
                    chest_cm, neck_cm, waist_cm,
                    body_length_cm, back_length_cm, leg_length_cm,
                    confidence, bounding_box,
                    image_cat, thumbnail_url,
                    age_category,
                    body_condition_score, body_condition, body_condition_description,
                    bmi,
                    posture,
                    size_recommendation,
                    size_ranges,
                    quality_flag,
                    analysis_version, analysis_method,
                    detected_at, updated_at
                ) VALUES (
                    $1,
                    $2,  $3,  $4,  $5,
                    $6,  $7,
                    $8,  $9,  $10,
                    $11, $12, $13,
                    $14, $15::jsonb,
                    $16, $17,
                    $18,
                    $19, $20, $21,
                    $22,
                    $23,
                    $24,
                    $25::jsonb,
                    $26,
                    $27, $28,
                    $29, $30
                ) RETURNING id
                """,
                firebase_uid,                                                           # $1
                analysis.get("cat_color", "Unknown"),                                  # $2
                analysis.get("breed"),                                                 # $3
                analysis.get("age"),                                                   # $4
                analysis.get("gender", 0),                                             # $5
                _f(analysis.get("weight_kg")),                                         # $6
                analysis.get("size_category", "M"),                                    # $7
                _f(measurements.get("chest_cm")),                                      # $8
                _f(measurements.get("neck_cm")),                                       # $9
                _f(measurements.get("waist_cm")),                                      # $10
                _f(measurements.get("body_length_cm")),                                # $11
                _f(measurements.get("back_length_cm")),                                # $12
                _f(measurements.get("leg_length_cm")),                                 # $13
                _f(analysis.get("confidence", 0.90)),                                  # $14
                json.dumps(analysis.get("bounding_box", [])),                          # $15 jsonb
                request.image_cat,                                                     # $16
                None,                                                                  # $17 thumbnail
                analysis.get("age_category", "adult"),                                 # $18
                analysis.get("body_condition_score"),                                  # $19
                analysis.get("body_condition"),                                        # $20
                analysis.get("body_condition_description"),                            # $21
                analysis.get("bmi"),                                                   # $22
                analysis.get("posture"),                                               # $23
                analysis.get("size_recommendation"),                                   # $24
                json.dumps(analysis.get("size_ranges")) if analysis.get("size_ranges") else None,  # $25
                analysis.get("quality_flag", "good"),                                  # $26
                analysis.get("analysis_version", "2.0"),                               # $27
                analysis.get("analysis_method", "gemini_1.5_flash_vision"),            # $28
                datetime.utcnow(),                                                     # $29
                datetime.utcnow(),                                                     # $30
            )

        print(f"✅ Saved → cat.id={cat_id}")

        # ── STEP 3: Response กลับ Flutter ─────────────────────
        return {
            # ── หลัก (ตรงกับ CatData.fromJson ใน Flutter) ──
            "is_cat":         True,
            "message":        "✅ วิเคราะห์แมวสำเร็จ!",
            "name":           analysis.get("cat_color", "Unknown"),
            "breed":          analysis.get("breed"),
            "age":            analysis.get("age"),
            "weight":         _f(analysis.get("weight_kg")) or 0.0,
            "size_category":  analysis.get("size_category", "M"),
            "chest_cm":       _f(measurements.get("chest_cm")) or 0.0,
            "neck_cm":        _f(measurements.get("neck_cm")),
            "body_length_cm": _f(measurements.get("body_length_cm")),
            "confidence":     _f(analysis.get("confidence", 0.90)),
            "bounding_box":   analysis.get("bounding_box", []),
            "image_cat":      request.image_cat,
            "thumbnail_url":  None,
            "detected_at":    datetime.utcnow().isoformat() + "Z",

            # ── extra fields ──
            "db_id":                      cat_id,
            "gender":                     analysis.get("gender", 0),
            "age_category":               analysis.get("age_category"),
            "body_condition":             analysis.get("body_condition"),
            "body_condition_score":       analysis.get("body_condition_score"),
            "body_condition_description": analysis.get("body_condition_description"),
            "bmi":                        analysis.get("bmi"),
            "waist_cm":                   _f(measurements.get("waist_cm")),
            "back_length_cm":             _f(measurements.get("back_length_cm")),
            "leg_length_cm":              _f(measurements.get("leg_length_cm")),
            "posture":                    analysis.get("posture"),
            "size_recommendation":        analysis.get("size_recommendation"),
            "size_ranges":                analysis.get("size_ranges"),
            "quality_flag":               analysis.get("quality_flag"),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


def _f(value) -> float | None:
    """แปลงเป็น float หรือคืน None"""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None