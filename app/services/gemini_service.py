import base64
import json

import requests

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.models.registry import registry
from app.services.reference_service import normalize_medicine_name, resolve_reference_info
from app.utils.text import build_buy_links


def build_gemini_prompt(easyocr_lines, easy_hints, trocr_reads):
    easy_lines_text = "\n".join(f"- {line}" for line in easyocr_lines[:40]) or "- none"
    easy_hints_text = "\n".join(f"- {hint}" for hint in easy_hints[:20]) or "- none"
    trocr_text = "\n".join(f"- {text}" for text in trocr_reads[:20]) or "- none"
    medicine_text = ", ".join(registry.known_medicines)

    return f"""
You are extracting medicine names from a handwritten prescription image.

Important rules:
- Use the attached prescription image as the primary source.
- EasyOCR and TrOCR text are only hints and may contain mistakes.
- Return only medicine names, not dosage, schedule, quantity, or instructions.
- Prefer normalized names from the known medicine list when there is a close match.
- Remove duplicates.
- If you are unsure, keep only high-probability medicine names.
- For medicines from the known medicine list, leave uses and side_effects empty because the app has local reference notes.
- For medicines outside the known medicine list, provide a short common uses summary and a short common side-effects summary.
- Keep uses and side effects concise and factual.

Known medicine list:
{medicine_text}

EasyOCR full-text lines:
{easy_lines_text}

EasyOCR medicine hints:
{easy_hints_text}

TrOCR region reads:
{trocr_text}
""".strip()


def extract_response_text(response_json):
    candidates = response_json.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if part.get("text")]
    return "".join(texts).strip()


def call_gemini_for_medicines(image_bytes, easyocr_lines, easy_hints, trocr_reads):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")

    prompt = build_gemini_prompt(easyocr_lines, easy_hints, trocr_reads)
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "medicines": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "name": {"type": "STRING"},
                                "uses": {"type": "STRING"},
                                "side_effects": {"type": "STRING"},
                            },
                            "required": ["name", "uses", "side_effects"],
                        },
                    }
                },
                "required": ["medicines"],
            },
        },
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Gemini request failed with status {response.status_code}")

    response_json = response.json()
    response_text = extract_response_text(response_json)
    if not response_text:
        raise RuntimeError("Gemini returned an empty response")

    parsed = json.loads(response_text)
    raw_medicines = parsed.get("medicines", [])
    if not isinstance(raw_medicines, list):
        raise RuntimeError("Gemini returned an invalid medicines list")

    normalized = []
    seen = set()
    for medicine in raw_medicines:
        if not isinstance(medicine, dict):
            continue
        normalized_name = normalize_medicine_name(medicine.get("name", ""))
        if not normalized_name or normalized_name in seen:
            continue
        seen.add(normalized_name)
        normalized.append(
            {
                "name": normalized_name,
                "uses": medicine.get("uses", "").strip(),
                "side_effects": medicine.get("side_effects", "").strip(),
            }
        )

    return normalized


def build_ai_medicine_results(medicine_items):
    results = []
    for medicine in medicine_items:
        medicine_name = medicine["name"]
        results.append(
            {
                "prefix": "",
                "easy_ocr": "",
                "raw_ocr": "",
                "corrected": medicine_name,
                "confidence": None,
                "status": "extracted",
                "badge_text": "Whole-image AI",
                "badge_class": "badge-green",
                "detail_note": "",
                "reference_info": resolve_reference_info(medicine_name, preferred_info=medicine),
                "buy_links": build_buy_links(medicine_name),
            }
        )
    return results


def merge_medicine_results(local_medicines, gemini_medicine_items):
    merged = []
    seen = set()
    gemini_map = {
        normalize_medicine_name(medicine["name"], fuzzy_threshold=100): medicine
        for medicine in gemini_medicine_items
        if medicine.get("name")
    }

    for medicine in local_medicines:
        normalized_name = normalize_medicine_name(medicine.get("corrected", ""), fuzzy_threshold=100)
        medicine_copy = dict(medicine)
        gemini_match = gemini_map.get(normalized_name)
        if gemini_match:
            medicine_copy["detail_note"] = "Also confirmed by the whole-image AI pass."
        medicine_copy["reference_info"] = resolve_reference_info(
            medicine_copy.get("corrected", ""),
            preferred_info=gemini_match,
            existing_info=medicine_copy.get("reference_info"),
        )
        merged.append(medicine_copy)
        seen.add(normalized_name)

    gemini_only = [
        medicine
        for medicine in gemini_medicine_items
        if normalize_medicine_name(medicine["name"], fuzzy_threshold=100) not in seen
    ]
    merged.extend(build_ai_medicine_results(gemini_only))
    return merged


def call_gemini_for_reference_translation(items, target_language):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing")

    prompt = (
        f"Translate the uses and side_effects fields into {target_language}. "
        "Keep each medicine name unchanged. Keep the same order. "
        "Use concise natural language for patients in India. "
        "Return strict JSON only."
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"text": json.dumps({"items": items}, ensure_ascii=False)},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "items": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "name": {"type": "STRING"},
                                "uses": {"type": "STRING"},
                                "side_effects": {"type": "STRING"},
                            },
                            "required": ["name", "uses", "side_effects"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Gemini translation failed with status {response.status_code}")

    response_json = response.json()
    response_text = extract_response_text(response_json)
    if not response_text:
        raise RuntimeError("Gemini translation returned an empty response")

    parsed = json.loads(response_text)
    translated_items = parsed.get("items", [])
    if not isinstance(translated_items, list):
        raise RuntimeError("Gemini translation returned an invalid items list")

    normalized = []
    for item in translated_items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "name": normalize_medicine_name(item.get("name", ""), fuzzy_threshold=100),
                "uses": item.get("uses", "").strip(),
                "side_effects": item.get("side_effects", "").strip(),
            }
        )
    return normalized
