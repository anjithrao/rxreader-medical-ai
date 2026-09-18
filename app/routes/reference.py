from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas import TranslateReferencesRequest
from app.services.gemini_service import call_gemini_for_reference_translation

router = APIRouter()


@router.post("/translate_references")
async def translate_references(payload: TranslateReferencesRequest):
    items = payload.items
    target_language = (payload.target_language or "").strip()

    if not items:
        return JSONResponse({"error": "No items to translate"}, status_code=400)
    if not target_language:
        return JSONResponse({"error": "No target language provided"}, status_code=400)
    if target_language.lower() == "english":
        return JSONResponse({"items": items, "target_language": "English"})

    try:
        translated_items = call_gemini_for_reference_translation(items, target_language)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)

    return JSONResponse({"items": translated_items, "target_language": target_language})
