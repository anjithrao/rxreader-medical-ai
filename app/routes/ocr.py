import os
import tempfile

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

from app.models.registry import registry
from app.services.ocr_service import (
    detect_medicine_regions,
    extract_local_medicines,
    build_gemini_image_bytes,
)
from app.services.gemini_service import call_gemini_for_medicines, merge_medicine_results
from app.utils.text import group_easyocr_lines

router = APIRouter()


@router.post("/process")
async def process(image: UploadFile = File(None)):
    if image is None:
        return JSONResponse({"error": "No image"}, status_code=400)

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        img.save(temp_path, format="PNG")
        results = registry.reader.readtext(temp_path)
        image_bytes = build_gemini_image_bytes(img)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    sorted_results = sorted(
        results,
        key=lambda item: (min(point[1] for point in item[0]), min(point[0] for point in item[0])),
    )
    easyocr_lines = group_easyocr_lines(sorted_results)
    medicine_regions = detect_medicine_regions(img, sorted_results)
    local_medicines, trocr_reads, easy_hints = extract_local_medicines(img, medicine_regions)

    medicines = local_medicines
    engine = "local_ocr"
    gemini_error = ""

    try:
        gemini_medicines = call_gemini_for_medicines(
            image_bytes=image_bytes,
            easyocr_lines=easyocr_lines,
            easy_hints=easy_hints,
            trocr_reads=trocr_reads,
        )
        if gemini_medicines:
            medicines = merge_medicine_results(local_medicines, gemini_medicines)
            engine = "hybrid_ocr"
    except Exception as exc:
        gemini_error = str(exc)

    return JSONResponse(
        {
            "medicines": medicines,
            "total": len(medicines),
            "engine": engine,
            "gemini_error": gemini_error,
        }
    )
