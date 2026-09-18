import io

import torch

from app.models.registry import registry, correct
from app.utils.text import is_prefix, is_dosage, build_buy_links
from app.services.reference_service import get_fallback_reference_info


def recognize(crop_pil):
    pixel_values = registry.processor(
        crop_pil.convert("RGB"),
        return_tensors="pt",
    ).pixel_values
    if torch.cuda.is_available():
        pixel_values = pixel_values.cuda()
    with torch.no_grad():
        output = registry.model.generate(pixel_values, max_new_tokens=32)
    return registry.processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()


def detect_medicine_regions(img, sorted_results):
    medicine_regions = []

    for i, (bbox, text, conf) in enumerate(sorted_results):
        if not is_prefix(text):
            continue

        y1_current = min(point[1] for point in bbox)
        y2_current = max(point[1] for point in bbox)
        x1_current = min(point[0] for point in bbox)

        for bbox2, text2, conf2 in sorted_results[i + 1:]:
            x1_next = min(point[0] for point in bbox2)
            y1_next = min(point[1] for point in bbox2)
            y2_next = max(point[1] for point in bbox2)
            overlap = min(y2_current, y2_next) - max(y1_current, y1_next)
            height = max(y2_current - y1_current, y2_next - y1_next)

            if overlap > height * 0.3 and x1_next > x1_current and not is_dosage(text2):
                x1 = max(0, int(min(point[0] for point in bbox2)) - 4)
                y1 = max(0, int(min(point[1] for point in bbox2)) - 4)
                x2 = min(img.width, int(max(point[0] for point in bbox2)) + 4)
                y2 = min(img.height, int(max(point[1] for point in bbox2)) + 4)
                medicine_regions.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "easy_text": text2,
                        "prefix": text,
                    }
                )
                break

    if not medicine_regions:
        for bbox, text, conf in sorted_results:
            if len(text) > 3 and not is_dosage(text) and not is_prefix(text):
                x1 = max(0, int(min(point[0] for point in bbox)) - 4)
                y1 = max(0, int(min(point[1] for point in bbox)) - 4)
                x2 = min(img.width, int(max(point[0] for point in bbox)) + 4)
                y2 = min(img.height, int(max(point[1] for point in bbox)) + 4)
                medicine_regions.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "easy_text": text,
                        "prefix": "?",
                    }
                )

    return medicine_regions


def extract_local_medicines(img, medicine_regions):
    medicines = []
    trocr_reads = []
    easy_hints = []
    seen = set()

    for region in medicine_regions:
        crop = img.crop(region["bbox"])
        raw = recognize(crop)
        trocr_reads.append(raw)
        easy_hints.append(region["easy_text"])
        corrected, score, status = correct(raw)

        if corrected in seen:
            continue

        seen.add(corrected)
        medicines.append(
            {
                "prefix": region["prefix"],
                "easy_ocr": region["easy_text"],
                "raw_ocr": raw,
                "corrected": corrected,
                "confidence": round(score, 1),
                "status": status,
                "reference_info": get_fallback_reference_info(corrected),
                "buy_links": build_buy_links(corrected),
            }
        )

    return medicines, trocr_reads, easy_hints


def build_gemini_image_bytes(image, max_size=(1600, 1600)):
    resized = image.copy()
    resized.thumbnail(max_size)
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue()
