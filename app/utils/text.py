import re
from urllib.parse import quote_plus

from app.config import MEDICINE_PREFIXES


def is_prefix(text):
    normalized_prefixes = [prefix.rstrip(".") for prefix in MEDICINE_PREFIXES]
    return text.lower().strip().rstrip(".") in normalized_prefixes


def is_dosage(text):
    patterns = [
        r"\d+mg",
        r"\d+ml",
        r"\d+-\d+-\d+",
        r"\b(od|bd|tds|qid|sos)\b",
        r"^\d+$",
    ]
    return any(re.search(pattern, text.lower()) for pattern in patterns)


def normalize_reference_key(name):
    return "".join(char for char in (name or "").lower() if char.isalnum())


def group_easyocr_lines(sorted_results, y_tolerance=20):
    lines = []
    for bbox, text, conf in sorted_results:
        y_center = sum(point[1] for point in bbox) / len(bbox)
        x_left = min(point[0] for point in bbox)

        if not lines or abs(y_center - lines[-1]["y_center"]) > y_tolerance:
            lines.append({"y_center": y_center, "tokens": [(x_left, text)]})
        else:
            lines[-1]["tokens"].append((x_left, text))

    grouped = []
    for line in lines:
        ordered_tokens = [text for _, text in sorted(line["tokens"], key=lambda item: item[0])]
        grouped.append(" ".join(ordered_tokens))
    return grouped


def build_buy_links(medicine_name):
    search_targets = [
        ("Tata 1mg", "www.1mg.com"),
        ("Apollo Pharmacy", "www.apollopharmacy.in"),
        ("PharmEasy", "pharmeasy.in"),
    ]
    return [
        {
            "label": label,
            "url": f"https://www.google.com/search?q={quote_plus(f'{medicine_name} site:{domain}')}",
        }
        for label, domain in search_targets
    ]
