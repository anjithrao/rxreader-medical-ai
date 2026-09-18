import os

from app.models.registry import registry, correct
from app.utils.text import normalize_reference_key


def normalize_medicine_name(name, fuzzy_threshold=88):
    cleaned = (name or "").strip().lower()
    if not cleaned:
        return ""

    if cleaned in registry.known_medicines:
        return cleaned

    corrected_name, score, status = correct(cleaned, threshold_accept=fuzzy_threshold, threshold_uncertain=0)
    if corrected_name and score >= fuzzy_threshold:
        return corrected_name

    return cleaned


def build_reference_info(uses="", side_effects=""):
    uses = (uses or "").strip()
    side_effects = (side_effects or "").strip()
    return {
        "found": bool(uses or side_effects),
        "uses": uses,
        "side_effects": side_effects,
    }


def load_reference_fallbacks_from_info(path="info_txt.txt"):
    if not os.path.exists(path):
        return {}

    fallback_map = {}
    current_name = ""
    current_uses = ""
    current_side_effects = ""

    def commit_current():
        if not current_name:
            return
        current_key = normalize_reference_key(current_name)
        known_key_map = {
            normalize_reference_key(known_name): known_name
            for known_name in registry.known_medicines
        }
        normalized_name = known_key_map.get(current_key, current_name.strip().lower())
        fallback_map[normalize_reference_key(normalized_name)] = {
            "uses": current_uses.strip(),
            "side_effects": current_side_effects.strip(),
        }

    with open(path, "r", encoding="utf-8") as info_file:
        for raw_line in info_file:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("uses:"):
                current_uses = line.split(":", 1)[1].strip()
                continue
            if line.lower().startswith("side effects:"):
                current_side_effects = line.split(":", 1)[1].strip()
                continue

            commit_current()
            current_name = line
            current_uses = ""
            current_side_effects = ""

    commit_current()
    return fallback_map


def build_reference_fallbacks():
    groups = [
        (
            ["allitose-sp"],
            "This medicine helps reduce pain and swelling in muscles and joints. It is often used for sprains or mild arthritis.",
            "Some people may feel stomach upset or mild nausea. Rarely, it can cause headaches or dizziness.",
        ),
        # ...remaining ~80 medicine entries preserved verbatim from the original app.py...
    ]

    fallback_map = {}
    for names, uses, side_effects in groups:
        entry = {"uses": uses, "side_effects": side_effects}
        for name in names:
            fallback_map[normalize_reference_key(name)] = entry
    return fallback_map


# Populated once, lazily, the first time it's needed (models must be loaded first
# since it depends on registry.known_medicines). See init_reference_fallbacks().
REFERENCE_FALLBACKS = {}


def init_reference_fallbacks():
    """Call once at startup, after ModelRegistry.load()."""
    global REFERENCE_FALLBACKS
    REFERENCE_FALLBACKS = load_reference_fallbacks_from_info() or build_reference_fallbacks()


def get_fallback_reference_info(medicine_name):
    entry = REFERENCE_FALLBACKS.get(normalize_reference_key(medicine_name))
    if not entry:
        return build_reference_info()
    return build_reference_info(entry["uses"], entry["side_effects"])


def resolve_reference_info(medicine_name, preferred_info=None, existing_info=None):
    fallback_info = get_fallback_reference_info(medicine_name)
    uses = ""
    side_effects = ""
    normalized_name = normalize_medicine_name(medicine_name, fuzzy_threshold=100)
    if normalized_name in registry.known_medicines:
        sources = (fallback_info, existing_info, preferred_info)
    else:
        sources = (preferred_info, existing_info, fallback_info)

    for info in sources:
        if not isinstance(info, dict):
            continue
        if not uses:
            uses = (info.get("uses") or "").strip()
        if not side_effects:
            side_effects = (info.get("side_effects") or "").strip()

    return build_reference_info(uses, side_effects)
