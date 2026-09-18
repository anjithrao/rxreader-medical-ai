import math


def parse_float(value, field_name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number")
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a valid number")
    return number


def haversine_distance_m(lat1, lon1, lat2, lon2):
    radius_m = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_pharmacy_address(tags):
    address_parts = []
    for key in ("addr:housenumber", "addr:street", "addr:suburb", "addr:city", "addr:postcode"):
        value = (tags.get(key) or "").strip()
        if value and value not in address_parts:
            address_parts.append(value)
    return ", ".join(address_parts)


def validate_lat_lon(lat, lon, label="location"):
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(f"{label} is outside valid latitude/longitude bounds")
