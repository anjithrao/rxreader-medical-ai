import requests

from app.utils.geo import haversine_distance_m, build_pharmacy_address


def find_nearby_pharmacies(lat, lon, radius_m):
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
      way["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
      relation["amenity"="pharmacy"](around:{radius_m},{lat},{lon});
      node["shop"="chemist"](around:{radius_m},{lat},{lon});
      way["shop"="chemist"](around:{radius_m},{lat},{lon});
      relation["shop"="chemist"](around:{radius_m},{lat},{lon});
      node["healthcare"="pharmacy"](around:{radius_m},{lat},{lon});
      way["healthcare"="pharmacy"](around:{radius_m},{lat},{lon});
      relation["healthcare"="pharmacy"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """.strip()

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": overpass_query},
        headers={"User-Agent": "PrescriptionOCR/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    pharmacies = []
    seen = set()
    for element in data.get("elements", []):
        tags = element.get("tags") or {}
        element_lat = element.get("lat")
        element_lon = element.get("lon")
        if element_lat is None or element_lon is None:
            center = element.get("center") or {}
            element_lat = center.get("lat")
            element_lon = center.get("lon")
        if element_lat is None or element_lon is None:
            continue

        pharmacy_id = f"{element.get('type', 'item')}/{element.get('id')}"
        if pharmacy_id in seen:
            continue
        seen.add(pharmacy_id)

        distance_m = haversine_distance_m(lat, lon, float(element_lat), float(element_lon))
        pharmacies.append(
            {
                "id": pharmacy_id,
                "name": tags.get("name") or "Medical store",
                "lat": float(element_lat),
                "lon": float(element_lon),
                "address": build_pharmacy_address(tags),
                "distance_m": round(distance_m),
                "tags": tags,
            }
        )

    pharmacies.sort(key=lambda pharmacy: pharmacy["distance_m"])
    return pharmacies[:25]


def find_route(start_lat, start_lon, end_lat, end_lon):
    coordinates = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    response = requests.get(
        f"https://router.project-osrm.org/route/v1/driving/{coordinates}",
        params={"overview": "full", "geometries": "geojson"},
        headers={"User-Agent": "PrescriptionOCR/1.0"},
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    routes = data.get("routes") or []
    if not routes:
        return None
    return routes[0]
