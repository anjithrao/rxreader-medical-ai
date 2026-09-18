from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas import NearbyPharmaciesRequest, RouteToPharmacyRequest
from app.services.pharmacy_service import find_nearby_pharmacies, find_route
from app.utils.geo import parse_float, validate_lat_lon

router = APIRouter()


@router.post("/nearby_pharmacies")
async def nearby_pharmacies(payload: NearbyPharmaciesRequest):
    try:
        lat = parse_float(payload.lat, "lat")
        lon = parse_float(payload.lon, "lon")
        radius_m = int(payload.radius_m or 2000)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        validate_lat_lon(lat, lon, "Location")
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    radius_m = max(100, min(radius_m, 10000))

    try:
        pharmacies = find_nearby_pharmacies(lat, lon, radius_m)
    except Exception as exc:
        return JSONResponse({"error": f"Nearby pharmacy search failed: {exc}"}, status_code=502)

    return JSONResponse({"pharmacies": pharmacies, "radius_m": radius_m})


@router.post("/route_to_pharmacy")
async def route_to_pharmacy(payload: RouteToPharmacyRequest):
    start = payload.start or {}
    end = payload.end or {}

    try:
        start_lat = parse_float(start.get("lat"), "start.lat")
        start_lon = parse_float(start.get("lon"), "start.lon")
        end_lat = parse_float(end.get("lat"), "end.lat")
        end_lon = parse_float(end.get("lon"), "end.lon")
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    for label, lat, lon in (
        ("start", start_lat, start_lon),
        ("end", end_lat, end_lon),
    ):
        try:
            validate_lat_lon(lat, lon, label)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        route = find_route(start_lat, start_lon, end_lat, end_lon)
    except Exception as exc:
        return JSONResponse({"error": f"Route search failed: {exc}"}, status_code=502)

    if route is None:
        return JSONResponse({"error": "No route found to the selected medical store"}, status_code=502)

    return JSONResponse(
        {
            "distance_m": route.get("distance"),
            "duration_s": route.get("duration"),
            "geometry": route.get("geometry"),
        }
    )
