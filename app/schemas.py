from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReferenceItem(BaseModel):
    name: str
    uses: str = ""
    side_effects: str = ""


class TranslateReferencesRequest(BaseModel):
    items: List[Dict[str, Any]] = []
    target_language: str = ""


class NearbyPharmaciesRequest(BaseModel):
    lat: Any
    lon: Any
    radius_m: Optional[Any] = None


class LatLon(BaseModel):
    lat: Any
    lon: Any


class RouteToPharmacyRequest(BaseModel):
    start: Dict[str, Any] = {}
    end: Dict[str, Any] = {}


class ChatRequest(BaseModel):
    question: str
    medicines: List[str] = []


class ChatSource(BaseModel):
    medicine: str
    section: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]
