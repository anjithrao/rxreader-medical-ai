from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.models.registry import registry
from app.services.reference_service import init_reference_fallbacks
from app.routes import pages, ocr, reference, pharmacy, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load EasyOCR + TrOCR + medicine dictionary exactly once, at startup.
    registry.load()
    # Depends on registry.known_medicines, so must run after registry.load().
    init_reference_fallbacks()
    yield


app = FastAPI(title="Prescription OCR API", lifespan=lifespan)

# Serves the built React frontend's static assets (js/css/images), same role
# `static/` played next to Flask's `templates/index.html`.
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(pages.router)
app.include_router(ocr.router)
app.include_router(reference.router)
app.include_router(pharmacy.router)
app.include_router(chat.router)

# NOTE on startup cost: the RAG feature (embeddings, FAISS index, BM25 corpus,
# cross-encoder reranker, Gemini client) is intentionally NOT loaded here.
# app/services/rag_service.py and app/services/vector_store.py lazy-load on
# the first /chat request, so OCR-only cold starts stay as fast as before.
# The FAISS index itself is pre-built offline (see app/data/build_index.py)
# and just read from disk on that first call — nothing is re-embedded at
# request time.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
