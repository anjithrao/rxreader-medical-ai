from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest
from app.services.rag_service import answer_question

router = APIRouter()


@router.post("/chat")
async def chat(payload: ChatRequest):
    question = (payload.question or "").strip()
    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    try:
        result = answer_question(question, medicines=payload.medicines)
    except Exception as exc:
        return JSONResponse({"error": f"Chat failed: {exc}"}, status_code=502)

    return JSONResponse(result)
