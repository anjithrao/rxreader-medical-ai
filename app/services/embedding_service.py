"""
Singleton for the embedding model used by the RAG feature.

Loaded lazily, on first use, so that the OCR-only endpoints (/process,
/translate_references, /nearby_pharmacies, /route_to_pharmacy) don't pay the
cost of downloading/loading an embedding model if nobody ever calls /chat.
"""
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            # bge models are trained for cosine similarity on normalized vectors.
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
