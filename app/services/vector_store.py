"""
FAISS vector store management for the RAG feature.

The index itself is built offline by app/data/build_index.py and saved to
disk (FAISS_INDEX_DIR). At query time, app/services/rag_service.py just
loads that saved index — no re-embedding of the corpus on every request.
"""
import os

from langchain_community.vectorstores import FAISS

from app.services.embedding_service import get_embeddings

FAISS_INDEX_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
)

_vector_store = None


def build_faiss_index(documents, index_dir=FAISS_INDEX_DIR):
    """Embed `documents` and persist a FAISS index to `index_dir`. Used by the
    offline ingestion script (app/data/build_index.py), not at request time.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    return vector_store


def load_faiss_index(index_dir=FAISS_INDEX_DIR):
    embeddings = get_embeddings()
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # index was built by us, in build_index.py
    )


def get_vector_store():
    """Lazy singleton — loads the saved FAISS index from disk on first call."""
    global _vector_store
    if _vector_store is None:
        if not os.path.isdir(FAISS_INDEX_DIR):
            raise RuntimeError(
                "FAISS index not found. Run `python -m app.data.build_index` "
                f"first (expected at {FAISS_INDEX_DIR})."
            )
        _vector_store = load_faiss_index()
    return _vector_store
