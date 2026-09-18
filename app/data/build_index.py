"""
Ingestion script for the RAG feature.

Parses every *.txt file in app/data/medicine_docs/ (same "Name / Uses: /
Side effects:" format your info_txt.txt already uses — you can literally
drop your existing info_txt.txt in here too) into one LangChain Document per
medicine per section (uses / side_effects), so retrieval and citations can
point at a specific fact instead of a whole medicine block.

It then:
  1. Embeds the documents and saves a FAISS index to app/data/faiss_index/.
  2. Pickles the same document list to app/data/documents.pkl, so
     app/services/rag_service.py can build a matching BM25 retriever at
     runtime without re-parsing the source files.

Run once, and again whenever medicine_docs/ changes:

    python -m app.data.build_index
"""
import os
import pickle

from langchain_core.documents import Document

from app.services.vector_store import build_faiss_index, FAISS_INDEX_DIR

MEDICINE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "medicine_docs")
DOCUMENTS_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "documents.pkl")


def parse_reference_notes(path):
    documents = []
    current_name = ""
    current_uses = ""
    current_side_effects = ""

    def commit():
        if not current_name:
            return
        if current_uses:
            documents.append(
                Document(
                    page_content=f"{current_name} — Uses: {current_uses}",
                    metadata={
                        "medicine": current_name.lower(),
                        "section": "uses",
                        "source": os.path.basename(path),
                    },
                )
            )
        if current_side_effects:
            documents.append(
                Document(
                    page_content=f"{current_name} — Side effects: {current_side_effects}",
                    metadata={
                        "medicine": current_name.lower(),
                        "section": "side_effects",
                        "source": os.path.basename(path),
                    },
                )
            )

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("uses:"):
                current_uses = line.split(":", 1)[1].strip()
                continue
            if line.lower().startswith("side effects:"):
                current_side_effects = line.split(":", 1)[1].strip()
                continue

            # A new non-"Uses:"/"Side effects:" line starts a new medicine block.
            commit()
            current_name = line
            current_uses = ""
            current_side_effects = ""

    commit()
    return documents


def load_all_documents():
    if not os.path.isdir(MEDICINE_DOCS_DIR):
        raise RuntimeError(f"No medicine_docs directory found at {MEDICINE_DOCS_DIR}")

    documents = []
    for filename in sorted(os.listdir(MEDICINE_DOCS_DIR)):
        if not filename.endswith(".txt"):
            continue
        documents.extend(parse_reference_notes(os.path.join(MEDICINE_DOCS_DIR, filename)))
    return documents


def main():
    documents = load_all_documents()
    if not documents:
        raise RuntimeError(f"No documents parsed — check {MEDICINE_DOCS_DIR}/*.txt")

    print(f"Parsed {len(documents)} chunks from {MEDICINE_DOCS_DIR}")

    build_faiss_index(documents)
    print(f"FAISS index saved to {FAISS_INDEX_DIR}")

    with open(DOCUMENTS_PICKLE_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"Document list pickled to {DOCUMENTS_PICKLE_PATH} (used for the BM25 retriever)")


if __name__ == "__main__":
    main()
