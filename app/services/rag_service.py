"""
RAG service for medicine-information Q&A.

Pipeline:

GENERAL AI:
question
    -> BM25 + FAISS hybrid retrieval
    -> CrossEncoder reranking
    -> Gemini
    -> answer + sources

MEDICINE-SPECIFIC AI:
question + selected medicine(s)
    -> restrict corpus to selected medicine(s)
    -> BM25 + FAISS retrieval within that corpus
    -> CrossEncoder reranking
    -> Gemini
    -> answer + sources

Important:
- A medicine-specific request MUST NEVER retrieve documents belonging
  to another medicine.
- If the selected medicine has no matching reference documents, return
  a safe "not covered" response.
- The LLM must answer only from retrieved context.
- Side effects must NOT be interpreted as overdose information.
- No diagnosis or specific dosage recommendations.
"""

import os
import pickle
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.data.build_index import DOCUMENTS_PICKLE_PATH
from app.services.reference_service import normalize_medicine_name
from app.services.vector_store import get_vector_store


# ============================================================
# CONFIG
# ============================================================

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FAISS_TOP_K = 8
BM25_TOP_K = 8

ENSEMBLE_WEIGHTS = [0.4, 0.6]

FINAL_CONTEXT_CHUNKS = 4


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a medicine-information assistant embedded in a prescription-OCR app.

You must follow these rules strictly:

1. ANSWER ONLY FROM CONTEXT
- Answer ONLY using information present in the Context section.
- Do NOT use outside medical knowledge.
- Do NOT guess, infer, or fill missing medical information.

2. MEDICINE SCOPE
- If a selected medicine is specified, answer ONLY about that medicine.
- Never introduce information about another medicine.
- Never compare the selected medicine with another medicine unless the
  provided Context explicitly contains that comparison.

3. MISSING INFORMATION
- If the Context does not contain information that answers the question,
  clearly say that the available reference notes do not cover the question.
- Do NOT manufacture an answer from general medical knowledge.

4. OVERDOSE / TOO MUCH MEDICINE
- Do NOT interpret ordinary side-effect information as overdose information.
- If the user asks what happens when too much of a medicine is taken,
  only answer if the Context explicitly contains information about
  overdose, excessive use, toxicity, or taking too much.
- If the Context does not contain such information, say that the available
  reference notes do not provide specific overdose information.

5. MEDICAL SAFETY
- This is informational only.
- Never diagnose a patient.
- Never interpret symptoms as a diagnosis.
- Never recommend a specific dosage.
- Never recommend increasing, decreasing, starting, or stopping a dose.
- If the user asks for a specific dosage or dose change, say that the
  reference notes do not provide a basis for making that decision and
  recommend consulting a licensed doctor or pharmacist.

6. SOURCES
- After answering, briefly identify the medicine and section(s) used.
- Only mention sections actually present in the Context.
- When brand information is available in the Context, you may list
  the verified/recognized brands provided by the reference source.
- Do not claim that a medicine or brand is fake solely because it is
  absent from the listed brands.
- Do not claim that a listed brand is genuine in a particular package
  or pharmacy transaction.
- If a user is concerned that a medicine may be counterfeit or
  suspicious, advise them to verify the product through appropriate
  official regulatory/pharmacy channels.

Context:
{context}
"""


# ============================================================
# LAZY SINGLETONS
# ============================================================

_bm25_retriever = None
_ensemble_retriever = None
_reranker = None
_llm = None


# ============================================================
# DOCUMENT LOADING
# ============================================================

def _load_documents():
    """
    Load the complete parsed medicine-document corpus.
    """

    if not os.path.exists(DOCUMENTS_PICKLE_PATH):
        raise RuntimeError(
            "documents.pkl not found — run "
            "`python -m app.data.build_index` first "
            f"(expected at {DOCUMENTS_PICKLE_PATH})."
        )

    with open(DOCUMENTS_PICKLE_PATH, "rb") as f:
        documents = pickle.load(f)

    return documents


# ============================================================
# TOKENIZER
# ============================================================

def _bm25_tokenize(text):
    """
    Better BM25 tokenizer.

    LangChain's default tokenizer uses simple whitespace splitting.
    This version:
    - lowercases
    - removes punctuation
    - keeps alphanumeric tokens

    Example:

        "What happens to Omcef?"
            ->
        ["what", "happens", "to", "omcef"]
    """

    return re.findall(r"[a-z0-9]+", text.lower())


# ============================================================
# GLOBAL BM25
# ============================================================

def get_bm25_retriever():
    """
    Return the global BM25 retriever.

    Used for general/global questions.
    """

    global _bm25_retriever

    if _bm25_retriever is None:

        documents = _load_documents()

        _bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=_bm25_tokenize,
        )

        _bm25_retriever.k = BM25_TOP_K

    return _bm25_retriever


# ============================================================
# GLOBAL HYBRID RETRIEVER
# ============================================================

def get_ensemble_retriever():
    """
    Global hybrid retriever:

        BM25 + FAISS
    """

    global _ensemble_retriever

    if _ensemble_retriever is None:

        faiss_retriever = get_vector_store().as_retriever(
            search_kwargs={
                "k": FAISS_TOP_K,
            }
        )

        _ensemble_retriever = EnsembleRetriever(
            retrievers=[
                get_bm25_retriever(),
                faiss_retriever,
            ],
            weights=ENSEMBLE_WEIGHTS,
        )

    return _ensemble_retriever


# ============================================================
# MEDICINE-SCOPED DOCUMENT FILTER
# ============================================================

def _get_documents_for_medicines(medicines):
    """
    Return ONLY documents belonging to the requested medicine(s).

    This is the critical protection for the Ask AI button inside
    individual medicine cards.

    Example:

        medicines = ["Omcef"]

    Result:

        Omcef Uses
        Omcef Side Effects

    NOT:

        Thyrox
        Pacimol
        Cali-D
        etc.
    """

    all_documents = _load_documents()

    normalized_targets = set()

    for medicine in medicines:

        if not medicine:
            continue

        normalized = normalize_medicine_name(
            str(medicine).strip(),
            fuzzy_threshold=100,
        )

        if normalized:
            normalized_targets.add(normalized)

    if not normalized_targets:
        return []

    scoped_documents = []

    for document in all_documents:

        document_medicine = document.metadata.get("medicine")

        if not document_medicine:
            continue

        if document_medicine in normalized_targets:
            scoped_documents.append(document)

    return scoped_documents


# ============================================================
# MEDICINE-SCOPED BM25
# ============================================================

def _get_scoped_bm25_retriever(documents):
    """
    Create a BM25 retriever over ONLY the supplied documents.

    This prevents unrelated medicines from entering the retrieval
    pipeline in medicine-specific Ask AI.
    """

    if not documents:
        return None

    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=_bm25_tokenize,
    )

    retriever.k = min(
        BM25_TOP_K,
        len(documents),
    )

    return retriever


# ============================================================
# MEDICINE-SCOPED FAISS RETRIEVAL
# ============================================================

def _scoped_faiss_documents(question, documents):
    """
    Perform semantic retrieval using the existing FAISS store,
    then strictly keep only documents belonging to the selected
    medicine(s).

    Important:
    The global FAISS index may initially return unrelated documents.
    Those are removed before they reach reranking/LLM.

    If the selected medicine's documents are not present in the
    initial FAISS candidates, BM25 retrieval still gives us a
    medicine-scoped lexical path.
    """

    if not documents:
        return []

    allowed_keys = {
        _document_key(document)
        for document in documents
    }

    faiss_retriever = get_vector_store().as_retriever(
        search_kwargs={
            "k": FAISS_TOP_K,
        }
    )

    candidates = faiss_retriever.invoke(question)

    filtered = [
        document
        for document in candidates
        if _document_key(document) in allowed_keys
    ]

    return filtered


# ============================================================
# DOCUMENT IDENTITY
# ============================================================

def _document_key(document):
    """
    Build a stable identity for a document.

    We use medicine + section + content so that documents loaded
    independently from the pickle and FAISS index can still be
    compared safely.
    """

    medicine = str(
        document.metadata.get("medicine", "")
    ).strip().lower()

    section = str(
        document.metadata.get("section", "")
    ).strip().lower()

    content = str(
        document.page_content
    ).strip()

    return (
        medicine,
        section,
        content,
    )


# ============================================================
# STRICT MEDICINE FILTER
# ============================================================

def _filter_by_medicines(documents, medicines):
    """
    Strictly filter documents to the selected medicine(s).

    IMPORTANT:
    There is intentionally NO fallback to the original documents.

    Previously the code did:

        return filtered or documents

    That was the bug.

    If Omcef was selected but no Omcef document appeared in the
    initial retrieval results, the code returned unrelated documents
    such as Thyrox/Pacimol/Cali-D.

    Now an empty result remains empty.
    """

    if not medicines:
        return documents

    normalized_targets = {
        normalize_medicine_name(
            str(name).strip(),
            fuzzy_threshold=100,
        )
        for name in medicines
        if name
    }

    normalized_targets.discard(None)

    return [
        document
        for document in documents
        if document.metadata.get("medicine") in normalized_targets
    ]


# ============================================================
# HYBRID MEDICINE-SCOPED RETRIEVAL
# ============================================================

def _retrieve_for_selected_medicines(question, medicines):
    """
    Medicine-specific retrieval.

    Steps:

        1. Load ONLY selected medicine documents.
        2. BM25 search inside that subset.
        3. FAISS semantic search.
        4. Strictly remove any FAISS result that belongs to another
           medicine.
        5. Combine BM25 + FAISS results.
        6. Deduplicate.

    This preserves the hybrid retrieval architecture while making
    medicine scope a hard boundary.
    """

    scoped_documents = _get_documents_for_medicines(medicines)

    if not scoped_documents:
        return []

    # --------------------------------------------------------
    # BM25 — restricted corpus
    # --------------------------------------------------------

    bm25_retriever = _get_scoped_bm25_retriever(
        scoped_documents
    )

    bm25_documents = []

    if bm25_retriever is not None:
        bm25_documents = bm25_retriever.invoke(question)

    # --------------------------------------------------------
    # FAISS — global semantic search followed by hard filter
    # --------------------------------------------------------

    faiss_documents = _scoped_faiss_documents(
        question,
        scoped_documents,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = []

    seen = set()

    for document in bm25_documents + faiss_documents:

        key = _document_key(document)

        if key in seen:
            continue

        seen.add(key)

        combined.append(document)

    # Final safety filter.
    #
    # Even though BM25 is already scoped and FAISS was filtered,
    # keep this final boundary before reranking.

    combined = _filter_by_medicines(
        combined,
        medicines,
    )

    return combined


# ============================================================
# RERANKER
# ============================================================

def get_reranker():
    """
    Lazy-load CrossEncoder reranker.
    """

    global _reranker

    if _reranker is None:

        _reranker = CrossEncoder(
            RERANKER_MODEL_NAME
        )

    return _reranker


def _rerank(
    question,
    documents,
    top_k=FINAL_CONTEXT_CHUNKS,
):
    """
    CrossEncoder reranking.

    The reranker only sees the documents that survived retrieval.
    """

    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        (
            question,
            document.page_content,
        )
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        document
        for document, _score in ranked[:top_k]
    ]


# ============================================================
# LLM
# ============================================================

def get_llm():
    """
    Lazy-load Gemini.
    """

    global _llm

    if _llm is None:

        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is missing"
            )

        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1,
        )

    return _llm


# ============================================================
# CONTEXT + SOURCES
# ============================================================

def _build_context_and_sources(documents):
    """
    Convert retrieved documents into:
        1. LLM context
        2. frontend source objects
    """

    context_lines = []
    sources = []

    for document in documents:

        medicine = document.metadata.get(
            "medicine",
            "unknown",
        )

        section = document.metadata.get(
            "section",
            "info",
        )

        content = document.page_content

        context_lines.append(
            f"[{medicine} — {section}] {content}"
        )

        sources.append(
            {
                "medicine": medicine,
                "section": section,
                "excerpt": content,
            }
        )

    return (
        "\n\n".join(context_lines),
        sources,
    )


# ============================================================
# LLM ANSWER
# ============================================================

def _generate_answer(question, context):
    """
    Generate the final answer from Gemini.

    Gemini receives ONLY the retrieved context.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    chain = prompt | get_llm()

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    if isinstance(response.content, str):

        return response.content.strip()

    return "".join(
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict)
    ).strip()


# ============================================================
# SAFE FALLBACK
# ============================================================

def _not_covered_response():
    """
    Standard response when the reference corpus does not contain
    the requested information.
    """

    return {
        "answer": (
            "The available reference notes don't cover that question. "
            "I don't want to guess or provide information that isn't "
            "supported by the reference notes. Please consult a "
            "licensed doctor or pharmacist for further information."
        ),
        "sources": [],
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def answer_question(question, medicines=None):
    """
    Main RAG entry point.

    medicines:
        []              -> general/global AI
        ["Omcef"]       -> Omcef-only AI
        ["Omcef", "Azee"] -> Omcef + Azee only
    """

    medicines = medicines or []

    question = (question or "").strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    # ========================================================
    # MEDICINE-SPECIFIC MODE
    # ========================================================

    if medicines:

        candidates = _retrieve_for_selected_medicines(
            question,
            medicines,
        )

    # ========================================================
    # GENERAL MODE
    # ========================================================

    else:

        candidates = get_ensemble_retriever().invoke(
            question
        )

    # ========================================================
    # NO RETRIEVAL RESULT
    # ========================================================

    if not candidates:
        return _not_covered_response()

    # ========================================================
    # FINAL SAFETY FILTER
    # ========================================================

    if medicines:

        candidates = _filter_by_medicines(
            candidates,
            medicines,
        )

        if not candidates:
            return _not_covered_response()

    # ========================================================
    # RERANK
    # ========================================================

    top_documents = _rerank(
        question,
        candidates,
    )

    if not top_documents:
        return _not_covered_response()

    # ========================================================
    # FINAL MEDICINE SAFETY CHECK
    # ========================================================

    if medicines:

        top_documents = _filter_by_medicines(
            top_documents,
            medicines,
        )

        if not top_documents:
            return _not_covered_response()

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context, sources = _build_context_and_sources(
        top_documents
    )

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    answer = _generate_answer(
        question,
        context,
    )

    return {
        "answer": answer,
        "sources": sources,
    }