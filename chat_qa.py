"""Conversational follow-up Q&A grounded in the same FAISS knowledge base used
by the one-shot report pipeline in rag_pipeline.py. Used by the Streamlit chat
interface for free-form questions after a report has been generated.
"""
import os

import yaml
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from mock_data import retrieve_evidence
from llm_provider import get_llm, has_api_key

load_dotenv()

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_TOP_K = _CONFIG["retrieval"]["top_k"]
_DB_PATH = _CONFIG["retrieval"]["db_path"]

_QA_PROMPT = PromptTemplate.from_template(
    "You are an assistant helping a clinician understand an AI-assisted retinal "
    "OCT screening result for Alzheimer's disease risk. Answer the clinician's "
    "question using ONLY the retrieved literature context and the current "
    "case's report below. If the context doesn't contain the answer, say so "
    "plainly rather than guessing. Keep answers concise and cite the source "
    "document/page when you use a retrieved fact.\n\n"
    "Current Case Report:\n{case_report}\n\n"
    "Retrieved Literature Context:\n{context}\n\n"
    "Conversation so far:\n{history}\n\n"
    "Clinician's Question: {question}\n\n"
    "Answer:"
)


def answer_question(question: str, case_report: str = "", history: list[tuple[str, str]] | None = None) -> dict:
    """Returns {"answer": str, "sources": list[dict]}. history is a list of
    (question, answer) tuples from earlier turns in this conversation."""
    history = history or []
    docs = retrieve_evidence(question, db_path=_DB_PATH, k=_TOP_K)
    context_str = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    )
    history_str = "\n".join(f"Q: {q}\nA: {a}" for q, a in history) or "(none yet)"

    sources = [
        {"source": d.metadata.get("source", "unknown"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]

    if not has_api_key():
        mock_answer = (
            f"**Mock answer** (set the active provider's API key for a real response)\n\n"
            f"Retrieved context:\n{context_str}"
        )
        return {"answer": mock_answer, "sources": sources}

    try:
        llm = get_llm()
        prompt = _QA_PROMPT.format(
            case_report=case_report or "(no active case report)",
            context=context_str,
            history=history_str,
            question=question,
        )
        response = llm.invoke(prompt)
        return {"answer": response.content, "sources": sources}
    except Exception as e:
        # Never let a live demo crash on an API hiccup (quota, network, etc.) —
        # fall back to showing the raw retrieved context instead.
        fallback_answer = (
            f"**LLM call failed ({type(e).__name__}), showing retrieved context directly:**\n\n"
            f"{context_str}"
        )
        return {"answer": fallback_answer, "sources": sources}
