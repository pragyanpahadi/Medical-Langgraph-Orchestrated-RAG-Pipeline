"""Builds the real FAISS knowledge base from actual literature PDFs, replacing
mock_data.MOCK_LITERATURE. Run this once (and again whenever you add papers):

    uv run python3 build_knowledge_base.py

Source PDFs are referenced by absolute path since they live in ~/Downloads,
outside this repo — add more entries to SOURCE_PDFS as you gather more papers.
"""
import os
import re
import sys

import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

from mock_data import get_embeddings

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_DB_PATH = _CONFIG["retrieval"]["db_path"]

SOURCE_PDFS = [
    "/Users/susanadhikari/Documents/RAG_Pipeline/data/OCTBiomarkersForAD.pdf",
    "/Users/susanadhikari/Documents/RAG_Pipeline/data/RiskFactorsForAD.pdf",
    "/Users/susanadhikari/Documents/RAG_Pipeline/data/octabiomarkersad.pdf",
    "/Users/susanadhikari/Documents/RAG_Pipeline/data/OCTADmoreinfo.pdf"
]

# Manual overrides, keyed by filename that is checked before auto-detection. Use
# this if a paper's DOI can't be found on its first two pages, or if
# auto-detection ever picks up the wrong one (e.g. a DOI cited by the paper
# rather than its own
SOURCE_DOIS = {}

# A DOI is "10.<4-9 digit registrant>/<suffix>". Real papers print their own
# DOI on page 1 or 2, often as "doi:10.xxxx/yyyy" or "https://doi.org/10.xxxx/yyyy".
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# Same idea as SOURCE_DOIS, for a paper's display title — checked before
# PDF-metadata detection. A PDF's embedded metadata title is sometimes an
# internal manuscript ID (e.g. "EB_A_235238 1..20") rather than the real
# title, so this is also where you fix that if it ever happens.
SOURCE_TITLES = {
    "OCTBiomarkersForAD.pdf": "Retinal biomarkers for early Alzheimer's detection: a systematic review of optical coherence tomography (OCT) findings",
    "OCTADmoreinfo.pdf": "Optical Coherence Tomography in Patients with Alzheimer's Disease: What Can It Tell Us?",
}


def _detect_doi(pages: list) -> str | None:
    """Scans the first two pages' (whitespace-normalized) text for a DOI."""
    text = " ".join(p.page_content for p in pages[:2])
    match = _DOI_PATTERN.search(text)
    if not match:
        return None
    # Trim trailing punctuation a sentence boundary can accidentally capture.
    doi = match.group(0).rstrip(".,;)")
    # A DOI's numeric article-id suffix can get visually wrapped onto the
    # next line by a PDF's layout (e.g. "10.3390/brainsci 14010041"). Only
    # treat a following digit run as that wrapped remainder if the suffix
    # matched so far has no digits at all — a suffix that already contains
    # digits (e.g. "bmjophth-2025-002328") is a complete DOI on its own, and
    # a bare number right after it (a footnote marker, a page number) is not
    # part of it.
    suffix = doi.split("/", 1)[1] if "/" in doi else ""
    if not any(ch.isdigit() for ch in suffix):
        continuation = re.match(r"^ (\d+)\b", text[match.end():])
        if continuation:
            doi += continuation.group(1)
    return doi


def _resolve_doi(source_name: str, pages: list) -> str | None:
    if source_name in SOURCE_DOIS:
        return SOURCE_DOIS[source_name] or None  # explicit "" opts out

    detected = _detect_doi(pages)
    if detected:
        print(f"  -> detected DOI: {detected}")
        return detected

    if sys.stdin.isatty():
        entered = input(
            f"  No DOI found in '{source_name}'. Paste one now, or press "
            f"Enter to skip: "
        ).strip()
        return entered or None

    print(f"  -> no DOI found for '{source_name}' and not running "
          f"interactively; leaving it unset.")
    return None


def _looks_like_real_title(title: str | None) -> bool:
    if not title:
        return False
    title = title.strip()
    # Rejects internal manuscript IDs (e.g. "EB_A_235238 1..20") and other
    # non-title metadata: real titles are prose, several words long, and
    # don't contain underscores or ".." page-range markers.
    return len(title) >= 15 and title.count(" ") >= 3 and "_" not in title and ".." not in title


def _resolve_title(pdf_path: str, source_name: str) -> str:
    if source_name in SOURCE_TITLES:
        return SOURCE_TITLES[source_name]

    try:
        pdf_title = PdfReader(pdf_path).metadata.title
    except Exception:
        pdf_title = None

    if _looks_like_real_title(pdf_title):
        print(f"  -> detected title (PDF metadata): {pdf_title}")
        return pdf_title

    if sys.stdin.isatty():
        entered = input(
            f"  No reliable title found in '{source_name}'. Paste the "
            f"paper's title, or press Enter to use the filename instead: "
        ).strip()
        return entered or source_name

    print(f"  -> no reliable title found for '{source_name}' and not "
          f"running interactively; using the filename instead.")
    return source_name


def load_and_chunk() -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for pdf_path in SOURCE_PDFS:
        if not os.path.exists(pdf_path):
            print(f"[skip] not found: {pdf_path}")
            continue
        print(f"Loading {pdf_path}...")
        pages = PyPDFLoader(pdf_path).load()
        # Some PDFs extract with a stray newline/space between nearly every
        # word (a per-glyph line-break artifact from how the PDF was
        # produced). Collapse all whitespace before chunking so retrieved
        # text renders as normal prose instead of one word per line.
        for page in pages:
            page.page_content = re.sub(r"\s+", " ", page.page_content).strip()
        chunks = splitter.split_documents(pages)
        source_name = os.path.basename(pdf_path)
        doi = _resolve_doi(source_name, pages)
        title = _resolve_title(pdf_path, source_name)
        for chunk in chunks:
            chunk.metadata["source"] = source_name
            chunk.metadata["title"] = title
            # PyPDFLoader already sets metadata["page"] (0-indexed) per chunk
            if doi:
                chunk.metadata["doi"] = doi
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")
    return all_chunks


def main():
    chunks = load_and_chunk()
    if not chunks:
        print("No source PDFs found — check SOURCE_PDFS paths. Aborting.")
        return

    print(f"\nTotal chunks: {len(chunks)}")
    embeddings = get_embeddings()
    print("Embedding and building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving to '{_DB_PATH}'...")
    vectorstore.save_local(_DB_PATH)
    print("Done! The Knowledge Base has been updated")


if __name__ == "__main__":
    main()
