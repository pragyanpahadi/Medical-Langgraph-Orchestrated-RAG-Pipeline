"""Builds the real FAISS knowledge base from actual literature PDFs, replacing
mock_data.MOCK_LITERATURE. Run this once (and again whenever you add papers):

    uv run python3 build_knowledge_base.py

Source PDFs are referenced by absolute path since they live in ~/Downloads,
outside this repo — add more entries to SOURCE_PDFS as you gather more papers.
"""
import os

import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from mock_data import get_embeddings

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_DB_PATH = _CONFIG["retrieval"]["db_path"]

SOURCE_PDFS = [
    "/Users/susanadhikari/Downloads/Alzheimer's OCT Research Paper Guidance.pdf",
    "/Users/susanadhikari/Downloads/s41746-024-01109-5.pdf",
]


def load_and_chunk() -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for pdf_path in SOURCE_PDFS:
        if not os.path.exists(pdf_path):
            print(f"[skip] not found: {pdf_path}")
            continue
        print(f"Loading {pdf_path}...")
        pages = PyPDFLoader(pdf_path).load()
        chunks = splitter.split_documents(pages)
        source_name = os.path.basename(pdf_path)
        for chunk in chunks:
            chunk.metadata["source"] = source_name
            # PyPDFLoader already sets metadata["page"] (0-indexed) per chunk
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
    print("Embedding and building FAISS index (this may take a minute)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving to '{_DB_PATH}'...")
    vectorstore.save_local(_DB_PATH)
    print("Done! The real literature knowledge base replaces the mock one.")


if __name__ == "__main__":
    main()
