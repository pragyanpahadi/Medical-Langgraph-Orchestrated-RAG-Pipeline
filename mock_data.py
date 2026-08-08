import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Mock literature based on the mid-term report (e.g., related to RNFL, GCIPL, and AD)
MOCK_LITERATURE = [
    Document(
        page_content="Thinning of the Retinal Nerve Fiber Layer (RNFL) is strongly associated with early cognitive decline and Alzheimer's disease.",
        metadata={"source": "Ko et al. 2018", "type": "clinical_study"}
    ),
    Document(
        page_content="Ganglion Cell-Inner Plexiform Layer (GCIPL) atrophy is a reliable surrogate biomarker for neurodegeneration in dementia patients.",
        metadata={"source": "Mutlu et al. 2018", "type": "clinical_study"}
    ),
    Document(
        page_content="Optical Coherence Tomography (OCT) allows non-invasive measurement of retinal layers, providing low-cost screening for AD.",
        metadata={"source": "Wagner et al. 2020", "type": "review_article"}
    ),
    Document(
        page_content="Patients with severe Alzheimer's show significant retinal thinning compared to age-matched healthy controls.",
        metadata={"source": "Clinical Guidelines 2025", "type": "guideline"}
    )
]

_embeddings = None

def get_embeddings():
    """
    Returns a cached HuggingFaceEmbeddings instance, loading it from disk only once.
    """
    global _embeddings
    if _embeddings is None:
        print("Initializing embedding model...")
        # Using a local embedding model to avoid API key requirements for testing
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def setup_mock_vectorstore(db_path: str = "faiss_index"):
    """
    Initializes a FAISS vector store with mock medical literature.
    """
    embeddings = get_embeddings()

    print("Creating FAISS index from mock literature...")
    vectorstore = FAISS.from_documents(MOCK_LITERATURE, embeddings)
    
    print(f"Saving vectorstore locally to '{db_path}'...")
    vectorstore.save_local(db_path)
    print("Done!")
    return vectorstore

def retrieve_evidence(query: str, db_path: str = "faiss_index", k: int = 2):
    """
    Retrieves the top-k most relevant documents for a given query.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def retrieve_evidence_with_scores(query: str, db_path: str = "faiss_index", k: int = 2):
    """
    Same as retrieve_evidence, but also returns each document's raw FAISS L2
    distance (lower = more relevant). Used to gate whether a query is even
    in-scope for this knowledge base before spending an LLM call on it —
    empirically, in-scope OCT/AD questions score ~0.7-0.9 against this index,
    while off-topic questions score ~1.6-1.9.
    """
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore.similarity_search_with_score(query, k=k)

if __name__ == "__main__":
    # Test setting up and querying the vector store
    setup_mock_vectorstore()
    print("\n--- Testing Retrieval ---")
    results = retrieve_evidence("retinal nerve fiber layer")
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res.page_content}")
