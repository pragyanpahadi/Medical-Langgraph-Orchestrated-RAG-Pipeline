from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import yaml

from mock_data import retrieve_evidence

# Secrets (e.g. GOOGLE_API_KEY) live in a local, git-ignored .env file rather
# than in config.yaml, which holds non-secret settings and is safe to commit.
load_dotenv()

# --- Load configuration from config.yaml ---
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

CONFIDENCE_THRESHOLD: float = _CONFIG["confidence"]["threshold"]
_TOP_K: int = _CONFIG["retrieval"]["top_k"]
_DB_PATH: str = _CONFIG["retrieval"]["db_path"]
_LLM_MODEL: str = _CONFIG["llm"]["model"]
_LLM_TEMPERATURE: float = _CONFIG["llm"]["temperature"]

class PatientMetadata(TypedDict):
    """
    Structured patient metadata.
    """
    patient_id: str
    scan_id: str
    age: int
    sex: str
    device_model: str
    diagnosis_label: Optional[str]
    cognitive_score: Optional[float]

class RAGState(TypedDict):
    """
    Represents the state of our RAG orchestration graph.
    """
    # 1. Inputs
    metadata: PatientMetadata
    
    # 2. Vision Model Outputs
    visual_features: str       
    cnn_prediction: str        
    cnn_confidence: float      
    
    # 3. Agentic Routing State
    requires_human_review: bool 
    
    # 4. RAG Pipeline
    retrieval_query: str       
    retrieved_context: List[Document] 
    
    # 5. Final Output
    clinical_report: str       

# --- Nodes ---

def evaluate_cnn_confidence(state: RAGState) -> RAGState:
    """
    Router logic node. Evaluates confidence to decide if human review is needed.
    """
    print(f"--- Evaluating CNN Confidence: {state['cnn_confidence']:.0%} ---")
    if state["cnn_confidence"] < CONFIDENCE_THRESHOLD:
        print("-> Confidence below threshold. Flagging for human review.")
        state["requires_human_review"] = True
    else:
        print("-> Confidence meets threshold. Proceeding to retrieval.")
        state["requires_human_review"] = False
    return state

def retrieve_clinical_context(state: RAGState) -> RAGState:
    """
    Retrieves evidence from the local FAISS vector store.
    """
    print("--- Retrieving Clinical Context ---")
    query = f"{state['visual_features']} in a {state['metadata']['age']} year old {state['metadata']['sex']}."
    state["retrieval_query"] = query
    
    docs = retrieve_evidence(query, db_path=_DB_PATH, k=_TOP_K)
    state["retrieved_context"] = docs
    
    return state

def synthesize_rationale(state: RAGState) -> RAGState:
    """
    Synthesizes the final context-aware clinical report using an LLM.
    """
    print("--- Synthesizing Rationale ---")
    
    # Mock fallback if a Gemini API key isn't provided
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Generating a mock report.")
        context_str = "\n".join([doc.page_content for doc in state.get("retrieved_context", [])])
        mock_report = (
            f"**Mock Clinical Rationale Report**\n\n"
            f"**Patient Details:** {state['metadata']['age']}yo {state['metadata']['sex']}\n"
            f"**Vision Module Finding:** {state['cnn_prediction']} (Confidence: {state['cnn_confidence']:.0%})\n\n"
            f"**Supporting Literature Context:**\n{context_str}\n\n"
            f"*Note: Please set GOOGLE_API_KEY (in a local .env file) to generate a real LLM report.*"
        )
        state["clinical_report"] = mock_report
        return state

    llm = ChatGoogleGenerativeAI(model=_LLM_MODEL, temperature=_LLM_TEMPERATURE)
    
    prompt = PromptTemplate.from_template(
        "You are an expert neuro-ophthalmologist AI assistant.\n"
        "Generate a structured, context-aware clinical rationale report.\n\n"
        "Patient Metadata:\nAge: {age}\nSex: {sex}\nCognitive Score: {cog_score}\n\n"
        "Vision Model Output:\nPrediction: {prediction}\nConfidence: {confidence}\nVisual Features: {visual_features}\n\n"
        "Retrieved Literature Context:\n{context}\n\n"
        "Provide a concise summary linking the vision model's findings with the literature context and patient metadata. "
        "Do not make a definitive diagnosis, summarize the evidence for the clinician."
    )
    
    context_str = "\n".join([doc.page_content for doc in state.get("retrieved_context", [])])
    
    formatted_prompt = prompt.format(
        age=state["metadata"]["age"],
        sex=state["metadata"]["sex"],
        cog_score=state["metadata"].get("cognitive_score", "N/A"),
        prediction=state["cnn_prediction"],
        confidence=f"{state['cnn_confidence']:.2f}",
        visual_features=state["visual_features"],
        context=context_str
    )
    
    response = llm.invoke(formatted_prompt)
    state["clinical_report"] = response.content
    
    return state

def request_human_review(state: RAGState) -> RAGState:
    """
    Handles cases where auto-synthesis is aborted due to low confidence.
    """
    print("--- Requesting Human Review ---")
    state["clinical_report"] = "AUTO-SYNTHESIS ABORTED: Vision model confidence below threshold. Manual human review requested."
    return state

# --- Edges & Graph Compilation ---

def route_based_on_confidence(state: RAGState):
    if state.get("requires_human_review"):
        return "request_human_review"
    return "retrieve_clinical_context"

def build_graph() -> StateGraph:
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("evaluate_cnn_confidence", evaluate_cnn_confidence)
    workflow.add_node("retrieve_clinical_context", retrieve_clinical_context)
    workflow.add_node("synthesize_rationale", synthesize_rationale)
    workflow.add_node("request_human_review", request_human_review)
    
    # Define edges
    workflow.set_entry_point("evaluate_cnn_confidence")
    
    # Conditional routing after evaluating confidence
    workflow.add_conditional_edges(
        "evaluate_cnn_confidence",
        route_based_on_confidence,
        {
            "request_human_review": "request_human_review",
            "retrieve_clinical_context": "retrieve_clinical_context"
        }
    )
    
    workflow.add_edge("retrieve_clinical_context", "synthesize_rationale")
    workflow.add_edge("synthesize_rationale", END)
    workflow.add_edge("request_human_review", END)
    
    return workflow.compile()
