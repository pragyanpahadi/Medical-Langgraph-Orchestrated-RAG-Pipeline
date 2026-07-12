from rag_pipeline import build_graph, PatientMetadata
from mock_data import setup_mock_vectorstore
import os

def main():
    print("Setting up local vectorstore for RAG...")
    # Ensure the vector store exists
    if not os.path.exists("faiss_index"):
        setup_mock_vectorstore()
        
    print("\nCompiling the LangGraph Agentic Pipeline...")
    app = build_graph()
    
    print("\n" + "="*50)
    print("--- TEST CASE 1: High Confidence (Auto-Synthesis) ---")
    print("="*50)
    high_conf_state = {
        "metadata": PatientMetadata(
            patient_id="PT123",
            scan_id="SCN001",
            age=72,
            sex="Female",
            device_model="Cirrus HD-OCT",
            diagnosis_label="Unknown",
            cognitive_score=22.5
        ),
        "visual_features": "Significant thinning in the temporal RNFL and overall GCIPL volume reduction.",
        "cnn_prediction": "Alzheimer's Risk Indicator Positive",
        "cnn_confidence": 0.92,
        "requires_human_review": False,
        "retrieval_query": "",
        "retrieved_context": [],
        "clinical_report": ""
    }
    
    result = app.invoke(high_conf_state)
    print("\nFINAL OUTPUT (High Confidence):")
    print(result.get("clinical_report"))
    
    print("\n" + "="*50)
    print("--- TEST CASE 2: Low Confidence (Human Review) ---")
    print("="*50)
    low_conf_state = {
        "metadata": PatientMetadata(
            patient_id="PT124",
            scan_id="SCN002",
            age=65,
            sex="Male",
            device_model="Spectralis OCT",
            diagnosis_label="Unknown",
            cognitive_score=28.0
        ),
        "visual_features": "Mild artifact present. Slight RNFL thinning observed, but inconclusive.",
        "cnn_prediction": "Alzheimer's Risk Indicator Positive",
        "cnn_confidence": 0.65,  # Below the 0.80 threshold
        "requires_human_review": False,
        "retrieval_query": "",
        "retrieved_context": [],
        "clinical_report": ""
    }
    
    result2 = app.invoke(low_conf_state)
    print("\nFINAL OUTPUT (Low Confidence):")
    print(result2.get("clinical_report"))

if __name__ == "__main__":
    main()
