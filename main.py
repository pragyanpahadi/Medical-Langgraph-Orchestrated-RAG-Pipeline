from rag_pipeline import build_graph, PatientMetadata
from mock_data import setup_mock_vectorstore
import os

# Real Dryad subjects whose OCT scans live in sample_images/ — age is the
# midpoint of the dataset's binned age range (e.g. "68_through_77" -> 72),
# since the source data reports age as a range rather than a single value.
DEMO_CASES = [
    {
        "image_path": "sample_images/AD362_DARK_00.png",
        "metadata": PatientMetadata(
            patient_id="AD362",
            scan_id="AD362_DARK_00",
            age=72,
            sex="Female",
            device_model="Dryad OCT (raw ANALYZE stack)",
            diagnosis_label="AD (clinically diagnosed, per Dryad cohort)",
            cognitive_score=None,
        ),
    },
    {
        "image_path": "sample_images/CO091_DARK_00.png",
        "metadata": PatientMetadata(
            patient_id="CO091",
            scan_id="CO091_DARK_00",
            age=72,
            sex="Male",
            device_model="Dryad OCT (raw ANALYZE stack)",
            diagnosis_label="CONTROL (age-matched, per Dryad cohort)",
            cognitive_score=None,
        ),
    },
]


def build_state_from_image(image_path: str, metadata: PatientMetadata) -> dict:
    from vision_module import run_inference

    vision_output = run_inference(image_path)
    return {
        "metadata": metadata,
        "visual_features": vision_output["visual_features"],
        "cnn_prediction": vision_output["cnn_prediction"],
        "cnn_confidence": vision_output["cnn_confidence"],
        "requires_human_review": False,
        "retrieval_query": "",
        "retrieved_context": [],
        "clinical_report": "",
    }


def main():
    print("Setting up local vectorstore for RAG...")
    if not os.path.exists("faiss_index"):
        setup_mock_vectorstore()

    print("\nCompiling the LangGraph Agentic Pipeline...")
    app = build_graph()

    for case in DEMO_CASES:
        print("\n" + "=" * 50)
        print(f"--- CASE: {case['metadata']['patient_id']} ---")
        print("=" * 50)
        try:
            state = build_state_from_image(case["image_path"], case["metadata"])
        except FileNotFoundError as e:
            print(f"[!] Vision model not available yet: {e}")
            print("Skipping real inference for this case — see models/README.md.")
            continue

        print(f"Vision output: {state['cnn_prediction']} "
              f"(confidence: {state['cnn_confidence']:.2%})")
        result = app.invoke(state)
        print("\nFINAL OUTPUT:")
        print(result.get("clinical_report"))


if __name__ == "__main__":
    main()
