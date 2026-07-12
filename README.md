# An Agentic, Multimodal Framework for Early Alzheimer's Disease Detection via Retinal OCT Images

## 📌 Project Overview
Optical Coherence Tomography (OCT) is a non-invasive imaging modality that produces high-resolution cross-sectional images of retinal layers. The emerging field of oculomics has established that neurodegeneration associated with early Alzheimer's Disease (AD) manifests in the retina, notably via thinning of the Retinal Nerve Fiber Layer (RNFL) and Ganglion Cell-Inner Plexiform Layer (GCIPL).

This project proposes a novel, multimodal clinical decision support system that combines optimized Convolutional Neural Networks (CNNs) for baseline feature extraction, Grad-CAM for anatomical explainability, and a Retrieval-Augmented Generation (RAG) pipeline to fuse visual data with patient metadata. 

By replacing opaque "black-box" models with an interpretable, confidence-gated agentic workflow, this system is designed to seamlessly integrate into and support clinical workflows.

---

## 🏗️ System Architecture & Project Structure

The project is structured into distinct developmental phases to ensure modularity, explainability, and rigorous evaluation:

### Phase 1: Data Handling and Preprocessing
* **Dataset:** Utilizes standard retinal OCT datasets (CNV, DME, DRUSEN, NORMAL) and Alzheimer's-specific OCT cohorts.
* **Pipeline:** Images are resized, optionally denoised, and normalized using channel statistics before feeding into the vision encoder. Patient metadata (age, sex, cognitive scores, etc.) is standardized and de-identified.

### Phase 2: Vision Subsystem (CNN & Explainability)
* **Model:** A ResNet50-based transfer-learning classifier acts as the core visual learner. 
* **Explainability:** Grad-CAM is applied to the final convolutional layers to generate anatomically grounded saliency maps (highlighting RNFL/GCIPL regions) to verify predictions clinically.
* **Output:** Produces a 1D feature vector, a class prediction, and a model confidence score.

### Phase 3: Multimodal Fusion & Agentic Reporting (RAG)
* **Vector Store:** FAISS is used as a local vector database to index mock medical literature and clinical guidelines.
* **Agentic Orchestration:** LangGraph dynamically controls the flow of information based on the Vision Subsystem's confidence.
* **Synthesis:** LangChain and Large Language Models (LLMs) synthesize context-aware diagnostic rationales.

---

## ⚙️ Step-by-Step Procedure Flow

Our LangGraph implementation mirrors a real-world clinical validation scheme. Rather than generating reports for every input, the system employs an **agentic gating mechanism**:

1. **Ingest Multimodal Request:** The system receives an OCT Image and structured Patient Metadata.
2. **Extract Visual Features:** The Vision Subsystem evaluates the OCT scan, returning visual findings, a prediction, and a confidence percentage.
3. **Evaluate CNN Confidence (The Router):** 
   * **Confidence ≥ Threshold (default 80%, configurable via `CNN_CONFIDENCE_THRESHOLD`):** The system proceeds automatically.
   * **Confidence < Threshold:** The system aborts auto-synthesis and routes to **Request Human Review**, notifying the clinician that manual inspection is required.
4. **Retrieve Clinical Context:** For high-confidence cases, the visual features and patient context are used to build a query against the FAISS vector database to fetch relevant literature.
5. **Synthesize Rationale:** An LLM processes the visual findings, retrieved literature, and patient metadata to generate a readable, context-aware diagnostic report for the clinician.

---

## 🛠️ Tools and Technologies
* **Deep Learning:** PyTorch, torchvision
* **Agentic Frameworks:** LangChain, LangGraph
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM Integration:** Google Gemini (free tier) via `langchain-google-genai`
* **Data Manipulation:** NumPy, Pandas

---

## 🚀 Getting Started

### Prerequisites
Make sure your virtual environment has the required dependencies installed (see `pyproject.toml`):
```bash
pip install langchain-core langgraph langchain-google-genai langchain-community faiss-cpu sentence-transformers python-dotenv pyyaml
```

Then copy `.env.example` to `.env` and add your free [Google AI Studio](https://aistudio.google.com/app/apikey) API key:
```bash
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=...
```
`.env` is git-ignored and never committed — it's the only place secrets live. Non-secret tunables (confidence threshold, FAISS `top_k`, Gemini model name, temperature) live in `config.yaml`, which is safe to commit.

### Running the RAG Pipeline
The primary testing script demonstrates the LangGraph agentic workflow utilizing mock data.
```bash
python main.py
```
*Note: To run the LLM synthesis node effectively, set `GOOGLE_API_KEY` in your `.env` file. If it's not found, the system safely falls back to printing a mock report for testing purposes.*

*Note: The confidence routing threshold (default 80%), FAISS `top_k`, and the Gemini model/temperature are all configured in `config.yaml`.*

*Note: `faiss_index/` is a locally generated, git-ignored artifact — it's rebuilt automatically from mock literature on first run and does not need to be committed.*

---
*This repository represents the ongoing fulfillment of the requirements for the Degree of Bachelors in Software Engineering.*
