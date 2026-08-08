# An Agentic, Multimodal Framework for Early Alzheimer's Disease Detection via Retinal OCT Images

## 📌 Project Overview

Optical Coherence Tomography (OCT) is a non-invasive imaging modality that produces high-resolution cross-sectional images of retinal layers. The emerging field of oculomics has established that neurodegeneration associated with early Alzheimer's Disease (AD) manifests in the retina, notably via thinning of the Retinal Nerve Fiber Layer (RNFL) and Ganglion Cell-Inner Plexiform Layer (GCIPL).

This project is a multimodal clinical decision-support pipeline that combines a CNN-based visual encoder, Grad-CAM explainability, and a confidence-gated Retrieval-Augmented Generation (RAG) system to fuse visual evidence with patient metadata into a clinician-readable diagnostic rationale — with a chatbot interface for direct interaction.

By replacing opaque "black-box" predictions with an interpretable, confidence-gated agentic workflow, this system is designed to support clinical review rather than replace it: low-confidence cases are routed to human review instead of being auto-synthesized into a report.

---

## 🏗️ System Architecture

```
OCT image ──► CLIP validity check ──► ResNet50 + Grad-CAM ──► confidence gate
                    │ (invalid)              │ (prediction, confidence,          │
                    ▼                        │  attention description)          │
              reject / re-upload             ▼                        ┌─────────┴─────────┐
                                                                 < threshold          ≥ threshold
                                                                       │                    │
                                                              request human review    retrieve from FAISS
                                                                                             │
                                                                                    synthesize rationale
                                                                                  (configurable LLM provider,
                                                                                   raw-context fallback on failure)
```

### Vision subsystem
* A ResNet50 backbone, transfer-learned in two stages: first validated on a general 4-class OCT benchmark (CNV/DME/DRUSEN/NORMAL), then fine-tuned on a disease-specific AD/NORMAL cohort.
* Grad-CAM, applied to the final convolutional block, produces both a visual heatmap and a natural-language description of where the model's attention is concentrated (e.g. "localized to the central region, overlapping the retinal band") — this description feeds directly into the RAG retrieval query.
* The active checkpoint lives at a fixed path (`models/vision_model.pth`); deploying an improved model is a matter of overwriting that one file, no code changes required.
* Before any image reaches the classifier, a zero-shot CLIP check verifies it's actually a retinal OCT scan, rejecting anything else (a photo, a document, an unrelated medical image) — a check the classifier itself cannot perform, since it was only ever trained to distinguish disease states *within* OCT images.

### Agentic RAG subsystem (LangGraph)
* A four-node state graph: evaluate confidence → (retrieve context → synthesize rationale) or (request human review).
* The knowledge base is a local FAISS index built from real literature PDFs (OCT biomarker studies, AD risk-factor reviews), not placeholder text — chunked, embedded, and indexed with source/page metadata so every retrieved passage is citable.
* Every query (both the automatic report-synthesis query and free-form follow-up questions) is checked against the knowledge base's relevance before any LLM call is made — off-topic or prompt-injection-style questions are declined outright, without spending an API call or exposing the model to attacker-controlled context.
* The LLM call itself is prompt-hardened: retrieved context and user input are explicitly delimited and framed as data to analyze, never as instructions to follow.
* If the LLM call fails for any reason (quota, network), both the report-synthesis and follow-up-Q&A paths fall back to showing the raw retrieved context instead of crashing.

### Clinician-facing interface (Streamlit)
Two modes:
* **Demo Case** — a fixed picker between two real, clinically-labeled subjects, for a reliable, deterministic walkthrough.
* **Upload & Chat** — upload any image, pass the CLIP validity gate, then answer the bot's one-at-a-time questions (age, sex, cognitive score) before it runs the full pipeline and shows the report, followed by grounded follow-up chat.

---

## 🛠️ Tools and Technologies
* **Deep Learning:** PyTorch, torchvision, OpenCV (Grad-CAM)
* **Image validation:** CLIP (`transformers`) for zero-shot OCT-scan screening
* **Agentic orchestration:** LangGraph, LangChain
* **Vector database:** FAISS (Facebook AI Similarity Search)
* **LLM providers:** Google Gemini, Groq, or OpenAI — swappable via `config.yaml`, no code changes
* **Interface:** Streamlit
* **Environment/dependency management:** `uv`

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
uv sync
```

### 2. Configure secrets
```bash
cp .env.example .env
```
Then fill in whichever key(s) you need — see `.env.example` for guidance. All keys can be present in `.env` at once; only the provider marked active in `config.yaml` (see below) is actually read.

### 3. Choose an LLM provider
In `config.yaml`, exactly one provider under `llm:` should have `active: true`:
```yaml
llm:
  gemini: {active: false, model: "gemini-2.0-flash", temperature: 0.0}
  groq:   {active: true,  model: "llama-3.3-70b-versatile", temperature: 0.0}
  openai: {active: false, model: "gpt-4o-mini", temperature: 0.0}
```
Startup fails fast with a clear error if zero or more than one provider is active.

### 4. Build the knowledge base
```bash
uv run python3 build_knowledge_base.py
```
This ingests the PDFs listed in `SOURCE_PDFS` (inside `build_knowledge_base.py`) into a local FAISS index at `faiss_index/`. Rerun this whenever you add or change source literature — it's a full rebuild each time, not incremental.

### 5. Add the vision model checkpoint
Place a trained checkpoint at `models/vision_model.pth` (see `models/README.md` for the expected architecture). This file is git-ignored; each collaborator supplies their own.

### 6. Run it

CLI smoke test (runs two demo cases through the full pipeline):
```bash
uv run python3 main.py
```

Full chatbot interface:
```bash
uv run streamlit run app.py
```

---

## ⚙️ Configuration reference (`config.yaml`)

| Key | Purpose |
|---|---|
| `confidence.threshold` | Minimum vision-model confidence required to auto-synthesize a report; below this, the case routes to human review instead. |
| `vision.checkpoint_path` | Path to the active CNN checkpoint. |
| `retrieval.top_k` | Number of literature passages retrieved per query. |
| `retrieval.relevance_distance_threshold` | Max FAISS distance for a query to be treated as in-scope; anything less relevant is declined before any LLM call. |
| `llm.<provider>.active` | Exactly one of `gemini` / `groq` / `openai` should be `true`. |

---

## ⚠️ Current Status & Limitations

This is a research prototype, not a validated clinical tool.

* All disease-specific results are trained and cross-validated on a small, clinically-verified cohort (14 AD, 14 age-matched controls) using subject-level Leave-One-Subject-Out cross-validation — results should be read as pilot-stage evidence, not a generalizable diagnostic claim.
* A structured-feature logistic regression on retinal thickness/reflectivity data is currently the project's strongest validated result. The CNN fine-tune, after a thorough iterative debugging process, currently performs at chance level on this small cohort — a data-scale finding, not an unexplained bug (see project documentation for the full investigation).
* A separate, larger candidate dataset was evaluated and explicitly excluded after being found confounded by scanner/source artifacts rather than genuine pathology — a deliberate validation step, not an oversight.
* Requires prospective, multi-site validation, clinician-in-the-loop evaluation, and regulatory review before any real clinical use.

---
*This repository represents the ongoing fulfillment of the requirements for the Degree of Bachelors in Software Engineering.*
