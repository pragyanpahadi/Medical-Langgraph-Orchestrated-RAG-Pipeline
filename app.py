"""Streamlit demo with two modes:
- Demo Case: pick a pre-loaded real Dryad scan, run the pipeline, chat about the report.
- Upload & Chat: upload any image, validated as a real OCT scan via CLIP, then
  the bot asks for patient details one at a time before running screening.

Run with: uv run streamlit run app.py
"""
import os
import tempfile

import streamlit as st

from main import DEMO_CASES, build_state_from_image
from mock_data import setup_mock_vectorstore
from rag_pipeline import build_graph, PatientMetadata
from chat_qa import answer_question
from oct_validity_check import is_valid_oct_image
from conversation_flow import parse_age, parse_sex, parse_cognitive_score

st.set_page_config(page_title="Alzheimer's OCT Screening Assistant", page_icon="🧠", layout="wide")


@st.cache_resource
def get_graph():
    if not os.path.exists("faiss_index"):
        setup_mock_vectorstore()
    return build_graph()


def run_case(image_path: str, metadata: PatientMetadata) -> dict:
    app = get_graph()
    state = build_state_from_image(image_path, metadata)
    result = app.invoke(state)
    return {**state, **result}


def render_report_and_chat(result: dict, chat_key: str):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Vision Model Output")
        badge = "🔴" if "Positive" in result["cnn_prediction"] else "🟢"
        st.metric(label="Prediction", value=f"{badge} {result['cnn_prediction']}")
        st.progress(result["cnn_confidence"], text=f"Confidence: {result['cnn_confidence']:.1%}")
        st.caption(result["visual_features"])
    with col2:
        st.subheader("Routing Decision")
        if result["requires_human_review"]:
            st.warning("⚠️ Confidence below threshold — routed to human review.")
        else:
            st.success("✅ Confidence meets threshold — auto-synthesized report below.")

    st.subheader("Clinical Rationale Report")
    st.markdown(result["clinical_report"])

    st.divider()
    st.subheader("💬 Ask a follow-up question")
    st.caption("Answers are grounded in the same literature knowledge base used for the report above.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    st.session_state.chat_history.setdefault(chat_key, [])

    for q, a in st.session_state.chat_history[chat_key]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    question = st.chat_input("e.g. What does GC-IPL thinning mean?", key=f"qa_input_{chat_key}")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.spinner("Retrieving and answering..."):
            qa_result = answer_question(
                question,
                case_report=result["clinical_report"],
                history=st.session_state.chat_history[chat_key],
            )
        with st.chat_message("assistant"):
            st.write(qa_result["answer"])
            with st.expander("Sources"):
                for s in qa_result["sources"]:
                    line = f"**[{s['number']}]** {s['title']}"
                    if s.get("doi"):
                        line += f" — DOI: [{s['doi']}]({s['doi_url']})"
                    st.markdown(line)
        st.session_state.chat_history[chat_key].append((question, qa_result["answer"]))


st.title("🧠 Alzheimer's OCT Screening Assistant")
st.caption(
    "Demo pipeline: retinal OCT scan → OCT-validity check → CNN vision model → "
    "confidence-gated RAG report synthesis → conversational follow-up, grounded in real literature."
)

mode = st.sidebar.radio("Mode", ["📋 Demo Case", "📤 Upload & Chat"])

# ---------------------------------------------------------------- Demo Case
if mode == "📋 Demo Case":
    with st.sidebar:
        st.header("Select a case")
        case_labels = [f"{c['metadata']['patient_id']} ({c['metadata']['diagnosis_label']})" for c in DEMO_CASES]
        selected_label = st.radio("Demo subjects (real Dryad OCT scans):", case_labels)
        selected_case = DEMO_CASES[case_labels.index(selected_label)]

        st.image(selected_case["image_path"], caption=selected_case["metadata"]["scan_id"], use_container_width=True)
        st.write(f"**Age:** {selected_case['metadata']['age']}")
        st.write(f"**Sex:** {selected_case['metadata']['sex']}")

        run_clicked = st.button("Run screening", type="primary", use_container_width=True)

    if "case_result" not in st.session_state:
        st.session_state.case_result = {}

    case_id = selected_case["metadata"]["patient_id"]

    if run_clicked:
        with st.spinner("Running vision model + RAG pipeline..."):
            try:
                st.session_state.case_result[case_id] = run_case(
                    selected_case["image_path"], selected_case["metadata"]
                )
            except FileNotFoundError as e:
                st.error(f"Vision model not available: {e}\n\nSee models/README.md.")

    result = st.session_state.case_result.get(case_id)
    if result:
        render_report_and_chat(result, chat_key=case_id)
    else:
        st.info("Select a case in the sidebar and click **Run screening** to begin.")

# ------------------------------------------------------------- Upload & Chat
else:
    st.sidebar.button("🔄 Start over", on_click=lambda: st.session_state.update(
        upload_stage="await_upload", upload_metadata={}, upload_transcript=[],
        upload_image_path=None, upload_result=None,
    ))

    defaults = {
        "upload_stage": "await_upload",
        "upload_metadata": {},
        "upload_transcript": [],
        "upload_image_path": None,
        "upload_result": None,
        "upload_last_file_id": None,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

    st.subheader("Upload a scan")
    uploaded_file = st.file_uploader("Upload an OCT image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and uploaded_file.file_id != st.session_state.upload_last_file_id:
        st.session_state.upload_last_file_id = uploaded_file.file_id
        suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getvalue())
        tmp.close()

        with st.spinner("Checking this is actually an OCT scan..."):
            validity = is_valid_oct_image(tmp.name)

        if not validity["is_valid_oct"]:
            st.session_state.upload_transcript.append((
                "assistant",
                f"⚠️ This doesn't look like a retinal OCT scan (OCT-likelihood: "
                f"{validity['oct_probability']:.0%}). Please upload a valid OCT image.",
            ))
            st.session_state.upload_stage = "await_upload"
        else:
            st.session_state.upload_image_path = tmp.name
            st.session_state.upload_transcript.append((
                "assistant",
                f" This looks like a valid OCT scan (confidence: {validity['oct_probability']:.0%}). "
                f"Let's collect a few patient details. What is the patient's **age**?",
            ))
            st.session_state.upload_stage = "collect_age"

    for role, text in st.session_state.upload_transcript:
        with st.chat_message(role):
            st.write(text)

    stage = st.session_state.upload_stage

    if stage in ("collect_age", "collect_sex", "collect_cognitive"):
        reply = st.chat_input("Your answer...")
        if reply:
            st.session_state.upload_transcript.append(("user", reply))

            if stage == "collect_age":
                age = parse_age(reply)
                if age is None:
                    st.session_state.upload_transcript.append(
                        ("assistant", "I didn't catch a valid age (1-120). Could you re-enter it?")
                    )
                else:
                    st.session_state.upload_metadata["age"] = age
                    st.session_state.upload_transcript.append(
                        ("assistant", "Got it. What is the patient's **sex** (Male/Female)?")
                    )
                    st.session_state.upload_stage = "collect_sex"

            elif stage == "collect_sex":
                sex = parse_sex(reply)
                if sex is None:
                    st.session_state.upload_transcript.append(
                        ("assistant", "Sorry, I need Male or Female — could you clarify?")
                    )
                else:
                    st.session_state.upload_metadata["sex"] = sex
                    st.session_state.upload_transcript.append((
                        "assistant",
                        "Thanks. Do you have a cognitive assessment score (e.g. MMSE)? "
                        "If not, just say 'skip'.",
                    ))
                    st.session_state.upload_stage = "collect_cognitive"

            elif stage == "collect_cognitive":
                score, skipped = parse_cognitive_score(reply)
                st.session_state.upload_metadata["cognitive_score"] = score
                st.session_state.upload_transcript.append(
                    ("assistant", "Thanks! Running the screening now...")
                )
                st.session_state.upload_stage = "processing"

            st.rerun()

    if stage == "processing":
        with st.spinner("Running vision model + RAG pipeline..."):
            meta = st.session_state.upload_metadata
            metadata = PatientMetadata(
                patient_id="UPLOADED",
                scan_id=os.path.basename(st.session_state.upload_image_path),
                age=meta["age"],
                sex=meta["sex"],
                device_model="Uploaded scan (device unknown)",
                diagnosis_label="Unknown (pending screening)",
                cognitive_score=meta.get("cognitive_score"),
            )
            try:
                st.session_state.upload_result = run_case(st.session_state.upload_image_path, metadata)
                st.session_state.upload_stage = "chatting"
            except FileNotFoundError as e:
                st.session_state.upload_transcript.append((
                    "assistant", f"⚠️ Vision model not available: {e}\n\nSee models/README.md.",
                ))
                st.session_state.upload_stage = "await_upload"
        st.rerun()

    if stage == "chatting" and st.session_state.upload_result:
        render_report_and_chat(st.session_state.upload_result, chat_key="uploaded")
    elif stage == "await_upload" and uploaded_file is None:
        st.info("Upload an OCT image above to start the conversation.")
