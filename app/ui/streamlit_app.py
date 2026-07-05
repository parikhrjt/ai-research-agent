"""Streamlit frontend for the AI Research Agent."""

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { color: #6b7280; margin-bottom: 2rem; }
    .citation-card {
        background: #f8fafc; border-left: 3px solid #3b82f6;
        padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0 6px 6px 0;
    }
    .metric-box { background: #f1f5f9; padding: 1rem; border-radius: 8px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def upload_file(file) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=120,
        )
        if r.ok:
            return r.json()
        st.error(f"Upload failed: {r.json().get('error', r.text)}")
    except requests.RequestException as e:
        st.error(f"API unreachable: {e}")
    return None


def ask_question(question: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=180,
        )
        if r.ok:
            return r.json()
        st.error(f"Query failed: {r.json().get('error', r.text)}")
    except requests.RequestException as e:
        st.error(f"API unreachable: {e}")
    return None


def list_documents() -> list:
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=10)
        return r.json() if r.ok else []
    except requests.RequestException:
        return []


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    health = api_health()
    if health:
        status_color = "🟢" if health["status"] == "healthy" else "🟡"
        st.markdown(f"{status_color} **{health['status'].title()}**")
        st.caption(f"v{health['version']} · {health['vector_store']} · {health['llm_provider']}")
        for component, ok in health.get("components", {}).items():
            icon = "✅" if ok else "❌"
            st.caption(f"{icon} {component}")
    else:
        st.error("API offline — start with `docker compose up`")

    st.divider()
    st.markdown("### 📚 Indexed Documents")
    docs = list_documents()
    if docs:
        for doc in docs:
            st.caption(f"📄 {doc['filename']} ({doc.get('chunk_count', '?')} chunks)")
    else:
        st.caption("No documents indexed yet")

    st.divider()
    st.markdown("### Supported Formats")
    st.caption("PDF · TXT · Markdown · CSV")

# Main layout
st.markdown('<p class="main-header">🔬 AI Research Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload research documents and ask questions with cited answers.</p>',
    unsafe_allow_html=True,
)

tab_upload, tab_ask, tab_about = st.tabs(["📤 Upload", "💬 Ask", "ℹ️ About"])

with tab_upload:
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Drop a document",
            type=["pdf", "txt", "md", "markdown", "csv"],
            help="Supported: PDF, TXT, Markdown, CSV notes",
        )
        if uploaded and st.button("Ingest Document", type="primary"):
            with st.spinner("Parsing, chunking, and embedding..."):
                result = upload_file(uploaded)
            if result:
                st.success(f"Indexed **{result['filename']}** — {result['chunk_count']} chunks")
                st.json(result)

    with col2:
        st.markdown("#### Quick Start")
        st.markdown(
            """
            1. Upload a PDF, markdown, or CSV file
            2. Wait for ingestion to complete
            3. Switch to **Ask** tab
            4. Query your knowledge base
            """
        )

        samples_dir = "data/samples"
        if os.path.isdir(samples_dir):
            st.markdown("#### Sample Files")
            for fname in sorted(os.listdir(samples_dir)):
                fpath = os.path.join(samples_dir, fname)
                with open(fpath, "rb") as f:
                    if st.download_button(f"⬇️ {fname}", f, file_name=fname):
                        pass

with tab_ask:
    question = st.text_area(
        "Your question",
        placeholder="What are the key findings from the transformer architecture paper?",
        height=100,
    )

    example_questions = [
        "What is the main contribution of the research?",
        "Summarize the methodology used.",
        "What metrics were reported in the experiments?",
        "List the key datasets mentioned.",
    ]
    st.caption("Try an example:")
    cols = st.columns(len(example_questions))
    for i, eq in enumerate(example_questions):
        if cols[i].button(eq, key=f"eq_{i}"):
            question = eq
            st.session_state["last_question"] = eq

    if "last_question" in st.session_state and not question:
        question = st.session_state["last_question"]

    if st.button("Get Answer", type="primary", disabled=not question):
        with st.spinner("Retrieving context and generating answer..."):
            result = ask_question(question)
        if result:
            st.markdown("### Answer")
            st.markdown(result["answer"])

            if result["citations"]:
                st.markdown("### Sources")
                for cite in result["citations"]:
                    st.markdown(
                        f"""<div class="citation-card">
                        <strong>[{cite['index']}] {cite['filename']}</strong>
                        <span style="color:#6b7280"> (score: {cite['score']})</span><br/>
                        <em>{cite['excerpt']}</em>
                        </div>""",
                        unsafe_allow_html=True,
                    )
            st.caption(f"Retrieved {result['retrieved_count']} chunks")

with tab_about:
    st.markdown(
        """
        ### Architecture

        This agent implements a production RAG pipeline:

        - **Ingestion** — PDF, TXT, Markdown, CSV parsers
        - **Chunking** — Format-aware recursive splitting with overlap
        - **Embeddings** — Local `all-MiniLM-L6-v2` (free, no API key)
        - **Vector Store** — PostgreSQL + pgvector (or ChromaDB)
        - **Retrieval** — Cosine similarity search with score thresholding
        - **Generation** — Ollama (local) or OpenAI with inline citations

        Built with FastAPI, Streamlit, LangGraph, and Docker Compose.
        """
    )
