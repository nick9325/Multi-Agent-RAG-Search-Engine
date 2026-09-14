import os
import sys
from pathlib import Path
import streamlit as st

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import settings
from src.database.hybrid_retriever import HybridRetriever
from src.agents.graph import RAGGraphPipeline
from pypdf import PdfReader

# Page Config
st.set_page_config(
    page_title="Multi-Agent RAG Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode & Glassmorphism Aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .badge-grounded {
        background-color: #238636;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-hallucinated {
        background-color: #da3633;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .agent-step {
        border-left: 3px solid #58a6ff;
        padding-left: 12px;
        margin-bottom: 16px;
    }
    .subquery-pill {
        background: #1f242d;
        border: 1px solid #30363d;
        padding: 6px 12px;
        border-radius: 8px;
        margin: 4px;
        display: inline-block;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    retriever = HybridRetriever()
    pipeline = RAGGraphPipeline(hybrid_retriever=retriever)
    return retriever, pipeline

retriever, pipeline = get_pipeline()

# Sidebar: Document Management & Stats
with st.sidebar:
    st.title("⚙️ RAG Control Panel")

    st.subheader("📊 Engine Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chroma Chunks", retriever.chroma.count())
    with col2:
        st.metric("BM25 Tokens", retriever.bm25.count())

    st.markdown("---")
    st.subheader("📥 Ingest Documents")

    ingest_type = st.radio("Ingestion Mode", ["Paste Text", "Upload PDF / TXT"])

    if ingest_type == "Paste Text":
        doc_title = st.text_input("Document Title / Source", "manual_notes.txt")
        doc_text = st.text_area("Document Content", height=180, placeholder="Paste knowledge context here...")
        if st.button("🚀 Ingest Text", use_container_width=True):
            if doc_text.strip():
                stats = retriever.ingest_documents([{"content": doc_text, "source": doc_title}])
                st.success(f"Ingested {stats['num_chunks']} chunks into Vector & Lexical DB!")
                st.rerun()
            else:
                st.warning("Please enter document content.")

    elif ingest_type == "Upload PDF / TXT":
        uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)
        if st.button("🚀 Ingest Files", use_container_width=True):
            if uploaded_files:
                docs_to_ingest = []
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.endswith(".pdf"):
                        reader = PdfReader(uploaded_file)
                        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    else:
                        text = uploaded_file.read().decode("utf-8")

                    docs_to_ingest.append({
                        "content": text,
                        "source": uploaded_file.name
                    })

                stats = retriever.ingest_documents(docs_to_ingest)
                st.success(f"Ingested {len(docs_to_ingest)} files ({stats['num_chunks']} chunks) successfully!")
                st.rerun()

    st.markdown("---")
    if st.button("🗑️ Reset Database", type="secondary", use_container_width=True):
        retriever.reset()
        st.success("Database cleared!")
        st.rerun()

    st.markdown("---")
    st.caption("🤖 Powered by LangGraph • Groq • ChromaDB • BM25 • Cross-Encoders")

# Main Interface
st.title("⚡ Multi-Agent RAG Search Engine")
st.caption("Dense + Sparse Hybrid Search (RRF) • Cross-Encoder Re-ranking • Self-RAG Hallucination Validation")

query_input = st.text_input("Ask a technical question:", placeholder="e.g. How does Reciprocal Rank Fusion improve retrieval quality over dense vector search alone?")

if st.button("🔍 Execute Search & Reasoning", type="primary", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a search query.")
    elif retriever.chroma.count() == 0:
        st.error("No documents indexed yet! Please ingest documents using the sidebar first.")
    else:
        with st.spinner("Executing Multi-Agent Orchestration Workflow..."):
            result = pipeline.run(query_input)

        st.markdown("---")

        # Top Bar: Answer & Validation Status
        grounded_score = result.get("grounded_score", 0.0)
        hallucination_valid = result.get("hallucination_valid", True)
        critique = result.get("critique", "")

        col_badge, col_score = st.columns([1, 3])
        with col_badge:
            if hallucination_valid and grounded_score >= 0.7:
                st.markdown('<span class="badge-grounded">✅ Factually Grounded</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-hallucinated">⚠️ Potential Hallucination Warning</span>', unsafe_allow_html=True)
        with col_score:
            st.markdown(f"**Groundedness Score:** `{grounded_score:.2f} / 1.00` | *{critique}*")

        st.markdown("### 📝 Synthesized Answer")
        st.markdown(result.get("generated_answer", "No answer generated."))

        st.markdown("---")

        # Tabs for Visualizing Agent Steps & Retrieved Contexts
        tab_flow, tab_rerank, tab_raw = st.tabs(["🧩 Agent Execution Trace", "🎯 Re-ranker Scores", "📚 Source Passages"])

        with tab_flow:
            st.markdown("#### Step-by-Step Multi-Agent Execution Log")
            sub_queries = result.get("sub_queries", [])

            st.markdown("**1. Query Decomposer Agent:**")
            for q in sub_queries:
                st.markdown(f'<span class="subquery-pill">🔍 {q}</span>', unsafe_allow_html=True)

            st.markdown("\n**2. Execution Trace:**")
            for log in result.get("execution_logs", []):
                with st.expander(f"🔹 {log.get('agent', 'Agent')} — {log.get('step', 'Step')}"):
                    st.write(log.get("details", ""))
                    if "sub_queries" in log:
                        st.json(log["sub_queries"])

        with tab_rerank:
            st.markdown("#### Cross-Encoder Re-ranking Scores (`ms-marco-MiniLM-L-6-v2`)")
            reranked = result.get("reranked_docs", [])
            if reranked:
                table_data = []
                for idx, doc in enumerate(reranked, start=1):
                    table_data.append({
                        "Rank": idx,
                        "Re-rank Score": doc.get("rerank_score"),
                        "RRF Score": doc.get("rrf_score"),
                        "Source": doc.get("metadata", {}).get("source", "Unknown"),
                        "Snippet": doc.get("content", "")[:120] + "..."
                    })
                st.dataframe(table_data, use_container_width=True)

        with tab_raw:
            st.markdown("#### Retrieved Source Document Passages")
            for idx, doc in enumerate(result.get("reranked_docs", []), start=1):
                with st.expander(f"Passage [{idx}] — Source: {doc.get('metadata', {}).get('source')} (Score: {doc.get('rerank_score')})"):
                    st.write(doc.get("content", ""))
