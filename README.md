# ⚡ Multi-Agent RAG Search Engine

A production-grade, 100% free-tier & local-capable **Multi-Agent RAG Search Engine** featuring **LangGraph** orchestration, **ChromaDB + BM25** Hybrid Search with Reciprocal Rank Fusion (RRF), **Cross-Encoder** re-ranking, **Self-RAG** reflection & hallucination validation, a **FastAPI** REST API, and an interactive **Streamlit** dashboard.

---

## 🌟 Key Architecture Features

- **Multi-Agent Pipeline (LangGraph)**:
  - **Query Decomposer Agent**: Breaks complex technical questions into 2-4 focused sub-queries using Groq (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`).
  - **Hybrid Search Agent**: Dense Vector embeddings (`BAAI/bge-small-en-v1.5`) + Sparse Lexical (`BM25`) combined via Reciprocal Rank Fusion (RRF).
  - **Re-ranker Agent**: Dynamically re-ranks passages using HuggingFace Cross-Encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to eliminate noise.
  - **Generator Agent**: Synthesizes factually grounded answers with inline numerical document citations (`[Doc 1]`).
  - **Self-Reflection & Hallucination Validator Agent**: Evaluates factual grounding and triggers corrective retry loops if hallucinations are detected.
- **100% Free-Tier & Local Stack**:
  - Local ChromaDB vector database (persistent).
  - Local Rank-BM25 pickle index.
  - Local SentenceTransformers embeddings and Cross-Encoder model.
  - Groq API for free high-speed LLM inference.
- **FastAPI REST API**: Comprehensive endpoints for query processing, document ingestion, system health, and resetting index.
- **Streamlit Interactive UI**: Dark-mode glassmorphism UI with live document uploader, agent step execution visualizers, re-ranker score tables, and grounding badges.
- **Containerization**: `Dockerfile` and `docker-compose.yml` for single & multi-service deployment.

---

## 📁 Repository Structure

```
multi_agent_rag_engine/
├── .env.example              # Environment variables template
├── Dockerfile                # Container definition
├── docker-compose.yml        # Docker compose multi-service definition
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── src/
│   ├── config.py             # System configuration & hyperparameters
│   ├── database/
│   │   ├── chroma_store.py   # Local ChromaDB vector store
│   │   ├── bm25_store.py     # Local BM25 sparse lexical search
│   │   └── hybrid_retriever.py # RRF hybrid search engine
│   ├── agents/
│   │   ├── state.py          # LangGraph RAGState schema
│   │   ├── decomposer.py     # Query Decomposer Agent
│   │   ├── reranker.py       # Cross-Encoder Re-ranker Agent
│   │   ├── generator.py      # Answer Synthesis Agent
│   │   ├── reflection.py     # Self-Reflection Agent
│   │   └── graph.py          # LangGraph StateGraph pipeline
│   └── api/
│       └── main.py           # FastAPI REST application
├── ui/
│   └── streamlit_app.py      # Streamlit Web Dashboard UI
└── tests/
    ├── test_hybrid_retriever.py
    ├── test_agents.py
    └── test_api.py
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and navigate into `multi_agent_rag_engine`:
```bash
cd multi_agent_rag_engine
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Get a free Groq API key from [https://console.groq.com](https://console.groq.com) and set it in `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key
```

### 3. Run FastAPI Backend
Start the FastAPI server:
```bash
python -m src.api.main
```
- API Docs: `http://localhost:8000/docs`
- Status check: `http://localhost:8000/api/status`

### 4. Run Streamlit UI Dashboard
In a separate terminal window:
```bash
streamlit run ui/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

To build and launch both FastAPI and Streamlit services via Docker Compose:
```bash
docker-compose up --build
```
- Streamlit UI: `http://localhost:8501`
- FastAPI REST API: `http://localhost:8000/docs`
