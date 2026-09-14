from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.database.hybrid_retriever import HybridRetriever
from src.agents.graph import RAGGraphPipeline

app = FastAPI(
    title="Multi-Agent RAG Search Engine API",
    description="Production-ready RAG Engine using LangGraph, RRF Hybrid Search, Cross-Encoder Re-ranking, and Self-Reflection Validation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton instances
retriever = HybridRetriever()
pipeline = RAGGraphPipeline(hybrid_retriever=retriever)

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user question or technical query.", json_schema_extra={"example": "How does Reciprocal Rank Fusion improve RAG retrieval precision?"})

class DocumentItem(BaseModel):
    content: str = Field(..., description="The main textual content of the document.")
    source: Optional[str] = Field(default="user_upload", description="Source identifier or filename.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom metadata tags.")

class IngestRequest(BaseModel):
    documents: List[DocumentItem]

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Multi-Agent RAG Search Engine",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/status")
def get_status():
    return {
        "status": "healthy",
        "chroma_count": retriever.chroma.count(),
        "bm25_count": retriever.bm25.count(),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "cross_encoder_model": settings.CROSS_ENCODER_MODEL_NAME,
        "primary_llm": settings.PRIMARY_LLM_MODEL,
        "fast_llm": settings.FAST_LLM_MODEL
    }

@app.post("/api/query")
def process_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty.")

    try:
        result_state = pipeline.run(request.query.strip())
        return {
            "query": result_state["original_query"],
            "sub_queries": result_state.get("sub_queries", []),
            "answer": result_state.get("generated_answer", ""),
            "grounded_score": result_state.get("grounded_score", 0.0),
            "hallucination_valid": result_state.get("hallucination_valid", True),
            "critique": result_state.get("critique", ""),
            "reranked_docs": result_state.get("reranked_docs", []),
            "execution_logs": result_state.get("execution_logs", [])
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/documents/ingest")
def ingest_documents(request: IngestRequest):
    if not request.documents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document list cannot be empty.")

    docs = [doc.model_dump() for doc in request.documents]
    try:
        stats = retriever.ingest_documents(docs)
        return {
            "message": "Documents successfully ingested and indexed.",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/api/reset")
def reset_stores():
    try:
        retriever.reset()
        return {"message": "All vector and lexical stores reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
