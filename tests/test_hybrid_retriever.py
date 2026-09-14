import os
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"

import tempfile
import pytest
from src.database.chroma_store import LocalChromaStore
from src.database.bm25_store import LocalBM25Store
from src.database.hybrid_retriever import HybridRetriever


@pytest.fixture
def temp_stores():
    chroma_dir = tempfile.mkdtemp()
    bm25_dir = tempfile.mkdtemp()
    bm25_path = f"{bm25_dir}/bm25.pkl"
    chroma = LocalChromaStore(persist_directory=chroma_dir)
    bm25 = LocalBM25Store(persist_path=bm25_path)
    retriever = HybridRetriever(chroma_store=chroma, bm25_store=bm25)
    yield retriever
    chroma.close()


def test_ingest_and_hybrid_search(temp_stores):
    sample_docs = [
        {
            "content": "Reciprocal Rank Fusion (RRF) combines dense vector embeddings and BM25 lexical keyword scores efficiently.",
            "source": "doc1.txt"
        },
        {
            "content": "ChromaDB provides local persistent vector database storage for semantic search.",
            "source": "doc2.txt"
        },
        {
            "content": "Cross-Encoder models re-rank candidate passages by scoring query-passage pairs.",
            "source": "doc3.txt"
        }
    ]

    stats = temp_stores.ingest_documents(sample_docs)
    assert stats["num_documents"] == 3
    assert stats["num_chunks"] >= 3

    results = temp_stores.search("Reciprocal Rank Fusion BM25", top_k=2)
    assert len(results) > 0
    assert "RRF" in results[0]["content"] or "BM25" in results[0]["content"]
    assert "rrf_score" in results[0]
