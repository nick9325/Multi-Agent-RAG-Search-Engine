import os
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"

import pytest
from src.agents.decomposer import QueryDecomposerAgent
from src.agents.reranker import ReRankerAgent
from src.agents.generator import GeneratorAgent
from src.agents.reflection import SelfReflectionAgent


def test_query_decomposer_agent():
    decomposer = QueryDecomposerAgent()
    queries = decomposer.decompose("What is Reciprocal Rank Fusion and Cross-Encoder re-ranking?")
    assert isinstance(queries, list)
    assert len(queries) >= 1

def test_reranker_agent():
    reranker = ReRankerAgent()
    docs = [
        {"id": "1", "content": "Reciprocal Rank Fusion (RRF) combines vector search with lexical BM25.", "metadata": {}},
        {"id": "2", "content": "Baking chocolate cookies requires flour, sugar, and butter.", "metadata": {}}
    ]
    reranked = reranker.rerank("Reciprocal Rank Fusion retrieval", docs, top_k=2)
    assert len(reranked) > 0
    assert reranked[0]["id"] == "1"


def test_generator_agent_offline():
    generator = GeneratorAgent(api_key="")
    docs = [{"id": "1", "content": "ChromaDB stores embeddings locally.", "metadata": {"source": "test.txt"}}]
    answer = generator.generate("What is ChromaDB?", docs)
    assert "ChromaDB" in answer or "Summary" in answer

def test_self_reflection_agent_offline():
    reflector = SelfReflectionAgent(api_key="")
    docs = [{"id": "1", "content": "BM25 uses term frequency and inverse document frequency.", "metadata": {}}]
    score, valid, critique = reflector.evaluate("Explain BM25", docs, "BM25 is a lexical ranking function.")
    assert 0.0 <= score <= 1.0
    assert isinstance(valid, bool)
