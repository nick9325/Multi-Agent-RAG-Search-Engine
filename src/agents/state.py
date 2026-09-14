from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class RAGState(TypedDict):
    original_query: str
    sub_queries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    generated_answer: str
    grounded_score: float
    hallucination_valid: bool
    critique: str
    retry_count: int
    execution_logs: List[Dict[str, Any]]
