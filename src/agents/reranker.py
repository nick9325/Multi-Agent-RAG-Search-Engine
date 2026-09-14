from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from src.config import settings

class ReRankerAgent:
    """Agent that re-ranks candidate retrieved documents using a Cross-Encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL_NAME
        self.model = CrossEncoder(self.model_name, device="cpu")
        self.threshold = settings.CROSS_ENCODER_THRESHOLD


    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Calculates cross-encoder relevance scores for (query, document_content) pairs,
        sorts descending, filters out scores below threshold, and returns top_k.
        """
        if not documents:
            return []

        # Prepare sentence pairs
        pairs = [[query, doc.get("content", "")] for doc in documents]
        scores = self.model.predict(pairs)

        reranked_docs = []
        for idx, doc in enumerate(documents):
            score = float(scores[idx])
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = round(score, 4)
            reranked_docs.append(doc_copy)

        # Sort descending by re-rank score
        reranked_docs = sorted(reranked_docs, key=lambda x: x["rerank_score"], reverse=True)

        # Filter by threshold if specified, or take top_k
        filtered = [doc for doc in reranked_docs if doc["rerank_score"] >= self.threshold]
        if not filtered:
            filtered = reranked_docs[:top_k]

        return filtered[:top_k]
