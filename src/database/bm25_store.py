import os
import pickle
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import settings

class LocalBM25Store:
    """Sparse BM25 lexical search store with persistent serialization."""

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or settings.BM25_PERSIST_PATH
        self.corpus_docs: List[Dict[str, Any]] = []  # list of {"id": str, "content": str, "metadata": dict}
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        self.load()

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercasing and regex word tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to BM25 index.
        documents format: list of dicts with {"id": str, "content": str, "metadata": dict}
        """
        new_tokens = [self._tokenize(doc["content"]) for doc in documents]
        self.corpus_docs.extend(documents)
        self.corpus_tokens.extend(new_tokens)

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self.save()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Execute BM25 lexical search."""
        if not self.bm25 or not self.corpus_docs:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            doc = self.corpus_docs[idx]
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": round(float(score), 4),
                "search_type": "sparse"
            })

        return results

    def save(self):
        """Persists the BM25 index to pickle file."""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        data = {
            "corpus_docs": self.corpus_docs,
            "corpus_tokens": self.corpus_tokens
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        """Loads BM25 index from pickle file if exists."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.corpus_docs = data.get("corpus_docs", [])
                    self.corpus_tokens = data.get("corpus_tokens", [])
                    if self.corpus_tokens:
                        self.bm25 = BM25Okapi(self.corpus_tokens)
            except Exception as e:
                print(f"[BM25Store] Failed to load index from {self.persist_path}: {e}")
                self.corpus_docs = []
                self.corpus_tokens = []
                self.bm25 = None

    def count(self) -> int:
        return len(self.corpus_docs)

    def reset(self):
        """Clears BM25 index."""
        self.corpus_docs = []
        self.corpus_tokens = []
        self.bm25 = None
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)
