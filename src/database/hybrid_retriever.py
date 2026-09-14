from typing import List, Dict, Any, Optional
from src.database.chroma_store import LocalChromaStore
from src.database.bm25_store import LocalBM25Store
from src.config import settings

class HybridRetriever:
    """Hybrid Retriever executing Dense Vector + Sparse BM25 combined via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        chroma_store: Optional[LocalChromaStore] = None,
        bm25_store: Optional[LocalBM25Store] = None,
        rrf_k: int = 60
    ):
        self.chroma = chroma_store or LocalChromaStore()
        self.bm25 = bm25_store or LocalBM25Store()
        self.rrf_k = rrf_k or settings.RRF_K

    def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest raw text/documents into both ChromaDB and BM25 store.
        Returns document count and chunk IDs.
        """
        # Add to Chroma (which chunks text automatically and returns IDs with metadata)
        chunk_ids = self.chroma.add_documents(documents)

        # Retrieve chunked docs from Chroma to index identical chunks in BM25
        if chunk_ids:
            # Query back all added chunks or fetch from collection
            results = self.chroma.collection.get(ids=chunk_ids)
            bm25_input_docs = []
            if results and results.get("documents"):
                for i in range(len(results["ids"])):
                    bm25_input_docs.append({
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
                self.bm25.add_documents(bm25_input_docs)

        return {
            "num_documents": len(documents),
            "num_chunks": len(chunk_ids),
            "chroma_total": self.chroma.count(),
            "bm25_total": self.bm25.count()
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Search for a single query using RRF.
        """
        dense_results = self.chroma.search(query, top_k=top_k * 2)
        sparse_results = self.bm25.search(query, top_k=top_k * 2)

        return self.reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Combines two ranked lists using RRF algorithm.
        RRF_score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # Process dense results
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            dense_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
            doc_map[doc_id] = doc

        # Process sparse results
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            sparse_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # Sort documents by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id], reverse=True)

        final_results = []
        for doc_id in sorted_ids[:top_k]:
            doc_info = doc_map[doc_id].copy()
            doc_info["rrf_score"] = round(rrf_scores[doc_id], 6)
            doc_info["dense_rank"] = dense_ranks.get(doc_id, None)
            doc_info["sparse_rank"] = sparse_ranks.get(doc_id, None)
            final_results.append(doc_info)

        return final_results

    def search_multi_queries(self, queries: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes hybrid search across multiple sub-queries and aggregates deduplicated passages.
        """
        all_results_map: Dict[str, Dict[str, Any]] = {}

        for q in queries:
            sub_results = self.search(q, top_k=top_k)
            for doc in sub_results:
                doc_id = doc["id"]
                if doc_id not in all_results_map or doc["rrf_score"] > all_results_map[doc_id]["rrf_score"]:
                    all_results_map[doc_id] = doc

        sorted_docs = sorted(all_results_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_docs[:top_k]

    def reset(self):
        """Reset both stores."""
        self.chroma.reset()
        self.bm25.reset()
