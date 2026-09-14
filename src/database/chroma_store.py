import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.config import settings

class LocalChromaStore:
    """Persistent local ChromaDB manager using SentenceTransformers embeddings."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_dir = persist_directory or settings.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device="cpu")

        self.collection = self.client.get_or_create_collection(
            name="multi_agent_rag_docs",
            metadata={"hnsw:space": "cosine"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest documents.
        Each doc is expected to be dict: {"content": str, "source": str, "metadata": optional dict}
        """
        all_chunks = []
        all_ids = []
        all_metadatas = []

        chunk_counter = 0
        for doc_idx, doc in enumerate(documents):
            text = doc.get("content", "")
            source = doc.get("source", f"doc_{doc_idx}")
            extra_meta = doc.get("metadata", {})

            chunks = self.text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{source}_chunk_{chunk_idx}_{chunk_counter}"
                metadata = {
                    "source": source,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    **extra_meta
                }
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append(metadata)
                chunk_counter += 1

        if not all_chunks:
            return []

        embeddings = self._embed_texts(all_chunks)
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        return all_ids

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Dense vector search query."""
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        formatted_results = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i in range(len(docs)):
                # Convert cosine distance to similarity score
                similarity = 1.0 - distances[i] if distances[i] is not None else 0.0
                formatted_results.append({
                    "id": ids[i],
                    "content": docs[i],
                    "metadata": metadatas[i],
                    "score": round(float(similarity), 4),
                    "search_type": "dense"
                })

        return formatted_results

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Clears all collection data."""
        self.client.delete_collection("multi_agent_rag_docs")
        self.collection = self.client.get_or_create_collection(
            name="multi_agent_rag_docs",
            metadata={"hnsw:space": "cosine"}
        )

    def close(self):
        """Releases Chroma client handles for clean deletion in tests."""
        if hasattr(self, "client") and self.client:
            try:
                self.client._system.stop()
            except Exception:
                pass
            del self.collection
            del self.client

