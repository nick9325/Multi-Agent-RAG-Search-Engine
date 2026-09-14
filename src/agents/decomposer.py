import json
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings

class QueryDecomposerAgent:
    """Agent that decomposes a complex user query into 2-4 focused sub-queries."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = ""):
        self.api_key = settings.GROQ_API_KEY if api_key is None else api_key
        self.model_name = model_name or settings.FAST_LLM_MODEL
        self.llm = None
        if self.api_key:

            try:
                self.llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=0.2
                )
            except Exception as e:
                print(f"[QueryDecomposerAgent] ChatGroq initialization warning: {e}")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Query Decomposer Agent in a Multi-Agent RAG Search Engine.
Your job is to analyze complex user questions and break them down into 2-4 distinct, focused sub-queries to optimize vector and keyword search retrieval.

Follow these rules:
1. Return ONLY a valid JSON array of strings containing the sub-queries.
2. Make each sub-query specific, clear, and focused on a single key aspect of the main question.
3. Do NOT include markdown code formatting backticks or explanation. Just return raw JSON array.

Example output:
["What is Reciprocal Rank Fusion in RAG?", "How does Cross-Encoder re-ranking work?", "How to integrate ChromaDB and BM25?"]"""),
            ("human", "Decompose the following user question into focused search sub-queries:\n\n{query}")
        ])

    def decompose(self, query: str) -> List[str]:
        """Runs decomposition logic."""
        if not self.llm:
            # Fallback if no LLM API Key is provided
            return [
                query,
                f"Core concepts in: {query}",
                f"Detailed details about: {query}"
            ]

        try:
            formatted_prompt = self.prompt.format_messages(query=query)
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()

            # Clean JSON formatting if enclosed in code blocks
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            sub_queries = json.loads(content)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                # Always ensure original query is included
                if query not in sub_queries:
                    sub_queries.insert(0, query)
                return sub_queries[:4]
        except Exception as e:
            print(f"[QueryDecomposerAgent] Decomposition failed: {e}. Falling back to default list.")

        return [query, f"Overview of {query}"]
