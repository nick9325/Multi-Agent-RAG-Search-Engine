from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.config import settings

class GeneratorAgent:
    """Agent that synthesizes grounded final answers with inline citations from reranked passages."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = ""):
        self.api_key = settings.GROQ_API_KEY if api_key is None else api_key
        self.model_name = model_name or settings.PRIMARY_LLM_MODEL
        self.llm = None
        if self.api_key:

            try:
                self.llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=self.model_name,
                    temperature=0.1
                )
            except Exception as e:
                print(f"[GeneratorAgent] ChatGroq initialization warning: {e}")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Synthesis & Answering Agent in a Multi-Agent RAG Search Engine.
Your objective is to provide a comprehensive, clear, and factually grounded answer to the user's question based strictly on the provided retrieved context passages.

Guidelines:
1. Base your answer ONLY on the provided context passages. Do not invent information.
2. Use inline numerical citations (e.g. [Doc 1], [Doc 2]) whenever citing facts from the context.
3. If the context does not contain enough information to answer the question fully, state clearly what is missing.
4. Structure your output clearly using markdown headers, bullet points, and clean formatting."""),
            ("human", """Question: {query}

Context Passages:
{context}

Please provide a detailed, grounded answer with citations.""")
        ])

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No context passages available."

        formatted = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.get("metadata", {}).get("source", "Unknown Source")
            content = doc.get("content", "").strip()
            formatted.append(f"[Doc {idx}] (Source: {source})\n{content}")

        return "\n\n".join(formatted)

    def generate(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Synthesizes final answer."""
        context_str = self._format_context(documents)

        if not self.llm:
            # Fallback answer synthesis if no Groq API Key is available
            if not documents:
                return "No document context available to answer the query. Please ingest documents into the RAG engine first."
            summary_bits = [f"- {doc['content'][:150]}... [Doc {idx+1}]" for idx, doc in enumerate(documents[:3])]
            return (
                f"**Answer Summary (Local Offline Mode)**:\n\n"
                f"Based on retrieved documents for '{query}':\n\n" +
                "\n".join(summary_bits) +
                "\n\n*(Note: Set your GROQ_API_KEY in .env for full LLM synthesis.)*"
            )

        try:
            formatted_prompt = self.prompt.format_messages(query=query, context=context_str)
            response = self.llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[GeneratorAgent] Answer generation error: {e}")
            return f"An error occurred while generating the answer: {str(e)}"
