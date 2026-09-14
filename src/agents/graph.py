from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from src.agents.state import RAGState
from src.agents.decomposer import QueryDecomposerAgent
from src.agents.reranker import ReRankerAgent
from src.agents.generator import GeneratorAgent
from src.agents.reflection import SelfReflectionAgent
from src.database.hybrid_retriever import HybridRetriever
from src.config import settings

class RAGGraphPipeline:
    """Multi-Agent Orchestration Graph built with LangGraph."""

    def __init__(self, hybrid_retriever: Optional[HybridRetriever] = None):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.decomposer = QueryDecomposerAgent()
        self.reranker = ReRankerAgent()
        self.generator = GeneratorAgent()
        self.reflector = SelfReflectionAgent()

        self.app = self._build_graph()

    def _decompose_node(self, state: RAGState) -> Dict[str, Any]:
        logs = state.get("execution_logs", [])
        query = state["original_query"]
        retry_count = state.get("retry_count", 0)
        critique = state.get("critique", "")

        current_query = query
        if retry_count > 0 and critique:
            current_query = f"{query} (Refining based on self-reflection: {critique})"

        sub_queries = self.decomposer.decompose(current_query)

        logs.append({
            "step": "Query Decomposition",
            "agent": "Query Decomposer Agent",
            "details": f"Generated {len(sub_queries)} focused sub-queries.",
            "sub_queries": sub_queries
        })

        return {
            "sub_queries": sub_queries,
            "execution_logs": logs
        }

    def _retrieve_node(self, state: RAGState) -> Dict[str, Any]:
        logs = state.get("execution_logs", [])
        sub_queries = state.get("sub_queries", [state["original_query"]])

        retrieved = self.retriever.search_multi_queries(
            queries=sub_queries,
            top_k=settings.RETRIEVAL_TOP_K
        )

        logs.append({
            "step": "Hybrid Search (Dense + BM25 RRF)",
            "agent": "Hybrid Search Agent",
            "details": f"Retrieved {len(retrieved)} candidate passages via RRF fusion.",
            "count": len(retrieved)
        })

        return {
            "retrieved_docs": retrieved,
            "execution_logs": logs
        }

    def _rerank_node(self, state: RAGState) -> Dict[str, Any]:
        logs = state.get("execution_logs", [])
        query = state["original_query"]
        retrieved_docs = state.get("retrieved_docs", [])

        reranked = self.reranker.rerank(
            query=query,
            documents=retrieved_docs,
            top_k=settings.RERANK_TOP_K
        )

        logs.append({
            "step": "Cross-Encoder Re-ranking",
            "agent": "Re-ranker Agent",
            "details": f"Re-ranked passages using ms-marco-MiniLM-L-6-v2. Selected top {len(reranked)} passages.",
            "reranked_scores": [doc.get("rerank_score") for doc in reranked]
        })

        return {
            "reranked_docs": reranked,
            "execution_logs": logs
        }

    def _generate_node(self, state: RAGState) -> Dict[str, Any]:
        logs = state.get("execution_logs", [])
        query = state["original_query"]
        reranked_docs = state.get("reranked_docs", [])

        answer = self.generator.generate(
            query=query,
            documents=reranked_docs
        )

        logs.append({
            "step": "Answer Synthesis",
            "agent": "Generator Agent",
            "details": "Synthesized grounded answer with inline document citations."
        })

        return {
            "generated_answer": answer,
            "execution_logs": logs
        }

    def _reflect_node(self, state: RAGState) -> Dict[str, Any]:
        logs = state.get("execution_logs", [])
        query = state["original_query"]
        reranked_docs = state.get("reranked_docs", [])
        answer = state.get("generated_answer", "")
        retry_count = state.get("retry_count", 0)

        grounded_score, hallucination_valid, critique = self.reflector.evaluate(
            query=query,
            documents=reranked_docs,
            answer=answer
        )

        logs.append({
            "step": "Self-Reflection & Hallucination Validation",
            "agent": "Self-Reflection Agent",
            "details": f"Grounded Score: {grounded_score:.2f} | Hallucination Valid: {hallucination_valid}",
            "grounded_score": grounded_score,
            "hallucination_valid": hallucination_valid,
            "critique": critique
        })

        return {
            "grounded_score": grounded_score,
            "hallucination_valid": hallucination_valid,
            "critique": critique,
            "retry_count": retry_count + 1 if not hallucination_valid else retry_count,
            "execution_logs": logs
        }

    def _should_retry(self, state: RAGState) -> str:
        hallucination_valid = state.get("hallucination_valid", True)
        retry_count = state.get("retry_count", 0)

        if not hallucination_valid and retry_count < 2:
            return "retry"
        return "end"

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("decompose", self._decompose_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("rerank", self._rerank_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("reflect", self._reflect_node)

        workflow.set_entry_point("decompose")

        workflow.add_edge("decompose", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "reflect")

        workflow.add_conditional_edges(
            "reflect",
            self._should_retry,
            {
                "retry": "decompose",
                "end": END
            }
        )

        return workflow.compile()

    def run(self, query: str) -> RAGState:
        """Executes the pipeline graph for a query."""
        initial_state: RAGState = {
            "original_query": query,
            "sub_queries": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "generated_answer": "",
            "grounded_score": 0.0,
            "hallucination_valid": True,
            "critique": "",
            "retry_count": 0,
            "execution_logs": []
        }

        final_state = self.app.invoke(initial_state)
        return final_state
