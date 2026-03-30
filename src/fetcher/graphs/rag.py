"""RAG sub-graph: Corrective RAG (CRAG) with retrieve → grade → decide → generate/rewrite/web_search."""

from langgraph.graph import StateGraph, START, END

from fetcher.state import RAGState
from fetcher.config import RAG_RELEVANCE_THRESHOLD, MAX_RAG_REWRITES
from fetcher.nodes.rag import (
    retrieve,
    grade_documents,
    decide_action,
    rewrite_query,
    web_search,
    generate,
)


def build_rag_graph() -> StateGraph:
    """Construct the CRAG sub-graph (uncompiled)."""
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)

    # Edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # Conditional: grade_documents → decide where to go
    graph.add_conditional_edges(
        "grade_documents",
        decide_action,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )

    # Rewrite loops back to retrieve
    graph.add_edge("rewrite_query", "retrieve")

    # Web search goes to generate
    graph.add_edge("web_search", "generate")

    # Generate is terminal
    graph.add_edge("generate", END)

    return graph


def compile_rag():
    """Compile the RAG sub-graph (no checkpointer — parent graph owns that)."""
    return build_rag_graph().compile()


def create_rag_initial_state(query: str) -> dict:
    """Helper to create the initial state for invoking the RAG sub-graph."""
    return {
        "messages": [],
        "query": query,
        "original_query": query,
        "documents": [],
        "relevance_scores": [],
        "relevance_threshold": RAG_RELEVANCE_THRESHOLD,
        "retrieval_grade": "irrelevant",
        "rewrite_count": 0,
        "max_rewrites": MAX_RAG_REWRITES,
        "use_web_search": False,
        "generation": "",
        "citations": [],
    }
