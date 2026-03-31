"""Supervisor graph: top-level orchestrator with conditional routing to sub-graphs."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from fetcher.state import SupervisorState
from fetcher.config import SQLITE_DB_PATH
from fetcher.nodes.supervisor import (
    intake_planner,
    router,
    route_by_task_type,
    rag_subgraph_stub,
    code_subgraph_stub,
    hybrid_stub,
    synthesizer,
    human_review,
    finalize,
)
from fetcher.nodes.integration import rag_node, code_node, hybrid_node


def build_supervisor_graph(use_stubs: bool = False) -> StateGraph:
    """Construct the supervisor StateGraph (uncompiled).

    Args:
        use_stubs: If True, use stub nodes (for testing without Docker/Qdrant).
                   If False (default), use real sub-graph integrations.
    """
    graph = StateGraph(SupervisorState)

    # Add nodes
    graph.add_node("intake_planner", intake_planner)
    graph.add_node("router", router)

    if use_stubs:
        graph.add_node("rag_subgraph", rag_subgraph_stub)
        graph.add_node("code_subgraph", code_subgraph_stub)
        graph.add_node("hybrid_subgraph", hybrid_stub)
    else:
        graph.add_node("rag_subgraph", rag_node)
        graph.add_node("code_subgraph", code_node)
        graph.add_node("hybrid_subgraph", hybrid_node)

    graph.add_node("synthesizer", synthesizer)
    graph.add_node("human_review", human_review)
    graph.add_node("finalize", finalize)

    # Edges
    graph.add_edge(START, "intake_planner")
    graph.add_edge("intake_planner", "router")

    # Conditional routing from router
    graph.add_conditional_edges(
        "router",
        route_by_task_type,
        {
            "research": "rag_subgraph",
            "code": "code_subgraph",
            "hybrid": "hybrid_subgraph",
            "done": "synthesizer",
        },
    )

    # After each sub-graph, loop back to router
    graph.add_edge("rag_subgraph", "router")
    graph.add_edge("code_subgraph", "router")
    graph.add_edge("hybrid_subgraph", "router")

    # Synthesis → human review → finalize
    graph.add_edge("synthesizer", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    return graph


def compile_supervisor(use_stubs: bool = False):
    """Compile the supervisor graph with SQLite checkpointer."""
    graph = build_supervisor_graph(use_stubs=use_stubs)
    checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{SQLITE_DB_PATH}")
    return graph.compile(checkpointer=checkpointer)
