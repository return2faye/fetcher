"""Code sub-graph: Generate → Execute → Verify with self-correction loop."""

from langgraph.graph import StateGraph, START, END

from fetcher.state import CodeState
from fetcher.config import MAX_CODE_RETRIES
from fetcher.nodes.code import (
    coder,
    executor,
    critic,
    error_handler,
    should_retry,
)


def build_code_graph() -> StateGraph:
    """Construct the Code sub-graph (uncompiled)."""
    graph = StateGraph(CodeState)

    # Nodes
    graph.add_node("coder", coder)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)
    graph.add_node("error_handler", error_handler)

    # Edges
    graph.add_edge(START, "coder")
    graph.add_edge("coder", "executor")
    graph.add_edge("executor", "critic")

    # Conditional: critic → end or retry
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "end": END,
            "retry": "error_handler",
        },
    )

    # error_handler loops back to coder
    graph.add_edge("error_handler", "coder")

    return graph


def compile_code():
    """Compile the Code sub-graph (no checkpointer — parent owns that)."""
    return build_code_graph().compile()


def create_code_initial_state(
    task_description: str,
    context: str = "",
) -> dict:
    """Helper to create the initial state for invoking the Code sub-graph."""
    return {
        "messages": [],
        "task_description": task_description,
        "context": context,
        "generated_code": "",
        "language": "python",
        "execution_result": "",
        "execution_error": None,
        "exit_code": None,
        "retry_count": 0,
        "max_retries": MAX_CODE_RETRIES,
        "critic_feedback": None,
        "verified_output": "",
        "is_verified": False,
    }
