"""Tests for the supervisor graph — routing logic and full flow with mocked LLM."""

import json
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from fetcher.nodes.supervisor import (
    intake_planner,
    router,
    route_by_task_type,
)
from fetcher.graphs.supervisor import build_supervisor_graph


# --- Unit tests for router ---


def test_router_advances_task_type_research():
    state = {
        "plan": ["[research] Find info", "[code] Write script"],
        "current_task_index": 0,
        "iteration_count": 0,
        "max_iterations": 10,
    }
    result = router(state)
    assert result["task_type"] == "research"
    assert result["iteration_count"] == 1


def test_router_advances_task_type_code():
    state = {
        "plan": ["[research] Find info", "[code] Write script"],
        "current_task_index": 1,
        "iteration_count": 1,
        "max_iterations": 10,
    }
    result = router(state)
    assert result["task_type"] == "code"


def test_router_signals_done_when_plan_exhausted():
    state = {
        "plan": ["[research] Find info"],
        "current_task_index": 1,
        "iteration_count": 0,
        "max_iterations": 10,
    }
    result = router(state)
    assert result["task_type"] == "done"


def test_router_signals_done_on_max_iterations():
    state = {
        "plan": ["[research] Task 1", "[research] Task 2"],
        "current_task_index": 0,
        "iteration_count": 10,
        "max_iterations": 10,
    }
    result = router(state)
    assert result["task_type"] == "done"


def test_router_hybrid_type():
    state = {
        "plan": ["[hybrid] Research then code"],
        "current_task_index": 0,
        "iteration_count": 0,
        "max_iterations": 10,
    }
    result = router(state)
    assert result["task_type"] == "hybrid"


# --- Unit tests for route_by_task_type ---


def test_route_by_task_type():
    assert route_by_task_type({"task_type": "research"}) == "research"
    assert route_by_task_type({"task_type": "code"}) == "code"
    assert route_by_task_type({"task_type": "hybrid"}) == "hybrid"
    assert route_by_task_type({"task_type": "done"}) == "done"


# --- Integration test: full graph with mocked LLM ---


def _mock_llm_response(content: str):
    """Create a real AIMessage so LangGraph can process it."""
    return AIMessage(content=content)


def test_full_graph_flow_with_mock_llm():
    """End-to-end: query → planner → router → stubs → synthesizer → finalize."""
    planner_response = json.dumps(
        {"tasks": [
            {"description": "Search for Python async patterns", "type": "research"},
            {"description": "Write example code", "type": "code"},
        ]}
    )
    synthesizer_response = "Here is the final answer combining research and code."

    call_count = {"n": 0}

    def mock_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _mock_llm_response(planner_response)
        return _mock_llm_response(synthesizer_response)

    graph = build_supervisor_graph()
    app = graph.compile()

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm_instance

        result = app.invoke({
            "user_query": "Explain Python async patterns and write an example",
            "messages": [],
        })

    assert result["task_type"] == "done"
    assert len(result["plan"]) == 2
    assert result["plan"][0].startswith("[research]")
    assert result["plan"][1].startswith("[code]")
    assert len(result["research_results"]) == 1
    assert len(result["code_results"]) == 1
    assert "final answer" in result["final_answer"].lower()
    assert result["current_task_index"] == 2
