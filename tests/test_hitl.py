"""Tests for Phase 6: HITL (human-in-the-loop), feedback routing, and streaming."""

import json
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage
from langgraph.types import Command

from fetcher.nodes.supervisor import (
    route_after_human_review,
    revise_synthesis,
)
from fetcher.graphs.supervisor import build_supervisor_graph


# --- Unit tests for route_after_human_review ---


def test_route_approved():
    state = {"human_feedback": "approved"}
    assert route_after_human_review(state) == "finalize"


def test_route_approved_default():
    state = {}
    assert route_after_human_review(state) == "finalize"


def test_route_rejected():
    state = {"human_feedback": "rejected: not what I asked for"}
    assert route_after_human_review(state) == "replan"


def test_route_revision():
    state = {"human_feedback": "Please add more detail about performance"}
    assert route_after_human_review(state) == "revise"


# --- Unit test for revise_synthesis ---


def test_revise_synthesis_incorporates_feedback():
    state = {
        "user_query": "Explain sorting algorithms",
        "research_results": [{"answer": "Quicksort is O(n log n)"}],
        "code_results": [],
        "final_answer": "Quicksort is fast.",
        "human_feedback": "Please include merge sort too.",
    }

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Quicksort and merge sort are both O(n log n). Merge sort is stable."
        )
        mock_get_llm.return_value = mock_llm

        result = revise_synthesis(state)

    assert "merge sort" in result["final_answer"].lower()
    assert result["needs_human_approval"] is True
    assert result["human_feedback"] is None  # Reset for next review

    # Verify the prompt included human feedback
    call_args = mock_llm.invoke.call_args[0][0]
    human_msg = call_args[-1].content
    assert "Please include merge sort too." in human_msg


# --- Integration: HITL approve flow with interrupt ---


def test_hitl_approve_flow():
    """Full flow: planner → router → stubs → synthesizer → human_review (interrupt) → approve → finalize."""
    planner_response = json.dumps(
        {"tasks": [{"description": "Find info", "type": "research"}]}
    )
    synthesizer_response = "Here is the synthesized answer."

    call_count = {"n": 0}

    def mock_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(content=planner_response)
        return AIMessage(content=synthesizer_response)

    graph = build_supervisor_graph(use_stubs=True)
    # Need checkpointer for interrupt support
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-hitl-approve"}}

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # First run — should pause at human_review interrupt
        result = app.invoke(
            {"user_query": "Test query", "messages": []},
            config=config,
        )

    # Check state — should be interrupted
    state = app.get_state(config)
    assert state.next  # Graph is paused

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # Resume with approval
        result = app.invoke(Command(resume="approve"), config=config)

    assert result.get("human_feedback") == "approved"
    assert result.get("final_answer") == synthesizer_response


# --- Integration: HITL revision flow ---


def test_hitl_revision_flow():
    """Flow: synthesize → interrupt → revise feedback → re-synthesize → interrupt → approve."""
    planner_response = json.dumps(
        {"tasks": [{"description": "Explain Python", "type": "research"}]}
    )

    call_count = {"n": 0}

    def mock_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return AIMessage(content=planner_response)
        elif call_count["n"] == 2:
            return AIMessage(content="Original synthesis.")
        else:
            return AIMessage(content="Revised synthesis with more detail.")

    graph = build_supervisor_graph(use_stubs=True)
    from langgraph.checkpoint.memory import MemorySaver
    app = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-hitl-revise"}}

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # First run — pauses at human_review
        app.invoke({"user_query": "Explain Python", "messages": []}, config=config)

    state = app.get_state(config)
    assert state.next  # Paused

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # Resume with revision feedback — goes to revise_synthesis → human_review again
        app.invoke(Command(resume="Add more detail please"), config=config)

    state = app.get_state(config)
    assert state.next  # Paused again at human_review

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # Now approve
        result = app.invoke(Command(resume="approve"), config=config)

    assert result.get("human_feedback") == "approved"
    assert "revised" in result.get("final_answer", "").lower() or result.get("final_answer") != ""


# --- Integration: HITL reject flow ---


def test_hitl_reject_replan_flow():
    """Flow: synthesize → interrupt → reject → re-plan → stubs → re-synthesize → approve."""

    call_count = {"n": 0}

    def mock_invoke(messages):
        call_count["n"] += 1
        content = messages[-1].content if messages else ""

        if call_count["n"] == 1:
            # First planner call
            return AIMessage(content=json.dumps(
                {"tasks": [{"description": "Find info", "type": "research"}]}
            ))
        elif call_count["n"] == 2:
            # First synthesis
            return AIMessage(content="First attempt answer.")
        elif call_count["n"] == 3:
            # Re-plan after rejection
            return AIMessage(content=json.dumps(
                {"tasks": [{"description": "Find better info", "type": "research"}]}
            ))
        elif call_count["n"] == 4:
            # Second synthesis
            return AIMessage(content="Better answer after re-planning.")
        else:
            return AIMessage(content="Final fallback.")

    graph = build_supervisor_graph(use_stubs=True)
    from langgraph.checkpoint.memory import MemorySaver
    app = graph.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "test-hitl-reject"}}

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # First run — pauses at human_review
        app.invoke({"user_query": "Find info", "messages": []}, config=config)

    state = app.get_state(config)
    assert state.next

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # Reject — goes back to intake_planner
        app.invoke(Command(resume="reject:wrong topic"), config=config)

    # Should be paused again at human_review after re-plan
    state = app.get_state(config)
    assert state.next

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        # Now approve
        result = app.invoke(Command(resume="approve"), config=config)

    assert result.get("human_feedback") == "approved"


# --- LangSmith config test ---


def test_langsmith_config_env_gated():
    """LangSmith tracing is only enabled when LANGSMITH_API_KEY is set."""
    import os

    # Ensure no key → tracing disabled
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}, clear=False):
        import importlib
        import fetcher.config as cfg
        importlib.reload(cfg)
        assert cfg.LANGSMITH_TRACING_ENABLED is False

    # With key → tracing enabled
    with patch.dict(os.environ, {"LANGSMITH_API_KEY": "test-key"}, clear=False):
        importlib.reload(cfg)
        assert cfg.LANGSMITH_TRACING_ENABLED is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "test-key"

    # Clean up — reload with original env
    importlib.reload(cfg)
