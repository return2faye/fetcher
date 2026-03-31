"""Tests for the Code sub-graph — node logic, self-correction loop, and Docker sandbox."""

import json
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from fetcher.nodes.code import (
    coder,
    executor,
    critic,
    error_handler,
    should_retry,
    _extract_code_block,
)
from fetcher.graphs.code import build_code_graph, create_code_initial_state


# --- Unit tests for helpers ---


def test_extract_code_block_with_python_fence():
    text = "Here is the code:\n```python\nprint('hello')\n```\nDone."
    assert _extract_code_block(text) == "print('hello')"


def test_extract_code_block_without_language():
    text = "```\nx = 1\nprint(x)\n```"
    assert _extract_code_block(text) == "x = 1\nprint(x)"


def test_extract_code_block_no_fence():
    text = "print('hello')"
    assert _extract_code_block(text) == "print('hello')"


# --- Unit tests for should_retry ---


def test_should_retry_verified():
    state = {"is_verified": True, "retry_count": 0, "max_retries": 3}
    assert should_retry(state) == "end"


def test_should_retry_retries_exhausted():
    state = {"is_verified": False, "retry_count": 3, "max_retries": 3}
    assert should_retry(state) == "end"


def test_should_retry_can_retry():
    state = {"is_verified": False, "retry_count": 1, "max_retries": 3}
    assert should_retry(state) == "retry"


# --- Unit test for error_handler ---


def test_error_handler_increments_retry():
    state = {
        "retry_count": 0,
        "execution_error": "Traceback (most recent call last):\n  File ...\nNameError: name 'x' is not defined",
        "critic_feedback": None,
    }
    result = error_handler(state)
    assert result["retry_count"] == 1
    assert "NameError" in result["critic_feedback"]


# --- Unit test for executor with real Docker sandbox ---


def test_executor_runs_code_in_sandbox():
    """Integration test: requires fetcher-sandbox container running."""
    state = {
        "generated_code": "print(2 + 2)",
        "language": "python",
    }
    result = executor(state)
    assert result["exit_code"] == 0
    assert "4" in result["execution_result"]
    assert result["execution_error"] is None


def test_executor_captures_errors():
    """Integration test: requires fetcher-sandbox container running."""
    state = {
        "generated_code": "raise ValueError('test error')",
        "language": "python",
    }
    result = executor(state)
    assert result["exit_code"] != 0
    assert "ValueError" in result["execution_error"]


def test_executor_handles_empty_code():
    state = {"generated_code": "", "language": "python"}
    result = executor(state)
    assert result["exit_code"] == 1
    assert "No code" in result["execution_error"]


# --- Integration test: happy path (code works on first try) ---


def test_code_graph_happy_path():
    """coder → executor → critic (pass) → END"""

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Coder generates code
            return AIMessage(content="```python\nprint(sum(range(10)))\n```")
        else:
            # Critic says pass
            return AIMessage(content='{"verdict": "pass"}')

    graph = build_code_graph()
    app = graph.compile()

    with (
        patch("fetcher.nodes.code._get_llm") as mock_get_llm,
        patch("fetcher.nodes.code.execute_in_sandbox") as mock_exec,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_get_llm.return_value = mock_llm
        mock_exec.return_value = {"stdout": "45\n", "stderr": "", "exit_code": 0}

        state = create_code_initial_state("Calculate sum of 0 to 9")
        result = app.invoke(state)

    assert result["is_verified"] is True
    assert result["verified_output"] == "45\n"
    assert result["retry_count"] == 0


# --- Integration test: error → self-correction → success ---


def test_code_graph_self_correction():
    """coder (bad) → executor (error) → critic (fail) → error_handler → coder (fixed) → executor (ok) → critic (pass)"""

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First attempt: buggy code
            return AIMessage(content="```python\nprint(undefined_var)\n```")
        elif call_count["n"] == 2:
            # Second attempt: fixed code
            return AIMessage(content="```python\nresult = 42\nprint(result)\n```")
        else:
            # Critic says pass
            return AIMessage(content='{"verdict": "pass"}')

    exec_count = {"n": 0}

    def mock_exec(code, language="python"):
        exec_count["n"] += 1
        if exec_count["n"] == 1:
            return {"stdout": "", "stderr": "NameError: name 'undefined_var' is not defined", "exit_code": 1}
        return {"stdout": "42\n", "stderr": "", "exit_code": 0}

    graph = build_code_graph()
    app = graph.compile()

    with (
        patch("fetcher.nodes.code._get_llm") as mock_get_llm,
        patch("fetcher.nodes.code.execute_in_sandbox", side_effect=mock_exec),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_get_llm.return_value = mock_llm

        state = create_code_initial_state("Print the answer to everything")
        result = app.invoke(state)

    assert result["is_verified"] is True
    assert result["verified_output"] == "42\n"
    assert result["retry_count"] == 1


# --- Integration test: retries exhausted ---


def test_code_graph_retries_exhausted():
    """Code keeps failing until max retries, then exits with is_verified=False."""

    def mock_llm_invoke(messages):
        return AIMessage(content="```python\nraise Exception('always fails')\n```")

    def mock_exec(code, language="python"):
        return {"stdout": "", "stderr": "Exception: always fails", "exit_code": 1}

    graph = build_code_graph()
    app = graph.compile()

    with (
        patch("fetcher.nodes.code._get_llm") as mock_get_llm,
        patch("fetcher.nodes.code.execute_in_sandbox", side_effect=mock_exec),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_get_llm.return_value = mock_llm

        state = create_code_initial_state("Do something impossible")
        state["max_retries"] = 2  # Limit retries for test speed
        result = app.invoke(state)

    assert result["is_verified"] is False
    assert result["retry_count"] == 2
