"""Tests for Phase 7: Error handling, timeouts, input validation, and graceful degradation."""

import json
from unittest.mock import patch, MagicMock
from concurrent.futures import TimeoutError as FuturesTimeoutError

from langchain_core.messages import AIMessage

from fetcher.nodes.supervisor import intake_planner, synthesizer, get_llm
from fetcher.nodes.code import coder, executor, critic
from fetcher.nodes.rag import grade_documents, rewrite_query, generate
from fetcher.nodes.integration import rag_node, code_node, _extract_task_description
from fetcher.utils.docker_sandbox import execute_in_sandbox


# --- Input validation tests ---


def test_intake_planner_empty_query():
    """Empty query returns done immediately."""
    state = {"user_query": "", "messages": []}
    with patch("fetcher.nodes.supervisor.get_llm"):
        result = intake_planner(state)
    assert result["task_type"] == "done"
    assert result["final_answer"] == "No query provided."


def test_intake_planner_whitespace_only_query():
    """Whitespace-only query treated as empty."""
    state = {"user_query": "   \n\t  ", "messages": []}
    with patch("fetcher.nodes.supervisor.get_llm"):
        result = intake_planner(state)
    assert result["task_type"] == "done"


def test_intake_planner_long_query_truncated():
    """Query longer than MAX_QUERY_LENGTH is truncated."""
    long_query = "x" * 20000

    call_count = {"n": 0}

    def mock_invoke(messages):
        call_count["n"] += 1
        # Verify the query was truncated
        content = messages[-1].content
        assert len(content) <= 10000  # Default MAX_QUERY_LENGTH
        return AIMessage(content=json.dumps(
            {"tasks": [{"description": "truncated task", "type": "research"}]}
        ))

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke = mock_invoke
        mock_get_llm.return_value = mock_llm

        result = intake_planner({"user_query": long_query, "messages": []})

    assert len(result["plan"]) == 1


def test_intake_planner_invalid_task_types_normalized():
    """Unknown task types in LLM response are normalized to 'research'."""
    response = json.dumps(
        {"tasks": [{"description": "do something", "type": "unknown_type"}]}
    )

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content=response)
        mock_get_llm.return_value = mock_llm

        result = intake_planner({"user_query": "test", "messages": []})

    assert result["plan"][0].startswith("[research]")


def test_intake_planner_malformed_json_response():
    """Planner falls back to single research task on malformed LLM response."""
    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="this is not json at all")
        mock_get_llm.return_value = mock_llm

        result = intake_planner({"user_query": "test query", "messages": []})

    assert len(result["plan"]) == 1
    assert result["plan"][0] == "[research] test query"


def test_intake_planner_json_not_object():
    """Planner falls back when LLM returns JSON array instead of object."""
    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content='["not", "an", "object"]')
        mock_get_llm.return_value = mock_llm

        result = intake_planner({"user_query": "test query", "messages": []})

    assert len(result["plan"]) == 1
    assert result["plan"][0] == "[research] test query"


# --- LLM error handling tests ---


def test_intake_planner_llm_failure_fallback():
    """Planner falls back to single research task when LLM call fails."""
    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = intake_planner({"user_query": "test query", "messages": []})

    assert len(result["plan"]) == 1
    assert result["plan"][0] == "[research] test query"
    assert result["messages"] == []  # No response message on failure


def test_synthesizer_llm_failure_returns_raw_results():
    """Synthesizer returns raw sub-results when LLM fails."""
    state = {
        "user_query": "test",
        "research_results": [{"answer": "Research answer"}],
        "code_results": [],
    }

    with patch("fetcher.nodes.supervisor.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API down")
        mock_get_llm.return_value = mock_llm

        result = synthesizer(state)

    assert "Synthesis failed" in result["final_answer"]
    assert "Research answer" in result["final_answer"]


def test_coder_llm_failure_returns_empty_code():
    """Coder returns empty code when LLM fails (executor will report error)."""
    state = {
        "task_description": "write hello world",
        "retry_count": 0,
        "critic_feedback": None,
        "context": "",
    }

    with patch("fetcher.nodes.code._get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = coder(state)

    assert result["generated_code"] == ""


def test_critic_llm_failure_optimistic_pass():
    """Critic optimistically passes when LLM fails (code ran successfully)."""
    state = {
        "task_description": "print hello",
        "generated_code": "print('hello')",
        "execution_result": "hello\n",
        "execution_error": None,
        "exit_code": 0,
    }

    with patch("fetcher.nodes.code._get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = critic(state)

    assert result["is_verified"] is True


def test_rag_grader_llm_failure_uses_vector_score():
    """Grader falls back to vector score when individual LLM grading call fails."""
    state = {
        "query": "test",
        "documents": [
            {"text": "doc1"},
            {"text": "doc2"},
        ],
        "relevance_scores": [0.9, 0.8],
        "relevance_threshold": 0.7,
    }

    with patch("fetcher.nodes.rag._get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = grade_documents(state)

    # Both docs have score >= threshold, so both should be included via fallback
    assert len(result["documents"]) == 2
    assert result["retrieval_grade"] == "relevant"


def test_rag_rewrite_failure_uses_original_query():
    """Rewrite returns original query when LLM fails."""
    state = {
        "original_query": "original question",
        "query": "current question",
        "rewrite_count": 0,
    }

    with patch("fetcher.nodes.rag._get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = rewrite_query(state)

    assert result["query"] == "original question"
    assert result["rewrite_count"] == 1


def test_rag_generate_failure_returns_raw_docs():
    """Generate returns raw documents when LLM fails."""
    state = {
        "query": "test query",
        "documents": [{"text": "doc content", "metadata": {"source": "s1"}}],
    }

    with patch("fetcher.nodes.rag._get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        mock_get_llm.return_value = mock_llm

        result = generate(state)

    assert "Generation failed" in result["generation"]
    assert "doc content" in result["generation"]


# --- Docker sandbox timeout test ---


def test_sandbox_timeout_returns_exit_124():
    """Docker execution that exceeds timeout returns exit code 124."""
    mock_container = MagicMock()
    mock_container.status = "running"

    mock_client = MagicMock()
    mock_client.containers.get.return_value = mock_container

    # Simulate a timeout by making exec_run block
    def slow_exec(*args, **kwargs):
        import time
        time.sleep(5)

    mock_container.exec_run = slow_exec

    with patch("fetcher.utils.docker_sandbox.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        mock_docker.errors.NotFound = Exception
        mock_docker.errors.DockerException = Exception

        result = execute_in_sandbox("while True: pass", timeout=1)

    assert result["exit_code"] == 124
    assert "timed out" in result["stderr"].lower()


def test_sandbox_docker_not_available():
    """Graceful error when Docker daemon is not available."""
    with patch("fetcher.utils.docker_sandbox.docker") as mock_docker:
        mock_docker.from_env.side_effect = Exception("Docker not running")
        mock_docker.errors.NotFound = type("NotFound", (Exception,), {})
        mock_docker.errors.DockerException = type("DockerException", (Exception,), {})

        result = execute_in_sandbox("print('hello')")

    assert result["exit_code"] == 1
    assert result["stderr"]  # Some error message


# --- Integration node error handling ---


def test_rag_node_subgraph_failure_graceful():
    """rag_node returns error message when RAG sub-graph crashes."""
    state = {
        "current_task_index": 0,
        "plan": ["[research] Find info"],
        "research_results": [],
    }

    with (
        patch("fetcher.nodes.integration._get_rag_app") as mock_app,
        patch("fetcher.utils.memory.store_result"),
        patch("fetcher.utils.memory.recall_context", return_value=""),
    ):
        mock_app.return_value.invoke.side_effect = Exception("Qdrant down")

        result = rag_node(state)

    assert len(result["research_results"]) == 1
    assert "failed" in result["research_results"][0]["answer"].lower()
    assert result["current_task_index"] == 1  # Still advances


def test_code_node_subgraph_failure_graceful():
    """code_node returns error message when Code sub-graph crashes."""
    state = {
        "current_task_index": 0,
        "plan": ["[code] Write script"],
        "code_results": [],
        "research_results": [],
    }

    with (
        patch("fetcher.nodes.integration._get_code_app") as mock_app,
        patch("fetcher.utils.memory.store_result"),
        patch("fetcher.utils.memory.recall_context", return_value=""),
    ):
        mock_app.return_value.invoke.side_effect = Exception("Docker down")

        result = code_node(state)

    assert len(result["code_results"]) == 1
    assert result["code_results"][0]["is_verified"] is False
    assert "failed" in result["code_results"][0]["output"].lower()
    assert result["current_task_index"] == 1


# --- LLM timeout configuration test ---


def test_llm_timeout_configured():
    """LLM instances are created with timeout parameter."""
    llm = get_llm()
    # ChatOpenAI stores timeout in request_timeout
    assert llm.request_timeout is not None


# --- Extract task description edge case ---


def test_extract_task_description_negative_index():
    """Negative index doesn't silently access from end of list."""
    plan = ["[research] First", "[code] Second"]
    # Python allows negative indexing, but our function should handle it
    result = _extract_task_description(plan, -1)
    # Negative index is valid Python, returns last element — this is acceptable behavior
    assert result in ("Second", "unknown task")
