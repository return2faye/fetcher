"""Integration tests: real sub-graphs wired into supervisor with mocked LLM.

These tests use the real RAG and Code sub-graphs (not stubs) but mock:
- LLM calls (deterministic, no API cost)
- Qdrant search (no dependency on ingested data)
- Docker sandbox (no dependency on running container)
- DuckDuckGo search (no network dependency)
"""

import json
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from fetcher.graphs.supervisor import build_supervisor_graph
from fetcher.nodes.integration import _extract_task_description
from fetcher.utils.memory import store_result, recall_context


# --- Unit tests for helpers ---


def test_extract_task_description():
    plan = ["[research] Find sorting algorithms", "[code] Write benchmark"]
    assert _extract_task_description(plan, 0) == "Find sorting algorithms"
    assert _extract_task_description(plan, 1) == "Write benchmark"
    assert _extract_task_description(plan, 5) == "unknown task"


def test_extract_task_description_hybrid():
    plan = ["[hybrid] Research and implement"]
    assert _extract_task_description(plan, 0) == "Research and implement"


# --- Integration: research-only query through real RAG sub-graph ---


def test_supervisor_research_with_real_rag():
    """Full flow: planner → router → real RAG → synthesizer."""

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        content = messages[-1].content if messages else ""

        if call_count["n"] == 1:
            # Planner
            return AIMessage(content=json.dumps(
                {"tasks": [{"description": "Explain Python decorators", "type": "research"}]}
            ))
        elif "Document:" in content:
            # RAG grader
            return AIMessage(content='{"relevant": true}')
        elif "Query:" in content and "Documents:" in content:
            # RAG generator
            return AIMessage(content="Python decorators are functions that modify other functions.")
        else:
            # Synthesizer
            return AIMessage(content="Final answer: Python decorators wrap functions to add behavior.")

    mock_search_results = [
        {"text": "Decorators use @ syntax in Python", "score": 0.9, "metadata": {"source": "doc1"}},
        {"text": "A decorator takes a function and returns a modified version", "score": 0.85, "metadata": {"source": "doc2"}},
    ]

    graph = build_supervisor_graph(use_stubs=False)
    app = graph.compile()

    with (
        patch("fetcher.nodes.supervisor.get_llm") as mock_sup_llm,
        patch("fetcher.nodes.rag._get_llm") as mock_rag_llm,
        patch("fetcher.nodes.rag.search_documents", return_value=mock_search_results),
        patch("fetcher.utils.memory.store_result"),
        patch("fetcher.utils.memory.recall_context", return_value=""),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_sup_llm.return_value = mock_llm
        mock_rag_llm.return_value = mock_llm

        result = app.invoke({
            "user_query": "Explain Python decorators",
            "messages": [],
        })

    assert result["task_type"] == "done"
    assert len(result["research_results"]) == 1
    assert "decorator" in result["research_results"][0]["answer"].lower()
    assert "Final answer" in result["final_answer"]


# --- Integration: code-only query through real Code sub-graph ---


def test_supervisor_code_with_real_code_subgraph():
    """Full flow: planner → router → real Code → synthesizer."""

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        content = messages[-1].content if messages else ""

        if call_count["n"] == 1:
            # Planner
            return AIMessage(content=json.dumps(
                {"tasks": [{"description": "Print the first 5 Fibonacci numbers", "type": "code"}]}
            ))
        elif "Task:" in content and ("Code:" not in content or "verdict" not in content.lower()):
            # Coder
            return AIMessage(content="```python\na, b = 0, 1\nfor _ in range(5):\n    print(a)\n    a, b = b, a + b\n```")
        elif "verdict" in str(messages[0].content).lower():
            # Critic
            return AIMessage(content='{"verdict": "pass"}')
        else:
            # Synthesizer
            return AIMessage(content="Final answer: The first 5 Fibonacci numbers are 0, 1, 1, 2, 3.")

    graph = build_supervisor_graph(use_stubs=False)
    app = graph.compile()

    with (
        patch("fetcher.nodes.supervisor.get_llm") as mock_sup_llm,
        patch("fetcher.nodes.code._get_llm") as mock_code_llm,
        patch("fetcher.nodes.code.execute_in_sandbox") as mock_exec,
        patch("fetcher.utils.memory.store_result"),
        patch("fetcher.utils.memory.recall_context", return_value=""),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_sup_llm.return_value = mock_llm
        mock_code_llm.return_value = mock_llm
        mock_exec.return_value = {"stdout": "0\n1\n1\n2\n3\n", "stderr": "", "exit_code": 0}

        result = app.invoke({
            "user_query": "Print the first 5 Fibonacci numbers",
            "messages": [],
        })

    assert result["task_type"] == "done"
    assert len(result["code_results"]) == 1
    assert result["code_results"][0]["is_verified"] is True
    assert "Final answer" in result["final_answer"]


# --- Integration: hybrid query (research then code) ---


def test_supervisor_hybrid_real_subgraphs():
    """Full flow: planner → router → real RAG → real Code → synthesizer."""

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        content = messages[-1].content if messages else ""

        if call_count["n"] == 1:
            # Planner
            return AIMessage(content=json.dumps(
                {"tasks": [{"description": "Research sorting algorithms and write a benchmark", "type": "hybrid"}]}
            ))
        elif "Document:" in content:
            # RAG grader
            return AIMessage(content='{"relevant": true}')
        elif "Query:" in content and "Documents:" in content:
            # RAG generator
            return AIMessage(content="Quicksort has O(n log n) average case.")
        elif "Task:" in content and "fix" not in content.lower() and "verdict" not in str(messages[0].content).lower():
            # Coder
            return AIMessage(content="```python\nimport time\nprint('benchmark: 0.5s')\n```")
        elif "verdict" in str(messages[0].content).lower():
            # Critic
            return AIMessage(content='{"verdict": "pass"}')
        else:
            # Synthesizer
            return AIMessage(content="Final answer: Quicksort is efficient. Benchmark shows 0.5s.")

    mock_search_results = [
        {"text": "Quicksort is a divide-and-conquer algorithm", "score": 0.9, "metadata": {"source": "doc1"}},
        {"text": "Merge sort guarantees O(n log n)", "score": 0.85, "metadata": {"source": "doc2"}},
    ]

    graph = build_supervisor_graph(use_stubs=False)
    app = graph.compile()

    with (
        patch("fetcher.nodes.supervisor.get_llm") as mock_sup_llm,
        patch("fetcher.nodes.rag._get_llm") as mock_rag_llm,
        patch("fetcher.nodes.code._get_llm") as mock_code_llm,
        patch("fetcher.nodes.rag.search_documents", return_value=mock_search_results),
        patch("fetcher.nodes.code.execute_in_sandbox") as mock_exec,
        patch("fetcher.utils.memory.store_result"),
        patch("fetcher.utils.memory.recall_context", return_value=""),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_sup_llm.return_value = mock_llm
        mock_rag_llm.return_value = mock_llm
        mock_code_llm.return_value = mock_llm
        mock_exec.return_value = {"stdout": "benchmark: 0.5s\n", "stderr": "", "exit_code": 0}

        result = app.invoke({
            "user_query": "Research sorting algorithms and write a benchmark",
            "messages": [],
        })

    assert result["task_type"] == "done"
    assert len(result["research_results"]) == 1
    assert len(result["code_results"]) == 1
    assert result["code_results"][0]["is_verified"] is True
    assert "Final answer" in result["final_answer"]


# --- Integration: multi-task plan (research + code as separate tasks) ---


def test_supervisor_multi_task_plan():
    """Planner creates 2 tasks: research then code. Both execute sequentially."""

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        content = messages[-1].content if messages else ""

        if call_count["n"] == 1:
            # Planner: 2 tasks
            return AIMessage(content=json.dumps({"tasks": [
                {"description": "Research Python list comprehensions", "type": "research"},
                {"description": "Write examples of list comprehensions", "type": "code"},
            ]}))
        elif "Document:" in content:
            return AIMessage(content='{"relevant": true}')
        elif "Query:" in content and "Documents:" in content:
            return AIMessage(content="List comprehensions provide concise syntax for creating lists.")
        elif "Task:" in content and "verdict" not in str(messages[0].content).lower():
            return AIMessage(content="```python\nresult = [x**2 for x in range(5)]\nprint(result)\n```")
        elif "verdict" in str(messages[0].content).lower():
            return AIMessage(content='{"verdict": "pass"}')
        else:
            return AIMessage(content="Final answer: List comprehensions are concise. Examples: [x**2 for x in range(5)]")

    mock_search = [
        {"text": "List comprehensions are syntactic sugar", "score": 0.9, "metadata": {"source": "d1"}},
        {"text": "They replace map/filter patterns", "score": 0.8, "metadata": {"source": "d2"}},
    ]

    graph = build_supervisor_graph(use_stubs=False)
    app = graph.compile()

    with (
        patch("fetcher.nodes.supervisor.get_llm") as mock_sup_llm,
        patch("fetcher.nodes.rag._get_llm") as mock_rag_llm,
        patch("fetcher.nodes.code._get_llm") as mock_code_llm,
        patch("fetcher.nodes.rag.search_documents", return_value=mock_search),
        patch("fetcher.nodes.code.execute_in_sandbox") as mock_exec,
        patch("fetcher.utils.memory.store_result"),
        patch("fetcher.utils.memory.recall_context", return_value=""),
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_sup_llm.return_value = mock_llm
        mock_rag_llm.return_value = mock_llm
        mock_code_llm.return_value = mock_llm
        mock_exec.return_value = {"stdout": "[0, 1, 4, 9, 16]\n", "stderr": "", "exit_code": 0}

        result = app.invoke({
            "user_query": "Explain Python list comprehensions and write examples",
            "messages": [],
        })

    assert result["task_type"] == "done"
    assert len(result["plan"]) == 2
    assert len(result["research_results"]) == 1
    assert len(result["code_results"]) == 1
    assert result["current_task_index"] == 2


# --- Unit test: memory graceful degradation ---


def test_memory_store_does_not_raise_on_failure():
    """store_result silently fails if Qdrant is unreachable."""
    with patch("fetcher.utils.memory._ensure_memory_collection"):
        with patch("fetcher.utils.memory.embed_texts", side_effect=Exception("embed error")):
            # Should not raise
            store_result("test task", "test result", result_type="research")


def test_memory_recall_returns_empty_on_failure():
    """recall_context returns empty string if Qdrant is unreachable."""
    with patch("fetcher.utils.memory._ensure_memory_collection"):
        with patch("fetcher.utils.memory.embed_query", side_effect=Exception("embed error")):
            context = recall_context("test query")
    assert context == ""


def test_memory_recall_with_live_qdrant():
    """Integration: store then recall via the running Qdrant instance."""
    import fetcher.utils.memory as mem

    # Reset initialization flag so collection is created
    mem._initialized = False

    store_result("fibonacci sequence", "0 1 1 2 3 5 8", result_type="research")
    context = recall_context("fibonacci numbers", top_k=1, score_threshold=0.3)

    assert "fibonacci" in context.lower()
