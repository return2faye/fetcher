"""Tests for the RAG sub-graph — node logic and CRAG routing."""

import json
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

from fetcher.nodes.rag import (
    decide_action,
    retrieve,
    grade_documents,
    web_search,
    generate,
)
from fetcher.graphs.rag import build_rag_graph, create_rag_initial_state


# --- Unit tests for decide_action ---


def test_decide_action_relevant():
    state = {"retrieval_grade": "relevant", "rewrite_count": 0, "max_rewrites": 2}
    assert decide_action(state) == "generate"


def test_decide_action_ambiguous_with_rewrites_left():
    state = {"retrieval_grade": "ambiguous", "rewrite_count": 0, "max_rewrites": 2}
    assert decide_action(state) == "rewrite_query"


def test_decide_action_ambiguous_rewrites_exhausted():
    state = {"retrieval_grade": "ambiguous", "rewrite_count": 2, "max_rewrites": 2}
    assert decide_action(state) == "web_search"


def test_decide_action_irrelevant():
    state = {"retrieval_grade": "irrelevant", "rewrite_count": 0, "max_rewrites": 2}
    assert decide_action(state) == "web_search"


# --- Unit test for web_search node ---


def test_web_search_returns_documents():
    mock_results = [
        {"title": "Python Async", "body": "Async programming guide", "href": "https://example.com/1"},
        {"title": "Asyncio Docs", "body": "Official docs", "href": "https://example.com/2"},
    ]

    state = {
        "query": "python async programming",
        "documents": [],
        "relevance_scores": [],
    }

    with patch("fetcher.nodes.rag.DDGS") as MockDDGS:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=False)
        mock_ddgs_instance.text.return_value = mock_results
        MockDDGS.return_value = mock_ddgs_instance

        result = web_search(state)

    assert len(result["documents"]) == 2
    assert result["use_web_search"] is True
    assert "Python Async" in result["documents"][0]["text"]


def test_web_search_handles_failure():
    state = {"query": "test", "documents": [], "relevance_scores": []}

    with patch("fetcher.nodes.rag.DDGS") as MockDDGS:
        MockDDGS.side_effect = Exception("network error")
        result = web_search(state)

    assert result["documents"] == []
    assert result["use_web_search"] is True


# --- Integration test: full CRAG flow with mocked LLM and Qdrant ---


def test_rag_graph_relevant_path():
    """Test the happy path: retrieve → grade (relevant) → generate."""

    mock_search_results = [
        {"text": "Python async uses coroutines", "score": 0.9, "metadata": {"source": "doc1"}},
        {"text": "Asyncio event loop manages tasks", "score": 0.85, "metadata": {"source": "doc2"}},
    ]

    call_count = {"n": 0}

    def mock_llm_invoke(messages):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            # Grader calls — mark as relevant
            return AIMessage(content='{"relevant": true}')
        else:
            # Generate call
            return AIMessage(content="Python async uses coroutines and asyncio event loops.")

    graph = build_rag_graph()
    app = graph.compile()

    with (
        patch("fetcher.nodes.rag.search_documents", return_value=mock_search_results),
        patch("fetcher.nodes.rag._get_llm") as mock_get_llm,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_get_llm.return_value = mock_llm

        state = create_rag_initial_state("how does Python async work?")
        result = app.invoke(state)

    assert result["retrieval_grade"] == "relevant"
    assert "async" in result["generation"].lower()
    assert len(result["citations"]) > 0


def test_rag_graph_irrelevant_falls_back_to_web():
    """Test: retrieve → grade (irrelevant) → web_search → generate."""

    # Empty retrieval
    mock_search_results = []

    def mock_llm_invoke(messages):
        return AIMessage(content="Answer from web search results.")

    mock_web_results = [
        {"title": "Web Result", "body": "Found on the web", "href": "https://example.com"},
    ]

    graph = build_rag_graph()
    app = graph.compile()

    with (
        patch("fetcher.nodes.rag.search_documents", return_value=mock_search_results),
        patch("fetcher.nodes.rag._get_llm") as mock_get_llm,
        patch("fetcher.nodes.rag.DDGS") as MockDDGS,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_get_llm.return_value = mock_llm

        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = mock_web_results
        MockDDGS.return_value = mock_ddgs

        state = create_rag_initial_state("some obscure topic")
        result = app.invoke(state)

    assert result["use_web_search"] is True
    assert result["generation"] != ""


def test_rag_graph_ambiguous_rewrites_then_generates():
    """Test: retrieve → grade (ambiguous) → rewrite → retrieve → grade (relevant) → generate."""

    call_count = {"retrieve": 0, "grade": 0, "other": 0}

    def mock_search(query, top_k=5):
        call_count["retrieve"] += 1
        if call_count["retrieve"] == 1:
            # First retrieval: only 1 doc (ambiguous)
            return [{"text": "Partial info", "score": 0.6, "metadata": {"source": "doc1"}}]
        else:
            # After rewrite: 2 good docs
            return [
                {"text": "Full info about topic", "score": 0.9, "metadata": {"source": "doc2"}},
                {"text": "More details on topic", "score": 0.85, "metadata": {"source": "doc3"}},
            ]

    def mock_llm_invoke(messages):
        content = messages[-1].content if messages else ""
        if "Document:" in content:
            call_count["grade"] += 1
            return AIMessage(content='{"relevant": true}')
        elif "rewrite" in content.lower():
            return AIMessage(content="improved search query about the topic")
        else:
            return AIMessage(content="Comprehensive answer after rewrite.")

    graph = build_rag_graph()
    app = graph.compile()

    with (
        patch("fetcher.nodes.rag.search_documents", side_effect=mock_search),
        patch("fetcher.nodes.rag._get_llm") as mock_get_llm,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke = mock_llm_invoke
        mock_get_llm.return_value = mock_llm

        state = create_rag_initial_state("ambiguous topic")
        result = app.invoke(state)

    assert call_count["retrieve"] == 2  # retrieved twice
    assert result["rewrite_count"] == 1
    assert result["retrieval_grade"] == "relevant"
    assert result["generation"] != ""
