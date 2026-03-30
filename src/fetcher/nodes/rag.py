"""RAG sub-graph nodes: retrieve, grade_documents, decide_action, rewrite_query, web_search, generate."""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS

from fetcher.config import (
    OPENAI_MODEL,
    RAG_RELEVANCE_THRESHOLD,
    MAX_RAG_REWRITES,
)
from fetcher.state import RAGState
from fetcher.utils.qdrant_client import search_documents


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


# --- Node: retrieve ---

def retrieve(state: RAGState) -> dict:
    """Query Qdrant for documents similar to the current query."""
    query = state["query"]
    results = search_documents(query, top_k=5)

    documents = [
        {"text": r["text"], "metadata": r["metadata"]}
        for r in results
    ]
    scores = [r["score"] for r in results]

    return {
        "documents": documents,
        "relevance_scores": scores,
    }


# --- Node: grade_documents ---

GRADER_SYSTEM_PROMPT = """\
You are a relevance grader. Given a user query and a retrieved document, \
determine if the document is relevant to answering the query.

Respond with ONLY a JSON object: {"relevant": true} or {"relevant": false}
"""


def grade_documents(state: RAGState) -> dict:
    """LLM grades each retrieved document for relevance. Sets retrieval_grade."""
    import json

    llm = _get_llm()
    query = state["query"]
    documents = state.get("documents", [])
    threshold = state.get("relevance_threshold", RAG_RELEVANCE_THRESHOLD)

    if not documents:
        return {"retrieval_grade": "irrelevant", "documents": [], "relevance_scores": []}

    graded_docs = []
    graded_scores = []

    for doc, score in zip(documents, state.get("relevance_scores", [])):
        # Use vector score as a fast pre-filter
        if score < threshold * 0.5:
            continue

        messages = [
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nDocument: {doc['text']}"),
        ]
        response = llm.invoke(messages)

        try:
            parsed = json.loads(response.content)
            if parsed.get("relevant", False):
                graded_docs.append(doc)
                graded_scores.append(score)
        except (json.JSONDecodeError, KeyError):
            # If parse fails, include doc if vector score is high enough
            if score >= threshold:
                graded_docs.append(doc)
                graded_scores.append(score)

    # Determine overall grade
    if len(graded_docs) >= 2:
        grade = "relevant"
    elif len(graded_docs) == 1:
        grade = "ambiguous"
    else:
        grade = "irrelevant"

    return {
        "documents": graded_docs,
        "relevance_scores": graded_scores,
        "retrieval_grade": grade,
    }


# --- Node: decide_action (conditional edge function) ---

def decide_action(state: RAGState) -> str:
    """Route based on retrieval_grade and rewrite count."""
    grade = state.get("retrieval_grade", "irrelevant")
    rewrites = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", MAX_RAG_REWRITES)

    if grade == "relevant":
        return "generate"
    elif grade == "ambiguous" and rewrites < max_rewrites:
        return "rewrite_query"
    else:
        # irrelevant, or ambiguous with rewrites exhausted
        return "web_search"


# --- Node: rewrite_query ---

REWRITE_SYSTEM_PROMPT = """\
You are a query rewriter. Given the original query and the current query that \
produced poor retrieval results, rewrite it to be more specific and likely to \
retrieve relevant documents.

Respond with ONLY the rewritten query, nothing else.
"""


def rewrite_query(state: RAGState) -> dict:
    """LLM rewrites the query for better retrieval."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Original query: {state['original_query']}\n"
            f"Current query: {state['query']}\n"
            f"This query returned insufficient results. Please rewrite it."
        ),
    ]
    response = llm.invoke(messages)
    new_query = response.content.strip()

    return {
        "query": new_query,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "messages": [response],
    }


# --- Node: web_search ---

def web_search(state: RAGState) -> dict:
    """Fallback: search the web using DuckDuckGo."""
    query = state["query"]

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception:
        results = []

    web_docs = [
        {
            "text": f"{r.get('title', '')}: {r.get('body', '')}",
            "metadata": {"source": r.get("href", ""), "origin": "web_search"},
        }
        for r in results
    ]

    # Merge with any existing docs (web results go first)
    existing = state.get("documents", [])
    return {
        "documents": web_docs + existing,
        "relevance_scores": [0.8] * len(web_docs) + state.get("relevance_scores", []),
        "use_web_search": True,
    }


# --- Node: generate ---

GENERATE_SYSTEM_PROMPT = """\
You are a research assistant. Given a query and a set of documents, synthesize \
a clear, accurate answer. Cite your sources by referencing document numbers.

If the documents don't contain enough information, say so honestly.
"""


def generate(state: RAGState) -> dict:
    """Generate a final answer from the graded/retrieved documents."""
    llm = _get_llm()
    documents = state.get("documents", [])

    doc_text = "\n\n".join(
        f"[Doc {i+1}] {d['text']}" for i, d in enumerate(documents)
    )

    messages = [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Query: {state['query']}\n\nDocuments:\n{doc_text}"
        ),
    ]
    response = llm.invoke(messages)

    citations = [
        d.get("metadata", {}).get("source", f"doc_{i}")
        for i, d in enumerate(documents)
    ]

    return {
        "generation": response.content,
        "citations": citations,
        "messages": [response],
    }
