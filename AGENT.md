# Agent Context — Fetcher

> This file exists so that any AI assistant (including future me) can pick up this project
> mid-stream and understand what is happening, what has been built, and what to do next.

## What is this project?

**Fetcher** is a multi-agent system built with LangGraph. It takes complex technical or
data-analysis questions from a user, autonomously researches the web, generates code,
executes it in an isolated sandbox, and verifies the results before returning a final answer.

## Architecture (3 sentences)

A **Supervisor graph** decomposes the user query into a plan of typed sub-tasks (research,
code, hybrid) and routes each task to one of two sub-graphs. The **Corrective RAG (CRAG)
sub-graph** retrieves from a Qdrant vector DB, LLM-grades relevance, rewrites queries or
falls back to DuckDuckGo web search, then generates an answer. The **Code sub-graph**
(not yet built) will generate code, execute it in a Docker sandbox, and self-correct on
failure.

## Tech stack

| What            | Choice                                       |
|-----------------|----------------------------------------------|
| Framework       | LangGraph (StateGraph, conditional edges)    |
| LLM             | OpenAI `gpt-4o-mini` (fast) / `gpt-4o` (heavy) via `langchain-openai` |
| Embeddings      | `all-MiniLM-L6-v2` via `sentence-transformers` (local, free) |
| Vector DB       | Qdrant (Docker container, port 6333)         |
| Web search      | `duckduckgo-search` (free, no API key)       |
| Code sandbox    | Docker container (Phase 4)                   |
| Checkpointer    | SQLite via `langgraph-checkpoint-sqlite`     |
| Python env      | Conda env named `fetcher`, Python 3.11       |

**Important constraints from user:**
- No Tavily API key — use DuckDuckGo for all web search
- No paid embedding APIs — use local sentence-transformers only
- OpenAI API key is available (the only paid external service)

## Project structure

```
src/fetcher/
├── config.py                  # All env vars, model names, constants
├── state.py                   # 3 TypedDict schemas: SupervisorState, RAGState, CodeState
├── graphs/
│   ├── supervisor.py          # Top-level graph: plan → route → sub-graphs → synthesize
│   └── rag.py                 # CRAG sub-graph: retrieve → grade → decide → generate
├── nodes/
│   ├── supervisor.py          # intake_planner, router, stubs, synthesizer, finalize
│   └── rag.py                 # retrieve, grade_documents, rewrite_query, web_search, generate
└── utils/
    ├── embeddings.py          # Singleton embedding model, embed_texts(), embed_query()
    └── qdrant_client.py       # Qdrant ops: ensure_collection, ingest, search
tests/
├── test_supervisor.py         # 7 tests (router logic + full graph integration)
└── test_rag.py                # 9 tests (decide_action + web_search + 3 CRAG paths)
```

## Current state of development

| Phase | Description                      | Status  |
|-------|----------------------------------|---------|
| 1     | Architecture design              | Done    |
| 2     | Supervisor graph & routing       | Done    |
| 3     | Corrective RAG sub-graph         | Done    |
| 4     | Code generation & Docker sandbox | **Next** |
| 5     | Integration & memory             | Pending |
| 6     | HITL, streaming & observability  | Pending |
| 7     | Hardening & polish               | Pending |

**16 tests, all passing.** Run with: `conda activate fetcher && PYTHONPATH=src pytest tests/ -v`

## How the graphs work

### Supervisor flow
```
user query → intake_planner (LLM decomposes into [type] tasks)
  → router (reads plan[current_index], sets task_type)
    → rag_subgraph   (if research) → router (advance index)
    → code_subgraph  (if code)     → router (advance index)
    → hybrid         (if hybrid)   → router (advance index)
    → synthesizer    (if done — all tasks complete)
      → human_review (HITL gate, Phase 6)
        → finalize → END
```

### CRAG flow
```
query → retrieve (Qdrant top-5)
  → grade_documents (LLM scores each doc)
    → decide_action:
      - "relevant" (2+ good docs)    → generate → END
      - "ambiguous" (1 doc, retries left) → rewrite_query → retrieve (loop)
      - "irrelevant" (0 docs or retries exhausted) → web_search → generate → END
```

## What to do next (Phase 4)

Build the Code Generation & Verification sub-graph in this order:
1. Docker sandbox setup — a Python Docker image for isolated code execution
2. `coder` node — LLM generates code from task description + research context
3. `executor` node — run code in Docker, capture stdout/stderr/exit_code
4. `critic` node — LLM evaluates: is the output correct?
5. `error_handler` node — extract traceback, format feedback for retry
6. Wire the self-correction loop: coder → executor → critic → {END | error_handler → coder}
7. Tests for the full cycle

After Phase 4, Phase 5 wires both sub-graphs into the supervisor (replacing stubs).

## Container management

Use `./scripts/start.sh` and `./scripts/stop.sh` to manage Docker containers (Qdrant, and
later the code sandbox). See `docker/docker-compose.yml` for the full stack definition.

## Session workflow

The user expects incremental development across sessions. At the end of every session:
1. Update `project_architecture_and_plan.md` — check off completed items
2. Update `dev_log.md` — add a new session entry with what was done, decisions, next steps
3. Commit with a descriptive message
4. All tests must pass before committing
