# Agent Context — Fetcher

> **Who is this for?** You — the AI agent working on this project. This file tells you
> who you are, what the project is, what has been built, and what to do next.
>
> **This is not the place for reasoning.** If you are making a design decision, evaluating
> trade-offs, or choosing between alternatives — write that in `docs/design_rationale.md`.
> That file is a shared thinking space: it helps the user understand your reasoning, and
> it helps you revisit and challenge your own past decisions. Keep this file operational;
> keep that file analytical.

## What is this project?

**Fetcher** is a multi-agent system built with LangGraph. It takes complex technical or
data-analysis questions from a user, autonomously researches the web, generates code,
executes it in an isolated sandbox, and verifies the results before returning a final answer.

## Architecture (4 sentences)

A **Supervisor graph** decomposes the user query into a plan of typed sub-tasks (research,
code, hybrid) and routes each task to one of two sub-graphs. The **Corrective RAG (CRAG)
sub-graph** retrieves from a Qdrant vector DB, LLM-grades relevance, rewrites queries or
falls back to DuckDuckGo web search, then generates an answer. The **Code sub-graph**
generates code, executes it in a Docker sandbox, and self-corrects on failure. A **HITL
gate** pauses after synthesis for human approval, with options to approve, reject (re-plan),
or request revisions.

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
├── config.py                  # All env vars, model names, constants, LangSmith config
├── state.py                   # 3 TypedDict schemas: SupervisorState, RAGState, CodeState
├── cli.py                     # CLI runner with HITL + streaming support
├── graphs/
│   ├── supervisor.py          # Top-level graph: plan → route → sub-graphs → HITL → finalize
│   ├── rag.py                 # CRAG sub-graph: retrieve → grade → decide → generate
│   └── code.py                # Code sub-graph: coder → executor → critic → retry
├── nodes/
│   ├── supervisor.py          # intake_planner, router, stubs, synthesizer, human_review, revise, finalize
│   ├── rag.py                 # retrieve, grade_documents, rewrite_query, web_search, generate
│   ├── code.py                # coder, executor, critic, error_handler
│   └── integration.py         # Adapters: rag_node, code_node, hybrid_node (real sub-graphs)
└── utils/
    ├── embeddings.py          # Singleton embedding model, embed_texts(), embed_query()
    ├── qdrant_client.py       # Qdrant ops: ensure_collection, ingest, search
    ├── docker_sandbox.py      # Docker exec wrapper for code execution
    └── memory.py              # Long-term memory: store/recall via Qdrant (fetcher_memory)
docker/
├── docker-compose.yml         # Qdrant + sandbox containers
└── Dockerfile.sandbox         # Python 3.11-slim sandbox image
scripts/
├── start.sh                   # Start all containers
├── stop.sh                    # Stop all containers
└── status.sh                  # Show container status
tests/
├── test_supervisor.py         # 7 tests (router logic + full graph with stubs)
├── test_rag.py                # 9 tests (decide_action + web_search + 3 CRAG paths)
├── test_code.py               # 13 tests (helpers + routing + sandbox + integration)
├── test_integration.py        # 9 tests (end-to-end with real sub-graphs + memory)
├── test_hitl.py               # 9 tests (HITL routing, feedback, interrupt/resume flows)
└── test_hardening.py          # 19 tests (error handling, timeouts, input validation)
```

## Current state of development

| Phase | Description                      | Status  |
|-------|----------------------------------|---------|
| 1     | Architecture design              | Done    |
| 2     | Supervisor graph & routing       | Done    |
| 3     | Corrective RAG sub-graph         | Done    |
| 4     | Code generation & Docker sandbox | Done    |
| 5     | Integration & memory             | Done    |
| 6     | HITL, streaming & observability  | Done    |
| 7     | Hardening & polish               | Done    |

**66 tests, all passing.** Run with: `conda activate fetcher && PYTHONPATH=src pytest tests/ -v`

## How the graphs work

### Supervisor flow
```
user query → intake_planner (LLM decomposes into [type] tasks)
  → router (reads plan[current_index], sets task_type)
    → rag_subgraph   (if research) → router (advance index)
    → code_subgraph  (if code)     → router (advance index)
    → hybrid         (if hybrid)   → router (advance index)
    → synthesizer    (if done — all tasks complete)
      → human_review (interrupt — waits for user feedback)
        → "approve"           → finalize → END
        → "reject:<reason>"   → intake_planner (re-plan)
        → "<revision notes>"  → revise_synthesis → human_review (loop)
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

### Code flow
```
task → coder (LLM generates Python)
  → executor (Docker sandbox, captures stdout/stderr)
    → critic (LLM evaluates output):
      - "pass"  → END (verified_output set)
      - "fail"  → error_handler → coder (retry with feedback)
      - retries exhausted → END (is_verified=False)
```

## What to do next (future work)

Potential improvements:
1. Document ingestion pipeline (chunking strategy for PDFs, web pages)
2. Adaptive retrieval (variable top-k based on query complexity)
3. Expand sandbox packages or add dynamic `pip install`
4. Multi-user support (Postgres checkpointer, auth)
5. Web UI or API server for non-CLI access
6. Selective context: embed task descriptions and pick most relevant research for each code task

## Container management

Use `./scripts/start.sh` and `./scripts/stop.sh` to manage Docker containers (Qdrant +
sandbox). See `docker/docker-compose.yml` for the full stack definition.

## Session workflow

The user expects incremental development across sessions. At the end of every session:
1. Update `project_architecture_and_plan.md` — check off completed items
2. Update `dev_log.md` — add a new session entry with what was done, decisions, next steps
3. Update `docs/design_rationale.md` — add a new section for every non-trivial design
   decision made during the session. Explain what you chose, what you rejected, and why.
   This is not optional. The user reads this to understand your thinking, and you read it
   to check whether past decisions still hold.
4. Commit with a descriptive message
5. All tests must pass before committing

## Key files and their purposes

| File | Purpose | Who reads it |
|------|---------|-------------|
| `AGENT.md` (this file) | Operational context: what exists, what to do next | Agent |
| `docs/design_rationale.md` | Analytical: why each decision was made, trade-offs, things to revisit | Agent + User |
| `project_architecture_and_plan.md` | Phase checklist and architecture spec | Agent + User |
| `dev_log.md` | Session-by-session progress log | Agent + User |
