# Development Log

## Session 1 — 2026-03-30

### Status: Architecture Design Complete

### What was done
- Designed full system architecture: Supervisor → 2 sub-graphs (RAG, Code)
- Defined all three LangGraph state schemas (`SupervisorState`, `RAGState`, `CodeState`)
- Specified all nodes (14 total) and their edge routing logic
- Chose tech stack: SQLite checkpointer, Qdrant (Docker), Tavily, Docker sandbox
- Broke development into 7 sequential phases

### Key Design Decisions
1. **Supervisor pattern over flat multi-agent** — cleaner task decomposition, the supervisor owns the plan and delegates to specialized sub-graphs.
2. **CRAG over Self-RAG** — CRAG's explicit retrieve→grade→decide flow maps cleanly to LangGraph nodes. Self-RAG's inline citation scoring adds complexity without proportional benefit at this stage.
3. **Docker sandbox over E2B** — local Docker keeps everything self-hosted, no external API dependency for code execution. Can swap to E2B later if needed.
4. **SQLite checkpointer for dev** — zero-setup, sufficient for single-user dev. Postgres migration is a config change when needed.
5. **Qdrant for vector DB** — runs in Docker alongside sandbox, good Python SDK, supports filtering.

### Resolved Questions (Session 1 follow-up)
- [x] **LLM**: OpenAI (API key available) — use `langchain-openai`, models: `gpt-4o` / `gpt-4o-mini`
- [x] **Embeddings**: Local `sentence-transformers` with `all-MiniLM-L6-v2` — free, no API key
- [x] **Web search**: DuckDuckGo (`duckduckgo-search`) — free, no API key (replaces Tavily)
- [x] **Docker**: Installed and available

### Dependencies (to install in Phase 2)
```
langgraph >= 0.2
langchain-core
langchain-openai
langchain-community
sentence-transformers        # local embeddings (all-MiniLM-L6-v2)
duckduckgo-search            # free web search fallback
qdrant-client
docker                       # Python Docker SDK
```

### Next Steps (Session 2 — Phase 2) ✅ DONE — see Session 2 below

---

## Session 2 — 2026-03-30

### Status: Phase 2 Complete — Supervisor Graph & Routing

### What was done
- Created conda environment `fetcher` (Python 3.11) with all dependencies
- Project scaffold: `pyproject.toml`, `src/fetcher/` package, `.env.example`, `.gitignore`
- Implemented all 3 state schemas in `src/fetcher/state.py`
- Built `intake_planner` node — LLM decomposes query into typed sub-tasks (JSON output)
- Built `router` node — reads plan, advances index, sets task_type
- Built `route_by_task_type` conditional edge function
- Built stub nodes: `rag_subgraph_stub`, `code_subgraph_stub`, `hybrid_stub`
- Built `synthesizer` node — merges research + code results via LLM
- Built `human_review` (placeholder) and `finalize` nodes
- Wired full supervisor graph in `src/fetcher/graphs/supervisor.py`
- SQLite checkpointer via `langgraph-checkpoint-sqlite`
- 7 tests (5 unit + 1 edge function + 1 integration with mocked LLM) — all passing

### Project Structure
```
fetcher/
├── pyproject.toml
├── .env.example
├── .gitignore
├── project_architecture_and_plan.md
├── dev_log.md
├── src/fetcher/
│   ├── __init__.py
│   ├── config.py              # env vars, model names, constants
│   ├── state.py               # SupervisorState, RAGState, CodeState
│   ├── graphs/
│   │   ├── __init__.py
│   │   └── supervisor.py      # build_supervisor_graph(), compile_supervisor()
│   ├── nodes/
│   │   ├── __init__.py
│   │   └── supervisor.py      # intake_planner, router, stubs, synthesizer
│   └── utils/
│       └── __init__.py
└── tests/
    ├── __init__.py
    └── test_supervisor.py     # 7 tests, all passing
```

### Key Design Decisions (Session 2)
1. **gpt-4o-mini for planning, gpt-4o for synthesis** — cheaper model handles decomposition, heavier model handles final answer quality.
2. **JSON-only planner prompt** — avoids parsing ambiguity; falls back to single research task if JSON parse fails.
3. **Stub sub-graphs return mock results** — allows full graph loop testing without real LLM/Docker/Qdrant.
4. **Needed `langgraph-checkpoint-sqlite`** — separate package from `langgraph`, not bundled.

### Dependencies Installed
```
langgraph 1.1.3, langchain-core 1.2.23, langchain-openai 1.1.12
langchain-community 0.4.1, sentence-transformers 5.3.0
duckduckgo-search 8.1.1, qdrant-client 1.17.1, docker 7.1.0
langgraph-checkpoint-sqlite 3.0.3
pytest 9.0.2, pytest-asyncio 1.3.0
```

### Next Steps (Session 3 — Phase 3) ✅ DONE — see Session 3 below

---

## Session 3 — 2026-03-30

### Status: Phase 3 Complete — Corrective RAG Sub-Graph

### What was done
- Started Qdrant Docker container (port 6333, named volume `qdrant_storage`)
- Built embedding utility: `utils/embeddings.py` — lazy-loaded `all-MiniLM-L6-v2` singleton
- Built Qdrant client: `utils/qdrant_client.py` — `ensure_collection`, `ingest_documents`, `search_documents`
- Verified end-to-end: ingest 3 docs → search → correct top result (score 0.741)
- Built all 6 RAG nodes in `nodes/rag.py`:
  - `retrieve` — Qdrant top-5 similarity search
  - `grade_documents` — LLM grades each doc with vector score pre-filter
  - `decide_action` — conditional edge: relevant→generate, ambiguous→rewrite, irrelevant→web
  - `rewrite_query` — LLM rewrites query for better retrieval
  - `web_search` — DuckDuckGo fallback with error handling
  - `generate` — LLM synthesizes answer with doc citations
- Wired full CRAG sub-graph in `graphs/rag.py` with conditional edges and rewrite loop
- 9 new tests (16 total), all passing:
  - 4 `decide_action` unit tests (all routing paths)
  - 2 `web_search` tests (success + failure handling)
  - 3 integration tests (relevant path, irrelevant→web fallback, ambiguous→rewrite→relevant)

### New Files
```
src/fetcher/utils/embeddings.py     # Local embedding model (all-MiniLM-L6-v2)
src/fetcher/utils/qdrant_client.py  # Qdrant operations (ingest, search)
src/fetcher/nodes/rag.py            # All 6 CRAG nodes
src/fetcher/graphs/rag.py           # CRAG sub-graph wiring
tests/test_rag.py                   # 9 tests
```

### Key Design Decisions (Session 3)
1. **Vector score pre-filter in grading** — skip LLM grading for docs with score < 0.5 × threshold. Saves tokens.
2. **Grading heuristic**: 2+ relevant docs = "relevant", 1 = "ambiguous", 0 = "irrelevant". Simple and effective.
3. **DuckDuckGo `with` context manager** — clean resource handling, graceful fallback on network errors.
4. **Embedding model singleton** — avoids reloading the 80MB model on every call.

### Next Steps (Session 4 — Phase 4) ✅ DONE — see Session 4 below

---

## Session 4 — 2026-03-31

### Status: Phase 4 Complete — Code Generation & Docker Sandbox

### What was done
- Built `Dockerfile.sandbox` — Python 3.11-slim with numpy, pandas, requests, matplotlib
- Added sandbox to `docker-compose.yml` (network_mode: none, mem_limit: 512m, non-root user)
- Built `utils/docker_sandbox.py` — execute code in container via `docker exec`, capture stdout/stderr/exit_code
- Built all 4 code nodes in `nodes/code.py`:
  - `coder` — LLM generates code with fenced block extraction; separate prompt for retries
  - `executor` — delegates to Docker sandbox, handles missing container and empty code
  - `critic` — LLM JSON verdict (pass/fail); skips LLM call if execution already errored
  - `error_handler` — extracts traceback, increments retry count, formats feedback
- `should_retry` conditional edge: verified → end, retries left → retry, exhausted → end
- Wired full Code sub-graph in `graphs/code.py`
- Also created AGENT.md, docker-compose, shell scripts, design_rationale.md (pre-Phase 4 tasks)
- 13 new tests (29 total), all passing

### New Files
```
docker/Dockerfile.sandbox              # Sandbox image definition
src/fetcher/utils/docker_sandbox.py    # Docker exec wrapper
src/fetcher/nodes/code.py              # All 4 code sub-graph nodes
src/fetcher/graphs/code.py             # Code sub-graph wiring
tests/test_code.py                     # 13 tests
AGENT.md                               # Agent context document
docs/design_rationale.md               # Design rationale report
scripts/start.sh, stop.sh, status.sh   # Container management
```

### Key Design Decisions (Session 4)
1. **Long-running sandbox container** — `sleep infinity` keeps the container alive; we `docker exec` into it. Avoids cold-start overhead of creating a new container per execution.
2. **network_mode: none** — sandbox has no internet access. Code can't exfiltrate data or make unexpected API calls.
3. **Critic skips LLM on execution errors** — if exit_code != 0, the error is already clear. No need to spend tokens asking the LLM to confirm.
4. **Separate coder prompts for first attempt vs retry** — retry prompt includes previous code + error feedback, focusing the LLM on fixing the specific issue.
5. **Regex code block extraction** — handles ```python, bare ```, and no-fence fallback. Robust to varying LLM output formats.

### Next Steps (Session 5 — Phase 5) ✅ DONE — see Session 5 below

---

## Session 5 — 2026-03-31

### Status: Phase 5 Complete — Integration & Memory

### What was done
- Built `nodes/integration.py` — adapter layer translating SupervisorState ↔ sub-graph states
  - `rag_node`: invokes compiled RAG sub-graph, stores result in long-term memory
  - `code_node`: invokes compiled Code sub-graph with research context, stores result
  - `hybrid_node`: RAG then Code sequentially for a single task
- Refactored `graphs/supervisor.py` — `build_supervisor_graph(use_stubs=False)` uses real sub-graphs; `use_stubs=True` preserves unit test compatibility
- Built `utils/memory.py` — long-term memory via separate Qdrant collection (`fetcher_memory`)
  - `store_result()`: embeds and upserts task+result with UUID
  - `recall_context()`: retrieves relevant past results by vector similarity
  - Best-effort: all operations silently fail if Qdrant unavailable
- Cross-sub-graph context: all research results concatenated and passed as `context` to Code coder
- Past context recall: before each sub-graph invocation, recall relevant memories from Qdrant
- 9 new tests (38 total), all passing:
  - 2 helper unit tests
  - 4 end-to-end integration tests (research, code, hybrid, multi-task)
  - 3 memory tests (graceful degradation + live Qdrant store/recall)

### New Files
```
src/fetcher/nodes/integration.py    # Adapter layer: SupervisorState ↔ sub-graphs
src/fetcher/utils/memory.py         # Long-term memory (Qdrant fetcher_memory collection)
tests/test_integration.py           # 9 integration tests
```

### Key Design Decisions (Session 5)
1. **Adapter pattern over direct sub-graph embedding** — sub-graphs have their own state schemas. Adapter functions translate between SupervisorState and RAGState/CodeState, keeping sub-graphs self-contained and independently testable.
2. **`use_stubs` flag** — existing unit tests continue to work without Docker/Qdrant. Real sub-graphs used by default for production, stubs for fast isolated testing.
3. **Separate memory collection** — `fetcher_memory` is distinct from `fetcher_docs`. User-ingested documents don't mix with system-generated memories.
4. **Best-effort memory** — memory operations silently fail rather than crashing the pipeline. Memory is a quality enhancement, not a critical path.
5. **Context concatenation** — all research results are joined and passed as code context. Simple and effective; could be improved with selective context later.

### Next Steps (Session 6 — Phase 6: HITL, Streaming & Observability) ✅ DONE — see Session 6 below

---

## Session 6 — 2026-03-31

### Status: Phase 6 Complete — HITL, Streaming & Observability

### What was done
- Implemented HITL gate using LangGraph's `interrupt()` API in `human_review` node
- Three feedback paths: approve → finalize, reject → re-plan from scratch, revise → re-synthesize with feedback
- Built `revise_synthesis` node — re-generates answer incorporating human revision instructions
- Built `route_after_human_review` conditional edge for feedback-based routing
- Wired conditional edges: `human_review` → {finalize, revise_synthesis, intake_planner}
- Revision loop: `revise_synthesis` → `human_review` (allows multiple revision rounds)
- LangSmith tracing: env-var gated in `config.py` — set `LANGSMITH_API_KEY` to enable
- Token-level streaming via `astream_events` (v2) in CLI runner
- Built CLI runner (`src/fetcher/cli.py`) with two modes:
  - `python -m fetcher.cli "query"` — sync mode with HITL prompts
  - `python -m fetcher.cli --stream "query"` — streaming mode with real-time token output
- 9 new tests (47 total), all passing:
  - 4 unit tests for `route_after_human_review` (approve, default, reject, revision)
  - 1 unit test for `revise_synthesis` (verifies feedback in prompt)
  - 3 integration tests: approve flow, revision flow (2 interrupt cycles), reject/re-plan flow
  - 1 LangSmith config test (env-var gating)

### New Files
```
src/fetcher/cli.py             # CLI runner with HITL + streaming
tests/test_hitl.py             # 9 tests for HITL and config
```

### Modified Files
```
src/fetcher/config.py          # Added LangSmith tracing config
src/fetcher/nodes/supervisor.py # Added interrupt(), revise_synthesis, route_after_human_review
src/fetcher/graphs/supervisor.py # Conditional edges for HITL, revise_synthesis node
```

### Key Design Decisions (Session 6)
1. **`interrupt()` over `interrupt_before`** — LangGraph's `interrupt()` function (called inside the node) is more flexible than `interrupt_before` (graph-level config). It lets us pass context (the answer, plan) to the human and receive structured feedback as the return value.
2. **Three-way feedback routing** — approve/reject/revise covers all practical HITL scenarios. Reject triggers a full re-plan, which is expensive but appropriate for fundamentally wrong answers. Revise re-synthesizes cheaply from existing sub-results.
3. **Revision loop** — `revise_synthesis → human_review` allows unlimited revision rounds. The human stays in control until satisfied.
4. **LangSmith via env-var gating** — no code changes needed to enable tracing. Just set `LANGSMITH_API_KEY`. The `setdefault` calls ensure we don't override user's explicit settings.
5. **CLI dual mode** — sync mode is simpler and works for debugging. Streaming mode (`--stream`) gives real-time token output for a better UX. Both support HITL interrupt/resume.

### Next Steps (Session 7 — Phase 7: Hardening & Polish)
1. Error handling and graceful degradation across all nodes
2. Timeout handling for LLM calls and Docker execution
3. Input validation on user queries
4. CLI polish: `--help`, progress indicators, better error messages
5. Document ingestion pipeline (chunking strategy)
