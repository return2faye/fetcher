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

### Next Steps (Session 3 — Phase 3: RAG Sub-Graph)
1. Start Qdrant Docker container (`docker run qdrant/qdrant`)
2. Build document ingestion pipeline: chunking → embed with `all-MiniLM-L6-v2` → upsert to Qdrant
3. Implement `retrieve` node — Qdrant similarity search
4. Implement `grade_documents` node — LLM relevance scoring
5. Implement `decide_action` conditional edge (relevant / rewrite / web search)
6. Implement `rewrite_query` node — LLM query transformation
7. Implement `web_search` node — DuckDuckGo fallback
8. Implement `generate` node — answer synthesis with citations
9. Wire RAG sub-graph with CRAG conditional edges
10. Integration test: full CRAG loop
