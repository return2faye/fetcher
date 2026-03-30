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

### Unresolved / Open Questions
- [ ] Which LLM provider/model for agent reasoning? (Claude via `langchain-anthropic` assumed — confirm API key availability)
- [ ] Embedding model choice: OpenAI `text-embedding-3-small` (paid) vs `nomic-embed-text` (local/free)?
- [ ] Tavily API key — needed for Phase 3
- [ ] Docker availability on host — needed for Phase 4

### Dependencies (to install in Phase 2)
```
langgraph >= 0.2
langchain-core
langchain-anthropic          # or langchain-openai
langchain-community
tavily-python
qdrant-client
docker                       # Python Docker SDK
```

### Next Steps (Session 2 — Phase 2)
1. Create project scaffold: `pyproject.toml`, `src/` package structure, `.env.example`
2. Implement `SupervisorState` dataclass
3. Build `intake_planner` node (LLM decomposes query into sub-tasks)
4. Build `router` node with conditional edges
5. Wire skeleton supervisor graph with stub sub-graph nodes
6. Add SQLite checkpointer
7. Write first test: query → plan → route → stub result
