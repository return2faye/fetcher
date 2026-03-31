# Multi-Agent Research & Code Verification System

## 1. System Overview

A LangGraph-based multi-agent system that accepts complex technical/data-analysis instructions,
autonomously researches the web, generates code, executes it in a sandboxed environment,
and verifies results before delivering a final answer.

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                   SUPERVISOR GRAPH                    │
│                                                       │
│  ┌─────────┐    ┌─────────────┐    ┌──────────────┐  │
│  │ Intake / │───▶│  Router /   │───▶│   Synthesizer│  │
│  │ Planner  │    │  Supervisor │    │   & HITL     │  │
│  └─────────┘    └──────┬──────┘    └──────────────┘  │
│                        │                              │
│            ┌───────────┴───────────┐                  │
│            ▼                       ▼                  │
│  ┌─────────────────┐   ┌─────────────────────┐       │
│  │  SUB-GRAPH 1    │   │   SUB-GRAPH 2        │       │
│  │  Advanced RAG   │   │   Code Gen & Verify  │       │
│  └─────────────────┘   └─────────────────────┘       │
│                                                       │
│  ┌────────────────────────────────────────────┐       │
│  │           SHARED SERVICES                   │       │
│  │  Checkpointer (SQLite) │ Vector DB (Qdrant)│       │
│  │  LangSmith Tracing     │ Token Streaming   │       │
│  └────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
```

---

## 2. LangGraph State Schemas

### 2.1 Supervisor State (Top-Level Graph)

```python
from typing import TypedDict, Literal, Annotated
from langgraph.graph.message import add_messages

class SupervisorState(TypedDict):
    # Core conversation
    messages: Annotated[list, add_messages]
    user_query: str

    # Planning
    plan: list[str]                         # Decomposed sub-tasks
    current_task_index: int
    task_type: Literal["research", "code", "hybrid", "done"]

    # Results from sub-graphs
    research_results: list[dict]            # From RAG sub-graph
    code_results: list[dict]                # From Code sub-graph

    # Final
    final_answer: str
    needs_human_approval: bool
    human_feedback: str | None

    # Meta
    iteration_count: int
    max_iterations: int                     # Safety cap (default 10)
```

### 2.2 RAG Sub-Graph State

```python
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str                              # Current retrieval query
    original_query: str                     # Preserved for rewrite reference

    # Retrieval
    documents: list[dict]                   # Retrieved docs
    relevance_scores: list[float]           # Per-document scores
    relevance_threshold: float              # Default 0.7

    # CRAG control flow
    retrieval_grade: Literal["relevant", "ambiguous", "irrelevant"]
    rewrite_count: int
    max_rewrites: int                       # Default 2
    use_web_search: bool                    # Fallback trigger

    # Output
    generation: str                         # Final RAG answer
    citations: list[str]
```

### 2.3 Code Sub-Graph State

```python
class CodeState(TypedDict):
    messages: Annotated[list, add_messages]
    task_description: str
    context: str                            # Research context fed in

    # Code generation
    generated_code: str
    language: Literal["python", "sql", "shell"]

    # Execution
    execution_result: str
    execution_error: str | None
    exit_code: int | None

    # Self-correction
    retry_count: int
    max_retries: int                        # Default 3
    critic_feedback: str | None

    # Output
    verified_output: str
    is_verified: bool
```

---

## 3. Node & Edge Definitions

### 3.1 Supervisor Graph (Top-Level)

| Node               | Responsibility                                        |
|---------------------|-------------------------------------------------------|
| `intake_planner`    | Parse user query, decompose into sub-tasks, set plan  |
| `router`            | Read current task, decide: research / code / hybrid   |
| `rag_subgraph`      | Invoke RAG sub-graph (compiled as a node)             |
| `code_subgraph`     | Invoke Code sub-graph (compiled as a node)            |
| `synthesizer`       | Merge all sub-results into a coherent final answer    |
| `human_review`      | `interrupt_before` — pause for human approval         |
| `finalize`          | Emit final answer, write to long-term memory          |

**Edges:**

```
START ──▶ intake_planner ──▶ router
router ──▶ rag_subgraph      (if task_type == "research")
router ──▶ code_subgraph     (if task_type == "code")
router ──▶ rag_subgraph      (if task_type == "hybrid", then code_subgraph)
router ──▶ synthesizer        (if task_type == "done")
rag_subgraph ──▶ router       (loop back for next task)
code_subgraph ──▶ router      (loop back for next task)
synthesizer ──▶ human_review  (interrupt_before)
human_review ──▶ finalize ──▶ END
```

### 3.2 RAG Sub-Graph (Corrective RAG)

| Node               | Responsibility                                        |
|---------------------|-------------------------------------------------------|
| `retrieve`          | Query vector DB (Qdrant) for relevant documents       |
| `grade_documents`   | LLM scores each doc for relevance to query            |
| `decide_action`     | Route: generate / rewrite query / web search fallback |
| `rewrite_query`     | LLM rewrites query for better retrieval               |
| `web_search`        | DuckDuckGo fallback search (free, no API key)         |
| `generate`          | LLM generates answer from graded documents            |

**Edges:**

```
START ──▶ retrieve ──▶ grade_documents ──▶ decide_action
decide_action ──▶ generate        (if grade == "relevant")
decide_action ──▶ rewrite_query   (if grade == "ambiguous" & rewrites < max)
decide_action ──▶ web_search      (if grade == "irrelevant" OR rewrites exhausted)
rewrite_query ──▶ retrieve        (loop)
web_search ──▶ generate
generate ──▶ END
```

### 3.3 Code Sub-Graph (Generate-Execute-Verify)

| Node               | Responsibility                                        |
|---------------------|-------------------------------------------------------|
| `coder`             | LLM generates code from task + context                |
| `executor`          | Run code in Docker sandbox, capture stdout/stderr     |
| `critic`            | LLM evaluates output: correct / incorrect / error     |
| `error_handler`     | Extract traceback, format feedback for coder retry    |

**Edges:**

```
START ──▶ coder ──▶ executor ──▶ critic
critic ──▶ END                (if is_verified == True)
critic ──▶ error_handler      (if execution_error or !is_verified)
error_handler ──▶ coder       (if retry_count < max_retries)
error_handler ──▶ END         (if retries exhausted — return partial result)
```

---

## 4. Memory Architecture

| Layer           | Technology          | Purpose                                |
|-----------------|---------------------|----------------------------------------|
| Short-term      | SQLite Checkpointer | Thread-level state, conversation turns  |
| Long-term       | Qdrant (Docker)     | Embedded docs, past research results    |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers, local) | Document embeddings — free, no API key |

---

## 5. Infrastructure

| Component       | Technology              | Notes                           |
|-----------------|-------------------------|---------------------------------|
| Sandbox         | Docker (local)          | Isolated code execution          |
| Web search      | DuckDuckGo (`duckduckgo-search`) | Free, no API key needed |
| LLM             | OpenAI via `langchain-openai` | GPT-4o / GPT-4o-mini          |
| Embeddings      | `sentence-transformers` (`all-MiniLM-L6-v2`) | Local, free   |
| Vector DB       | Qdrant (Docker)         | Long-term memory                 |
| Checkpointer    | SQLite                  | Dev-friendly, swap to Postgres later |
| Observability   | LangSmith (optional)    | Tracing, token tracking          |

---

## 6. Development Phases

### Phase 1 — Project Scaffold & Core State ✅ (Session 1)
- [x] System architecture design
- [x] State schema definitions
- [x] Node/edge specifications
- [x] Development phase breakdown

### Phase 2 — Supervisor Graph & Routing ✅ (Session 2)
- [x] Project scaffold (pyproject.toml, src/fetcher package, conda env)
- [x] Implement `SupervisorState`, `RAGState`, `CodeState` in `state.py`
- [x] Build `intake_planner` node (LLM query decomposition)
- [x] Build `router` node (conditional edges by task_type)
- [x] Wire supervisor graph with stub sub-graph nodes
- [x] SQLite checkpointer integration (`langgraph-checkpoint-sqlite`)
- [x] Unit tests: 5 router tests + 1 full graph integration test (all passing)

### Phase 3 — RAG Sub-Graph (Corrective RAG) ✅ (Session 2)
- [x] Qdrant Docker container setup (port 6333)
- [x] Embedding utility (`all-MiniLM-L6-v2` via sentence-transformers, singleton)
- [x] Qdrant client: `ensure_collection`, `ingest_documents`, `search_documents`
- [x] `retrieve` node — Qdrant similarity search (top-5)
- [x] `grade_documents` node — LLM relevance scoring with vector pre-filter
- [x] `rewrite_query` node — LLM query transformation
- [x] `web_search` node — DuckDuckGo fallback (free, no API key)
- [x] `generate` node — answer synthesis with citations
- [x] `decide_action` conditional edge (relevant → generate, ambiguous → rewrite, irrelevant → web)
- [x] CRAG graph wired: retrieve → grade → decide → {generate | rewrite→retrieve | web→generate}
- [x] 9 tests (4 unit + 2 web_search + 3 integration paths) — all passing

### Phase 4 — Code Sub-Graph (Generate-Execute-Verify) ✅ (Session 4)
- [x] Docker sandbox: `Dockerfile.sandbox` (Python 3.11-slim + numpy/pandas/matplotlib)
- [x] Docker Compose: sandbox container added (network_mode: none, mem_limit: 512m)
- [x] `docker_sandbox.py` utility — execute code in container, capture stdout/stderr/exit_code
- [x] `coder` node — LLM generates code (first attempt + retry with error feedback)
- [x] `executor` node — runs code in Docker sandbox
- [x] `critic` node — LLM evaluates output correctness (JSON verdict)
- [x] `error_handler` node — traceback extraction, retry count increment
- [x] `should_retry` conditional edge (verified → end, retries left → retry, exhausted → end)
- [x] Code graph wired: coder → executor → critic → {END | error_handler → coder}
- [x] 13 tests (3 helpers + 3 routing + 1 error_handler + 3 executor/sandbox + 3 integration) — all passing

### Phase 5 — Integration & Memory ✅ (Session 5)
- [x] Integration layer: `nodes/integration.py` — adapter functions for RAG, Code, Hybrid
- [x] `build_supervisor_graph(use_stubs=False)` wires real sub-graphs; `use_stubs=True` for unit tests
- [x] End-to-end flows tested: research-only, code-only, hybrid, multi-task plan
- [x] Long-term memory: `utils/memory.py` — separate Qdrant collection (`fetcher_memory`)
- [x] Memory is best-effort: silently degrades if Qdrant unavailable
- [x] Cross-sub-graph context: research results passed as context to Code coder node
- [x] Past results recalled via vector similarity before each sub-graph invocation
- [x] 9 new tests (38 total) — all passing

### Phase 6 — HITL, Streaming & Observability
- [ ] `interrupt_before` on `human_review` node
- [ ] Token-level streaming (`astream_events`)
- [ ] LangSmith tracing configuration
- [ ] Human feedback → re-route or approve flow

### Phase 7 — Hardening & Polish
- [ ] Error handling and graceful degradation
- [ ] Iteration safety caps enforcement
- [ ] Comprehensive integration tests
- [ ] Docker Compose for full stack (Qdrant + sandbox)
- [ ] README with setup and usage instructions
