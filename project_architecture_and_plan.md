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

### Phase 1 — Project Scaffold & Core State ✅ (Session 1 — Design only)
- [x] System architecture design
- [x] State schema definitions
- [x] Node/edge specifications
- [x] Development phase breakdown
- [ ] Project scaffold (pyproject.toml, directory structure)

### Phase 2 — Supervisor Graph & Routing
- [ ] Implement `SupervisorState`
- [ ] Build `intake_planner` node (query decomposition)
- [ ] Build `router` node (conditional edges)
- [ ] Wire supervisor graph with placeholder sub-graph nodes
- [ ] SQLite checkpointer integration
- [ ] Unit tests for routing logic

### Phase 3 — RAG Sub-Graph (Corrective RAG)
- [ ] Qdrant Docker container setup
- [ ] Document ingestion pipeline (chunking, embedding, upsert)
- [ ] `retrieve` node — Qdrant similarity search
- [ ] `grade_documents` node — LLM relevance scoring
- [ ] `rewrite_query` node — query transformation
- [ ] `web_search` node — Tavily API integration
- [ ] `generate` node — answer synthesis with citations
- [ ] Conditional edge logic (CRAG flow)
- [ ] Integration test: full RAG loop

### Phase 4 — Code Sub-Graph (Generate-Execute-Verify)
- [ ] Docker sandbox setup (execution environment)
- [ ] `coder` node — LLM code generation
- [ ] `executor` node — Docker-based execution, stdout/stderr capture
- [ ] `critic` node — output evaluation
- [ ] `error_handler` node — traceback extraction, retry routing
- [ ] Self-correction loop wiring
- [ ] Integration test: code gen → execute → verify cycle

### Phase 5 — Integration & Memory
- [ ] Connect sub-graphs to supervisor as compiled nodes
- [ ] End-to-end supervisor loop (plan → research → code → synthesize)
- [ ] Long-term memory: store/retrieve past results from Qdrant
- [ ] Cross-sub-graph context passing (research → code)

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
