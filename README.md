# Fetcher

A multi-agent system that takes complex technical questions, autonomously researches the web, generates code, executes it in a sandboxed environment, and verifies the results before delivering a final answer.

Built with [LangGraph](https://github.com/langchain-ai/langgraph).

## How it works

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│              Supervisor Graph                     │
│                                                   │
│  Planner ──▶ Router ──▶ Synthesizer ──▶ HITL     │
│                 │                         │       │
│        ┌───────┴────────┐          approve/revise │
│        ▼                ▼          /reject        │
│   RAG Sub-graph   Code Sub-graph        │        │
│   (CRAG flow)     (sandbox + verify)    ▼        │
│                                      Finalize     │
└──────────────────────────────────────────────────┘
```

1. **Supervisor** decomposes your query into typed sub-tasks (research, code, hybrid)
2. **Corrective RAG** retrieves from a vector DB, grades relevance, rewrites queries or falls back to DuckDuckGo web search
3. **Code Engine** generates Python, runs it in an isolated Docker sandbox, and self-corrects on failure
4. **Synthesizer** merges all results into a coherent final answer
5. **Human Review** pauses for your approval — you can approve, request revisions, or reject and re-plan

## Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/) (or any Python environment manager)
- [Docker](https://docs.docker.com/get-docker/) with Docker Compose
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
# 1. Create and activate conda environment
conda create -n fetcher python=3.11 -y
conda activate fetcher

# 2. Install the package
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Start Docker containers (Qdrant vector DB + code sandbox)
./scripts/start.sh
```

The start script launches two containers:

| Container | Purpose | Port |
|-----------|---------|------|
| `fetcher-qdrant` | Qdrant vector DB for document retrieval and long-term memory | 6333 |
| `fetcher-sandbox` | Isolated Python sandbox for code execution (no network access) | — |

Verify they're running:

```bash
./scripts/status.sh
```

## Usage

### Interactive mode

```bash
conda activate fetcher
PYTHONPATH=src python -m fetcher.cli
```

You'll be prompted to enter a query. The system will plan, research, and/or write code, then ask for your approval before finalizing.

### Pass a query directly

```bash
PYTHONPATH=src python -m fetcher.cli "Explain quicksort and write a Python benchmark"
```

### Streaming mode

Stream LLM tokens in real-time as they're generated:

```bash
PYTHONPATH=src python -m fetcher.cli --stream "Compare merge sort and quicksort performance"
```

### Human-in-the-loop review

After the system synthesizes an answer, it pauses and shows you the result. You have three options:

| Input | What happens |
|-------|-------------|
| `Enter` (empty) or `approve` | Accept the answer and finalize |
| `reject:<reason>` | Reject and re-plan the entire query with your feedback |
| Any other text | Revise — the system re-synthesizes incorporating your feedback, then asks again |

Example session:

```
Processing: Explain Python decorators and write an example

  [intake_planner] ...
  [router] ...
  [rag_subgraph] ...
  [synthesizer] ...

--- Execution Plan ---
  1. [research] Explain Python decorators
  2. [code] Write a Python decorator example

--- Final Answer (pending approval) ---
Python decorators are functions that modify other functions...

--- Human Review ---
Options:
  [Enter]          -> Approve
  reject:<reason>  -> Reject and re-plan
  <your feedback>  -> Revise with instructions

Your feedback: Add a section about class decorators too
```

### Ingesting documents

To add documents to the vector DB for retrieval:

```python
from fetcher.utils.qdrant_client import ensure_collection, ingest_documents

ensure_collection()
ingest_documents(
    texts=["Document text here...", "Another document..."],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}],
)
```

## Configuration

All configuration is via environment variables (set in `.env` or your shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for planning, grading, and code critique |
| `OPENAI_MODEL_HEAVY` | `gpt-4o` | Model for synthesis and code generation |
| `LLM_TIMEOUT` | `60` | LLM call timeout in seconds |
| `DOCKER_EXEC_TIMEOUT` | `30` | Code sandbox execution timeout in seconds |
| `MAX_QUERY_LENGTH` | `10000` | Max characters for user queries |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `LANGSMITH_API_KEY` | (empty) | Set to enable LangSmith tracing |
| `LANGSMITH_PROJECT` | `fetcher` | LangSmith project name |

## Container management

```bash
# Start Qdrant + sandbox
./scripts/start.sh

# Check status
./scripts/status.sh

# Stop all containers
./scripts/stop.sh
```

The sandbox container runs with:
- **No network access** (`network_mode: none`) — generated code cannot make external requests
- **512 MB memory limit** — prevents runaway processes
- **Non-root user** — defense in depth
- **Pre-installed packages**: numpy, pandas, matplotlib, requests (requests can't connect due to no network)

## Running tests

```bash
conda activate fetcher
PYTHONPATH=src pytest tests/ -v
```

66 tests covering: supervisor routing, CRAG retrieval paths, code self-correction, HITL interrupt/resume flows, error handling, timeouts, and input validation.

## Project structure

```
src/fetcher/
├── cli.py                     # CLI with HITL + streaming
├── config.py                  # Environment variables and constants
├── state.py                   # LangGraph state schemas
├── graphs/
│   ├── supervisor.py          # Top-level orchestrator graph
│   ├── rag.py                 # Corrective RAG sub-graph
│   └── code.py                # Code generation sub-graph
├── nodes/
│   ├── supervisor.py          # Planner, router, synthesizer, HITL nodes
│   ├── rag.py                 # Retrieve, grade, rewrite, search, generate
│   ├── code.py                # Coder, executor, critic, error handler
│   └── integration.py         # Adapters wiring sub-graphs into supervisor
└── utils/
    ├── embeddings.py          # Local embedding model (all-MiniLM-L6-v2)
    ├── qdrant_client.py       # Vector DB operations
    ├── docker_sandbox.py      # Docker sandbox execution with timeout
    └── memory.py              # Long-term memory via Qdrant
docker/
├── docker-compose.yml         # Qdrant + sandbox container definitions
└── Dockerfile.sandbox         # Python 3.11-slim sandbox image
scripts/
├── start.sh                   # Start all containers
├── stop.sh                    # Stop all containers
└── status.sh                  # Show container status
docs/
└── design_rationale.md        # Why each decision was made
```

## Development status

All phases complete.

| Phase | Description                      | Status |
|-------|----------------------------------|--------|
| 1     | Architecture design              | Done   |
| 2     | Supervisor graph & routing       | Done   |
| 3     | Corrective RAG sub-graph         | Done   |
| 4     | Code generation & Docker sandbox | Done   |
| 5     | Integration & memory             | Done   |
| 6     | HITL, streaming & observability  | Done   |
| 7     | Hardening & polish               | Done   |

See [docs/design_rationale.md](docs/design_rationale.md) for the reasoning behind each architectural decision.
