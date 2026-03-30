# Fetcher

A multi-agent system that takes complex technical questions, autonomously researches the web, generates code, executes it in a sandboxed environment, and verifies the results.

Built with [LangGraph](https://github.com/langchain-ai/langgraph).

## How it works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│             Supervisor Graph                 │
│                                              │
│  Planner ──▶ Router ──▶ Synthesizer ──▶ HITL │
│                 │                             │
│        ┌───────┴────────┐                    │
│        ▼                ▼                    │
│   RAG Sub-graph   Code Sub-graph             │
│   (CRAG flow)     (generate-execute-verify)  │
└─────────────────────────────────────────────┘
```

1. **Supervisor** decomposes your query into typed sub-tasks (research, code, hybrid)
2. **Corrective RAG** retrieves from a vector DB, grades relevance, rewrites queries or falls back to web search
3. **Code Engine** generates code, runs it in a Docker sandbox, and self-corrects on failure *(in progress)*
4. **Synthesizer** merges all results into a final answer
5. **Human Review** pauses for your approval before delivering the output

## Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/en/latest/)
- [Docker](https://docs.docker.com/get-docker/)
- An OpenAI API key

## Setup

```bash
# 1. Create conda environment
conda create -n fetcher python=3.11 -y
conda activate fetcher

# 2. Install dependencies
pip install langgraph langchain-core langchain-openai langchain-community \
    sentence-transformers duckduckgo-search qdrant-client docker \
    python-dotenv langgraph-checkpoint-sqlite

# 3. Install dev dependencies
pip install pytest pytest-asyncio

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Start containers (Qdrant)
./scripts/start.sh
```

## Usage

```bash
# Start containers
./scripts/start.sh

# Check container status
./scripts/status.sh

# Stop containers
./scripts/stop.sh
```

## Running tests

```bash
conda activate fetcher
PYTHONPATH=src pytest tests/ -v
```

## Project structure

```
src/fetcher/
├── config.py              # Environment variables and constants
├── state.py               # LangGraph state schemas
├── graphs/
│   ├── supervisor.py      # Top-level orchestrator graph
│   └── rag.py             # Corrective RAG sub-graph
├── nodes/
│   ├── supervisor.py      # Planner, router, synthesizer nodes
│   └── rag.py             # Retrieve, grade, rewrite, search, generate nodes
└── utils/
    ├── embeddings.py      # Local embedding model (all-MiniLM-L6-v2)
    └── qdrant_client.py   # Vector DB operations
```

## Development status

| Phase | Description                      | Status      |
|-------|----------------------------------|-------------|
| 1     | Architecture design              | Done        |
| 2     | Supervisor graph & routing       | Done        |
| 3     | Corrective RAG sub-graph         | Done        |
| 4     | Code generation & Docker sandbox | Next        |
| 5     | Integration & memory             | Pending     |
| 6     | HITL, streaming & observability  | Pending     |
| 7     | Hardening & polish               | Pending     |

See [docs/design_rationale.md](docs/design_rationale.md) for the reasoning behind each architectural decision.
