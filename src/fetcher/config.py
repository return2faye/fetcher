import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL_HEAVY = os.getenv("OPENAI_MODEL_HEAVY", "gpt-4o")

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.db")

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "fetcher_docs")

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Limits
MAX_PLAN_ITERATIONS = 10
MAX_RAG_REWRITES = 2
RAG_RELEVANCE_THRESHOLD = 0.7
MAX_CODE_RETRIES = 3

# LangSmith tracing (opt-in via env vars)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "fetcher")
LANGSMITH_TRACING_ENABLED = bool(LANGSMITH_API_KEY)

# When LANGSMITH_API_KEY is set, configure the required env vars for LangSmith
if LANGSMITH_TRACING_ENABLED:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
