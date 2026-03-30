import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL_HEAVY = os.getenv("OPENAI_MODEL_HEAVY", "gpt-4o")

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "checkpoints.db")

MAX_PLAN_ITERATIONS = 10
MAX_RAG_REWRITES = 2
RAG_RELEVANCE_THRESHOLD = 0.7
MAX_CODE_RETRIES = 3
