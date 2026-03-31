"""Long-term memory: store and recall past results via Qdrant.

Uses a separate collection ('fetcher_memory') from the document store
to avoid mixing user-ingested docs with system-generated memories.
"""

from fetcher.config import QDRANT_HOST, QDRANT_PORT, EMBEDDING_DIM
from fetcher.utils.embeddings import embed_texts, embed_query

MEMORY_COLLECTION = "fetcher_memory"

_initialized = False


def _ensure_memory_collection() -> None:
    """Create the memory collection if it doesn't exist."""
    global _initialized
    if _initialized:
        return

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = [c.name for c in client.get_collections().collections]
        if MEMORY_COLLECTION not in collections:
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        _initialized = True
    except Exception:
        # If Qdrant is not running, memory ops silently degrade
        pass


def store_result(task: str, result: str, result_type: str = "research") -> None:
    """Store a task result in long-term memory.

    Silently fails if Qdrant is unavailable (memory is best-effort).
    """
    try:
        _ensure_memory_collection()

        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        import uuid

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        text = f"[{result_type}] {task}: {result}"
        vector = embed_texts([text])[0]

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "task": task,
                "result": result,
                "type": result_type,
            },
        )

        client.upsert(collection_name=MEMORY_COLLECTION, points=[point])
    except Exception:
        pass  # Memory is best-effort


def recall_context(query: str, top_k: int = 3, score_threshold: float = 0.5) -> str:
    """Recall relevant past results from long-term memory.

    Returns a concatenated string of relevant memories, or empty string.
    """
    try:
        _ensure_memory_collection()

        from qdrant_client import QdrantClient

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        query_vector = embed_query(query)

        results = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        if not results.points:
            return ""

        memories = [hit.payload.get("text", "") for hit in results.points]
        return "\n\n".join(memories)
    except Exception:
        return ""
