"""Qdrant vector DB client: collection management and document operations."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)

from fetcher.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_DIM
from fetcher.utils.embeddings import embed_texts, embed_query

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Lazy-load Qdrant client (singleton)."""
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection(collection_name: str = QDRANT_COLLECTION) -> None:
    """Create the collection if it doesn't exist."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def ingest_documents(
    texts: list[str],
    metadatas: list[dict] | None = None,
    collection_name: str = QDRANT_COLLECTION,
) -> int:
    """Chunk-free ingestion: embed texts and upsert into Qdrant.

    Returns the number of points upserted.
    """
    client = get_qdrant_client()
    ensure_collection(collection_name)

    vectors = embed_texts(texts)
    metadatas = metadatas or [{} for _ in texts]

    points = [
        PointStruct(
            id=idx,
            vector=vec,
            payload={"text": text, **meta},
        )
        for idx, (text, vec, meta) in enumerate(zip(texts, vectors, metadatas))
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def search_documents(
    query: str,
    top_k: int = 5,
    collection_name: str = QDRANT_COLLECTION,
    score_threshold: float | None = None,
) -> list[dict]:
    """Search Qdrant for documents similar to query.

    Returns list of {"text": ..., "score": ..., "metadata": ...} dicts.
    """
    client = get_qdrant_client()
    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": hit.score,
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
        }
        for hit in results.points
    ]
