"""Local embedding model using sentence-transformers."""

from sentence_transformers import SentenceTransformer

from fetcher.config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returns list of float vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
