"""
Embedding service - the ONLY place SentenceTransformer is instantiated.

The model is loaded once (module-level cache via lru_cache) rather than
per-request, since loading weights from disk is expensive relative to
running inference on them.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a batch of texts. Returns L2-normalized vectors, since
    BAAI/bge-* models are trained/recommended for cosine similarity
    comparisons, which normalized vectors make numerically cleaner."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]