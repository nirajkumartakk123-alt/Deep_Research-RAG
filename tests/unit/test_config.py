"""
Configuration tests - explicitly required by the original spec's
testing checklist ("Test: configuration").
"""
from app.core.config import Settings, get_settings


def test_settings_has_required_defaults():
    settings = Settings()
    assert settings.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert settings.RERANKER_MODEL == "BAAI/bge-reranker-base"
    assert settings.TOP_K == 20
    assert settings.RERANK_TOP_K == 5
    assert settings.MAX_RETRIES == 2


def test_get_settings_is_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_chunk_overlap_smaller_than_chunk_size_by_default():
    settings = Settings()
    assert settings.CHUNK_OVERLAP < settings.MAX_CHUNK_SIZE