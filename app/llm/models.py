"""
LLM provider abstraction.

This is the ONLY place a chat model client is instantiated. Every
agent node (Phase 6+) imports get_llm() rather than constructing its
own ChatGroq/ChatMistralAI/etc. instance - this is what makes swapping
providers later (as we just did, Mistral -> Groq) a one-file change
instead of a hunt-and-replace across the codebase.

Model is cached via lru_cache since client construction has some
overhead and there's no reason to rebuild it on every call.
"""
from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import get_settings


@lru_cache
def get_llm(temperature: float = 0.0) -> ChatGroq:
    """temperature defaults to 0.0 (deterministic) since most of this
    project's LLM calls are structured-output classification/extraction
    tasks (query analysis, planning, verification) where consistency
    matters more than creative variation. The synthesizer node (Phase 6,
    later) explicitly overrides this with a higher temperature for
    more natural report-writing prose."""
    settings = get_settings()
    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file - see .env.example."
        )
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=temperature,
    )