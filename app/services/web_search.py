"""
Web search provider abstraction. This is the ONLY place Tavily's
client is instantiated or its response shape is depended on - if
another provider ever replaces Tavily, only this file needs to change,
same principle as the LLM abstraction in app/llm/models.py.

Failures are caught and returned as an empty result list rather than
raised, since a single failed web search shouldn't crash an entire
research run when private-document evidence might still be available -
callers (web_researcher_node) are expected to handle "zero results"
gracefully rather than assuming search always succeeds.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache

from tavily import TavilyClient

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    url: str
    content: str
    score: float


@lru_cache
def _get_client() -> TavilyClient:
    settings = get_settings()
    if not settings.TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not set. Add it to your .env file - see .env.example.")
    return TavilyClient(api_key=settings.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 5) -> list[WebSearchResult]:
    try:
        client = _get_client()
        response = client.search(query, max_results=max_results)
    except Exception as exc:
        logger.error(f"Web search failed for query '{query}': {exc}")
        return []

    return [
        WebSearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=float(r.get("score", 0.0)),
        )
        for r in response.get("results", [])
    ]