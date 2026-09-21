"""
Lightweight observability: wraps each agent node with timing +
structured logging, and accumulates a run_metadata trail in state.
No external tracing service required (LangSmith is optional - set
LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env to enable it;
the app works identically without it).
"""
import functools
import logging
import time
from typing import Callable

logger = logging.getLogger("app.observability")


def track_node(node_name: str):
    """Decorator for LangGraph node functions. Records latency and
    logs entry/exit, and appends a trace entry to state['run_metadata']
    so the full node-by-node timeline is visible in the final result."""

    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(state: dict) -> dict:
            start = time.perf_counter()
            logger.info(f"[{node_name}] started")
            try:
                result = await fn(state)
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.info(f"[{node_name}] completed in {latency_ms}ms")

                trace_entry = {"node": node_name, "latency_ms": latency_ms, "status": "success"}
                existing_metadata = state.get("run_metadata", {"trace": []})
                trace = existing_metadata.get("trace", []) + [trace_entry]
                result["run_metadata"] = {**existing_metadata, "trace": trace}
                return result
            except Exception as exc:
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.error(f"[{node_name}] failed after {latency_ms}ms: {exc}")
                raise

        return wrapper

    return decorator


def track_node_sync(node_name: str):
    """Sync variant for citation_builder_node (no I/O, no async)."""

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            start = time.perf_counter()
            result = fn(state)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(f"[{node_name}] completed in {latency_ms}ms")

            trace_entry = {"node": node_name, "latency_ms": latency_ms, "status": "success"}
            existing_metadata = state.get("run_metadata", {"trace": []})
            trace = existing_metadata.get("trace", []) + [trace_entry]
            result["run_metadata"] = {**existing_metadata, "trace": trace}
            return result

        return wrapper

    return decorator