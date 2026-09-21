"""
Liveness/readiness endpoint.

Actually pings Postgres and Redis rather than returning a static
{"status": "ok"} — a health check that doesn't check anything is
worse than no health check.
"""
import logging

import redis.asyncio as redis
from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import get_settings
from app.database.connection import engine

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()

    db_status = "ok"
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — intentionally broad for a health probe
        logger.error("Database health check failed", exc_info=exc)
        db_status = "unreachable"

    redis_status = "ok"
    client = redis.from_url(settings.REDIS_URL)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        logger.error("Redis health check failed", exc_info=exc)
        redis_status = "unreachable"
    finally:
        await client.aclose()

    overall = "ok" if db_status == "ok" and redis_status == "ok" else "degraded"

    return {
        "status": overall,
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "dependencies": {
            "database": db_status,
            "redis": redis_status,
        },
    }