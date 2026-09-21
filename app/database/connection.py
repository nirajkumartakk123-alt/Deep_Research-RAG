"""
Async SQLAlchemy engine/session setup.

This is the ONLY place that constructs the engine. Everything else
(repositories, health checks) imports get_db / engine from here.

NullPool is used deliberately, not as an oversight: Starlette's
TestClient (and, more generally, any client that spins up a new event
loop per call) is incompatible with SQLAlchemy's default connection
pooling on Windows - asyncpg connections are bound to the event loop
that created them, and Windows' ProactorEventLoop raises a low-level
AttributeError when a pooled connection from a closed loop is reused
by a new one. NullPool sidesteps this entirely by never reusing a
connection across checkouts: every request opens a fresh connection
and closes it when done.

Trade-off, stated honestly: this adds per-request connection setup
latency instead of reusing warm connections. Acceptable at this
project's scale (single dev instance, MVP traffic). If this were
deployed at real production scale with sustained concurrent load,
the better fix would be running the actual app under uvicorn (a
single long-lived event loop, which pools fine) and only disabling
pooling for the test suite specifically - not for the app as a whole.
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import get_settings

settings = get_settings()

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    poolclass=NullPool,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a request-scoped DB session."""
    async with AsyncSessionLocal() as session:
        yield session