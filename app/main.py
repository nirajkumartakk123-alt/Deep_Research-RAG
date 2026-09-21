"""
Application entrypoint.
"""
import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, jobs, research
from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import configure_logging
from fastapi.middleware.cors import CORSMiddleware


settings = get_settings()
configure_logging(debug=settings.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic Research & Decision Intelligence Platform",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # portfolio/dev only - restrict in real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
        extra={"request_id": request_id, "latency_ms": duration_ms},
    )
    return response


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """FastAPI/Pydantic's own validation errors (bad request body,
    missing required field, etc.) - map to the same error envelope
    shape as our custom AppException, so API consumers get a
    consistent error format regardless of failure source."""
    return JSONResponse(
        status_code=422,
        content={"error_code": "validation_error", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort handler - never let a raw traceback leak to the
    client. Logs the full exception server-side, returns a generic
    500 to the caller."""
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error_code": "internal_error", "message": "An unexpected error occurred."},
    )


app.include_router(health.router)
app.include_router(documents.router)
app.include_router(jobs.router)
app.include_router(research.router)


@app.get("/")
async def root() -> dict:
    return {"message": f"{settings.APP_NAME} API is running. See /docs for the API schema."}