"""
Structured JSON logging configuration.

Every log line is a JSON object so it can later be shipped to any
log aggregator without reformatting. Keep this dependency-free
(stdlib `logging` only) so it works identically in and out of Docker.
"""
import logging
import sys
import json
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Allow callers to attach structured context via `extra={...}`
        for key in ("request_id", "research_id", "node", "latency_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(debug: bool = True) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Avoid duplicate handlers if configure_logging is called more than once
    # (e.g. under --reload).
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)