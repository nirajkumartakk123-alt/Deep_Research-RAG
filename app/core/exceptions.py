"""
Typed application exceptions.

Services/agents raise these instead of generic Exception, so the
FastAPI exception handlers in main.py can map them to correct,
consistent HTTP responses without every route needing its own
try/except block.
"""


class AppException(Exception):
    """Base class for all application-specific exceptions."""
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NotFoundError(AppException):
    status_code = 404
    error_code = "not_found"


class ValidationAppError(AppException):
    status_code = 422
    error_code = "validation_error"


class ServiceUnavailableError(AppException):
    """Raised when a downstream dependency (DB, Redis, LLM, search) is unreachable."""
    status_code = 503
    error_code = "service_unavailable"


class UnsupportedFileTypeError(ValidationAppError):
    error_code = "unsupported_file_type"


class EmptyDocumentError(ValidationAppError):
    error_code = "empty_document"


class DocumentExtractionError(AppException):
    status_code = 422
    error_code = "extraction_failed"