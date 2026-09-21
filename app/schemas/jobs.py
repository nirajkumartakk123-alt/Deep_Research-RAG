from pydantic import BaseModel


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    result: dict | None = None
    error: str | None = None