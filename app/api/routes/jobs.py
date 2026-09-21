"""
Job status endpoint for async-submitted work (currently: document
processing). Reads Celery's AsyncResult directly - no separate job
table needed, since Celery's Redis result backend already persists
task state/result for result_expires seconds (see celery_app.py).
"""
from fastapi import APIRouter

from app.schemas.jobs import JobStatusResponse
from app.workers.celery_app import celery_app

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    result = celery_app.AsyncResult(job_id)

    response = JobStatusResponse(job_id=job_id, status=result.status)

    if result.status == "SUCCESS":
        response.result = result.result
    elif result.status == "FAILURE":
        response.error = str(result.result)

    return response