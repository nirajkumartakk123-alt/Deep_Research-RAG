"""
Celery application instance.

Windows note: the default 'prefork' worker pool relies on os.fork(),
which doesn't exist on Windows. Use --pool=solo when starting the
worker on Windows (see run command below). solo runs tasks one at a
time in the main process - no true parallelism, but reliable, and
adequate for this project's scale. A production Linux deployment
would drop --pool=solo and get real prefork concurrency for free.
"""
from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "deepresearch_rag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
)