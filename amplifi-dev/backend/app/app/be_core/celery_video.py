# Celery is good for data-intensive application or some long-running tasks in other simple cases use Fastapi background tasks
# Reference https://towardsdatascience.com/deploying-ml-models-in-production-with-fastapi-and-celery-7063e539a5db
from celery import Celery

from app.be_core.config import settings

# Get the PostgreSQL URI from settings
postgres_uri = str(settings.SYNC_CELERY_DATABASE_URI)

# Parse the URI components properly
if "+" in postgres_uri:
    # Handle postgresql+psycopg2:// format
    dialect = postgres_uri[postgres_uri.find("postgresql") : postgres_uri.find("://")]
    rest_of_uri = postgres_uri[postgres_uri.find("://") :]
else:
    # Handle postgresql:// format
    dialect = "postgresql"
    rest_of_uri = postgres_uri[postgres_uri.find("://") :]

# Create the backend URI with the 'db+' prefix
backend_uri = f"db+{dialect}{rest_of_uri}"


celery = Celery(
    "async_task",
    broker=f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASSWORD}@{settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}",
    # broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
    # backend=str(settings.SYNC_CELERY_DATABASE_URI),
    backend=backend_uri,
    include=[  # route where tasks are defined
        "app.api.video_ingestion_task",  # video ingestion task
    ],  # route where tasks are defined
)

celery.conf.update(
    {
        "task_routes": {
            "tasks.video_ingestion_task": {"queue": "video_ingestion_queue"},
        },
        "beat_dburi": str(settings.SYNC_CELERY_BEAT_DATABASE_URI),
        # "task_acks_late": True,  # Ack after task succeeds
        "task_reject_on_worker_lost": True,  # Re-queue tasks on worker failure
        "worker_prefetch_multiplier": 1,  # Prefetch 1 task at a time
    }
)
celery.autodiscover_tasks()
