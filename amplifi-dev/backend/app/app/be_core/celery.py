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
        "app.api.celery_task",
        "app.api.workflow_exec_task",
        "app.api.azure_source_file_pull_task",
        "app.api.azure_blob_monitor_task",
        "app.api.azure_monitor_task",
        "app.api.pg_vector_source_data_pull_task",
        "app.api.mysql_source_data_pull_task",
        "app.api.prepare_split_ingestion_task",  # Split preparation task
        "app.api.image_ingestion_task",
        "app.api.audio_ingestion_task",  # New audio ingestion task
        "app.api.compress_pdf_task",
        "app.api.check_pdf_task",  # New PDF compression task
        "app.api.ingestion_task_v2",  # v2 ingestion tasks
        "app.api.prepare_split_ingestion_task_v2",  # v2 split preparation task
        "app.api.ingest_split_task_v2",  # v2 split ingestion task
        "app.api.image_ingestion_task_v2",  # v2 image ingestion task
        "app.api.audio_ingestion_task_v2",  # v2 audio ingestion task
        "app.api.video_ingestion_task",  # video ingestion task
        "app.api.vanna_training_task",  # training vanna task
        "app.api.kuzu_graph_extraction_task",  # kuzu graph extraction task
        "app.api.document_timeout_tasks",  # document timeout cleanup tasks,
        "app.api.groove_source_file_pull_task",
        "app.api.groove_monitor_task",  # Add this line
        "app.api.groove_blob_monitor_task",  # Add this line
        # "app.api.groove_data_fetch_task",
        "app.api.groove_source_file_pull_task",
        # "app.api.groove_data_fetch_task",
    ],  # route where tasks are defined
)

celery.conf.update(
    {
        "task_routes": {
            "tasks.ingest_files_task": {"queue": "ingestion_queue"},
            "tasks.delete_files_task": {"queue": "ingestion_queue"},
            "tasks.prepare_split_ingestion_task": {
                "queue": "ingestion_queue"
            },  # New split preparation task
            "tasks.image_ingestion_task": {"queue": "ingestion_queue"},
            "tasks.audio_ingestion_task": {
                "queue": "ingestion_queue"
            },  # New audio ingestion task route
            "tasks.video_ingestion_task": {"queue": "video_ingestion_queue"},
            "tasks.compress_pdf_task": {
                "queue": "file_compression_queue"
            },  # New PDF flattening/compression task
            "tasks.check_pdf_task": {"queue": "check_pdf_queue"},
            "tasks.aggregate_search_eval_task": {"queue": "search_eval_queue"},
            "tasks.dataset_search_eval_task": {"queue": "search_eval_queue"},
            "tasks.dataset_search_batch_task": {"queue": "search_eval_queue"},
            "tasks.process_new_azure_blob_task": {"queue": "file_pull_queue"},
            "tasks.process_updated_azure_blob_task": {"queue": "file_pull_queue"},
            "tasks.monitor_azure_sources_task": {"queue": "file_pull_queue"},
            "tasks.execute_workflow_task": {"queue": "workflow_queue"},
            "tasks.fetch_vectors_task": {"queue": "fetch_vector_queue"},
            "tasks.store_vectors_task": {"queue": "store_vector_queue"},
            "tasks.pull_files_from_azure_source_task": {"queue": "file_pull_queue"},
            "tasks.pull_tables_from_pg_db_task": {"queue": "pgvector_queue"},
            "tasks.pull_tables_from_mysql_db_task": {"queue": "mysql_queue"},
            # v2 tasks
            # "tasks.ingest_files_task_v2": {"queue": "document_ingestion_queue"},
            # "tasks.ingest_split_task_v2": {"queue": "document_ingestion_queue"},
            "tasks.ingest_files_task_v2": {"queue": "ingestion_queue"},
            "tasks.ingest_split_task_v2": {"queue": "ingestion_queue"},
            # "tasks.prepare_split_ingestion_task_v2": {
            #     "queue": "document_ingestion_queue"
            # },  # New split preparation task for v2
            "tasks.prepare_split_ingestion_task_v2": {
                "queue": "ingestion_queue"
            },  # New split preparation task for v2
            "tasks.image_ingestion_task_v2": {"queue": "ingestion_queue"},
            "tasks.audio_ingestion_task_v2": {
                "queue": "ingestion_queue"
            },  # New audio ingestion task for v2
            "task.vanna_training_task": {"queue": "training_vanna_queue"},
            "tasks.pull_files_from_groove_source_task": {"queue": "file_pull_queue"},
            "tasks.process_new_groove_ticket_task": {"queue": "file_pull_queue"},
            "tasks.process_updated_groove_ticket_task": {"queue": "file_pull_queue"},
            "tasks.monitor_groove_sources_task": {"queue": "file_pull_queue"},
            "tasks.kuzu_graph_extraction_task": {"queue": "ingestion_queue"},
            "tasks.kuzu_relationships_extraction_task": {"queue": "ingestion_queue"},
            "tasks.extract_text_from_dataset": {"queue": "ingestion_queue"},
            # Document timeout cleanup tasks
            "tasks.cleanup_stuck_documents": {"queue": "file_cleanup_queue"},
            "tasks.generate_stuck_documents_report": {"queue": "file_cleanup_queue"},
        },
        "beat_dburi": str(settings.SYNC_CELERY_BEAT_DATABASE_URI),
        # "task_acks_late": True,  # Ack after task succeeds
        "task_reject_on_worker_lost": True,  # Re-queue tasks on worker failure
        "worker_prefetch_multiplier": 1,  # Prefetch 1 task at a time
        "beat_schedule": {
            "monitor-azure-sources": {
                "task": "tasks.monitor_azure_sources_task",
                "schedule": 60.0,  # Run every 1 minute
            },
            "tasks.cleanup-stuck-documents": {
                "task": "tasks.cleanup_stuck_documents",
                "schedule": 600.0,  # Run every 10 minutes to clean stuck documents
            },
            "tasks.generate-stuck-documents-report": {
                "task": "tasks.generate_stuck_documents_report",
                "schedule": 600.0,  # Run every 10 minutes for generating reports
            },
            "monitor-groove-sources": {
                "task": "tasks.monitor_groove_sources_task",
                "schedule": 60.0,  # Run every 1 minute
            },
            # Periodic task schedule
            "timezone": "UTC",
        },
    }
)
celery.autodiscover_tasks()
