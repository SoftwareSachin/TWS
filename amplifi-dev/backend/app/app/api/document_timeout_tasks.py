"""
Celery tasks for handling document timeouts and stuck document cleanup.

These tasks run automatically via Celery Beat to clean up documents
that have been stuck in processing for too long.
"""

from app.be_core.celery import celery
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.utils.ingestion_status_propagation import (
    get_stuck_documents_report,
    mark_stuck_documents_as_failed,
)


@celery.task(bind=True, name="tasks.cleanup_stuck_documents", acks_late=True)
def cleanup_stuck_documents_task(self):
    """
    Automatic periodic task to find and mark stuck documents as failed.

    This task runs every hour via Celery Beat to clean up documents
    that have been stuck in processing for too long (configurable via
    DOCUMENT_PROCESSING_TIMEOUT_SECONDS in config.py).

    Returns:
        dict: Results of the cleanup operation
    """
    task_id = self.request.id
    logger.info(f"🧹 [AUTOMATIC] Starting stuck document cleanup task {task_id}")

    try:
        with SyncSessionLocal() as db:
            result = mark_stuck_documents_as_failed(db)

        if result["success"]:
            if result["marked_failed"] > 0:
                logger.warning(
                    f"⚠️ [CLEANUP] Task {task_id} completed: "
                    f"marked {result['marked_failed']} documents as failed "
                    f"out of {result['total_checked']} checked"
                )
                # Log details for monitoring
                for doc in result.get("documents", []):
                    logger.warning(
                        f"📄 Failed document: {doc['document_id']} "
                        f"(stuck {doc['hours_stuck']}h in {doc['original_status']})"
                    )
            else:
                logger.info(
                    f"✅ [CLEANUP] Task {task_id} completed: no stuck documents found"
                )
        else:
            logger.error(f"❌ [CLEANUP] Task {task_id} failed: {result['message']}")

        return result

    except Exception as e:
        error_msg = (
            f"Error in automatic stuck document cleanup task {task_id}: {str(e)}"
        )
        logger.error(error_msg, exc_info=True)
        return {"success": False, "message": error_msg, "task_id": task_id}


@celery.task(bind=True, name="tasks.generate_stuck_documents_report", acks_late=True)
def generate_stuck_documents_report_task(self):
    """
    Automatic periodic task to generate a report of stuck documents.

    This task runs every 30 minutes via Celery Beat to monitor and log
    documents that are approaching the timeout threshold. This provides
    early warning before documents are automatically marked as failed.

    Returns:
        dict: Report of stuck and warning documents
    """
    task_id = self.request.id
    logger.info(f"📊 [MONITORING] Generating stuck documents report - task {task_id}")

    try:
        with SyncSessionLocal() as db:
            report = get_stuck_documents_report(db)

        stuck_count = report["stuck_documents"]["count"]
        warning_count = report["warning_documents"]["count"]

        if stuck_count > 0:
            logger.warning(
                f"⚠️ [MONITOR] Task {task_id}: Found {stuck_count} stuck documents "
                f"and {warning_count} documents approaching timeout"
            )
            # Log details for immediate attention
            for doc in report["stuck_documents"]["documents"]:
                logger.error(
                    f"🚨 STUCK DOCUMENT: {doc['document_id']} "
                    f"(stuck {doc['hours_stuck']}h in {doc['status']}) - "
                    f"will be marked as failed in next cleanup cycle"
                )

        if warning_count > 0:
            logger.info(
                f"⚠️ [MONITOR] Task {task_id}: {warning_count} documents approaching timeout threshold"
            )

        if stuck_count == 0 and warning_count == 0:
            logger.info(
                f"✅ [MONITOR] Task {task_id}: All documents processing normally"
            )

        return report

    except Exception as e:
        error_msg = (
            f"Error generating stuck documents report - task {task_id}: {str(e)}"
        )
        logger.error(error_msg, exc_info=True)
        return {"success": False, "message": error_msg, "task_id": task_id}
