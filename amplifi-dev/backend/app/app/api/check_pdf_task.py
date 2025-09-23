import os
from typing import Any, Dict
from uuid import UUID

import fitz
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.file_model import File
from app.schemas.file_schema import FileStatusEnum


@celery.task(name="tasks.check_pdf_task", bind=True, max_retries=3, acks_late=True)
def check_pdf_task(
    self,
    file_id: UUID,
    workspace_id: UUID,
    user_id: UUID = None,
    pixel_threshold: int = 2000,
    pages_to_check: int = 3,
) -> Dict[str, Any]:
    """
    Fast PDF processing task that checks if compression is needed and either:
    1. Sets status to Uploaded if no compression needed
    2. Queues actual compression task if compression is needed
    """
    logger.info(f"Starting PDF processing check for file_id: {file_id}")

    db_session: Session = SyncSessionLocal()

    try:
        # Get file record from database
        file_record = (
            db_session.query(File)
            .filter(
                File.id == file_id,
                File.workspace_id == workspace_id,
                File.deleted_at.is_(None),
            )
            .first()
        )

        if not file_record:
            logger.error(f"File not found: {file_id}")
            return {"success": False, "error": "File not found"}

        original_path = file_record.file_path
        if not os.path.exists(original_path):
            logger.error(f"File path does not exist: {original_path}")
            file_record.status = FileStatusEnum.Failed
            db_session.commit()
            return {"success": False, "error": "File path does not exist"}

        # Quick check to see if compression is needed
        needs_compression = _quick_compression_check(
            original_path, pixel_threshold, pages_to_check
        )

        if needs_compression:
            logger.info(f"File {file_id} needs compression, queuing compression task")

            # Queue the actual compression task
            task_kwargs = {
                "file_id": file_id,
                "workspace_id": workspace_id,
                # "pixel_threshold": pixel_threshold,
                # "pages_to_check": pages_to_check,
            }

            if user_id:
                task_kwargs["user_id"] = user_id

            celery.signature(
                "tasks.compress_pdf_task",
                kwargs=task_kwargs,
            ).apply_async()

            logger.info(f"Compression task queued for file: {file_id}")
            return {
                "success": True,
                "needs_compression": True,
                "message": "Compression task queued",
                "file_id": str(file_id),
            }
        else:
            # No compression needed, set status to Uploaded immediately
            file_record.status = FileStatusEnum.Uploaded
            db_session.commit()
            logger.info(
                f"No compression needed for file: {file_id}, status set to Uploaded"
            )

            return {
                "success": True,
                "needs_compression": False,
                "message": "No compression needed",
                "file_id": str(file_id),
            }

    except Exception as e:
        logger.error(
            f"PDF processing check failed for file_id {file_id}: {str(e)}",
            exc_info=True,
        )

        # Update file status to failed
        try:
            file_record = db_session.query(File).filter(File.id == file_id).first()
            if file_record:
                file_record.status = FileStatusEnum.Failed
                db_session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update file status: {str(db_error)}")

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying PDF processing check for file_id {file_id}, attempt {self.request.retries + 1}"
            )
            raise self.retry(exc=e, countdown=30)  # Shorter retry interval for checks
        else:
            logger.error(f"Max retries reached for PDF processing check: {file_id}")
            return {"success": False, "error": str(e), "file_id": str(file_id)}

    finally:
        db_session.close()


def _quick_compression_check(
    pdf_path: str,
    pixel_threshold: int = 2000,
    pages_to_check: int = 3,
) -> bool:
    """
    Quick check to determine if PDF needs compression.
    Only opens the PDF and checks resolution without doing any processing.
    """
    logger.debug(f"Quick compression check for: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        logger.debug(
            f"Checking {min(pages_to_check, len(doc))} pages for high resolution"
        )

        for i in range(min(pages_to_check, len(doc))):
            page = doc.load_page(i)
            # Get page dimensions without creating full pixmap (faster)
            rect = page.rect
            # Estimate pixel dimensions at 100 DPI
            width_pixels = int(rect.width * 100 / 72)  # 72 DPI is default
            height_pixels = int(rect.height * 100 / 72)

            if width_pixels > pixel_threshold or height_pixels > pixel_threshold:
                logger.info(
                    f"High resolution detected: ~{width_pixels}x{height_pixels} px"
                )
                doc.close()
                return True

        doc.close()
        logger.debug("No high-resolution pages found")
        return False

    except Exception as e:
        logger.error(f"Quick compression check failed: {str(e)}")
        doc.close() if "doc" in locals() else None
        # If check fails, assume compression is not needed to avoid blocking
        return False
