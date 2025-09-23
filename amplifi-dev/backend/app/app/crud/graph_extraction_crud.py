from datetime import UTC, datetime

from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.graph_extraction_model import (
    ExtractionStatusEnum,
    GraphExtractionStatus,
)


def get_extraction_counts(dataset_id):
    """Get counts of extraction statuses"""
    with SyncSessionLocal() as db:
        total_documents = (
            db.query(GraphExtractionStatus)
            .filter(GraphExtractionStatus.dataset_id == dataset_id)
            .count()
        )

        completed_count = (
            db.query(GraphExtractionStatus)
            .filter(
                GraphExtractionStatus.dataset_id == dataset_id,
                GraphExtractionStatus.status == ExtractionStatusEnum.COMPLETED,
            )
            .count()
        )

        failed_count = (
            db.query(GraphExtractionStatus)
            .filter(
                GraphExtractionStatus.dataset_id == dataset_id,
                GraphExtractionStatus.status == ExtractionStatusEnum.FAILED,
            )
            .count()
        )

        pending_count = (
            db.query(GraphExtractionStatus)
            .filter(
                GraphExtractionStatus.dataset_id == dataset_id,
                GraphExtractionStatus.status == ExtractionStatusEnum.PENDING,
            )
            .count()
        )

    return {
        "total": total_documents,
        "completed": completed_count,
        "failed": failed_count,
        "pending": pending_count,
    }


def process_pending_extractions(dataset_id, client):
    """Process all pending extractions"""
    with SyncSessionLocal() as db:
        pending_extractions = (
            db.query(GraphExtractionStatus)
            .filter(
                GraphExtractionStatus.dataset_id == dataset_id,
                GraphExtractionStatus.status == ExtractionStatusEnum.PENDING,
            )
            .all()
        )

        for extraction in pending_extractions:
            try:
                doc_response = client.documents.retrieve(id=extraction.document_id)
                extraction_status = doc_response.results.extraction_status.value

                if extraction_status == "success":
                    extraction.status = ExtractionStatusEnum.COMPLETED
                    extraction.extraction_completed_at = datetime.now(UTC)
                    logger.info(
                        f"Document {extraction.document_id} extraction completed"
                    )

                    # Deduplicate entities
                    try:
                        client.documents.deduplicate(id=extraction.document_id)
                        logger.info(
                            f"Deduplication completed for document {extraction.document_id}"
                        )
                    except Exception as dedupe_error:
                        logger.warning(
                            f"Deduplication failed for {extraction.document_id}: {dedupe_error}"
                        )

                elif extraction_status == "failed":
                    extraction.status = ExtractionStatusEnum.FAILED
                    extraction.extraction_completed_at = datetime.now(UTC)
                    extraction.error_message = "Extraction failed according to R2R"
                    logger.error(f"Document {extraction.document_id} extraction failed")
                    db.commit()
                    return False

            except Exception as e:
                logger.error(f"Error checking document {extraction.document_id}: {e}")

        db.commit()
        return True
