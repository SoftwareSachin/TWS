import sys
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Removed Redis dependency - using database-only status tracking
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.be_core.vanna_connector_manager import VannaConnectorManager
from app.crud.dataset_crud_v2 import dataset_v2
from app.db.connection_handler import DatabaseConnectionHandler
from app.utils.openai_utils import generate_text2sql_embedding


# Use OpenAI embeddings
class OpenAIHuggingFaceEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text):
        """Generate embedding for a single text query using OpenAI"""
        return generate_text2sql_embedding(text)

    def embed_documents(self, texts):
        """Generate embeddings for multiple documents using OpenAI"""
        return [generate_text2sql_embedding(text) for text in texts]


# Replace the mock module with OpenAI-powered embeddings
sys.modules["langchain_huggingface"] = type(
    "MockModule", (), {"HuggingFaceEmbeddings": OpenAIHuggingFaceEmbeddings}
)()


@celery.task(name="task.vanna_training_task", bind=True, max_retries=3)
def vanna_training_task(
    self,
    db_details,
    db_type,
    documentation=None,
    question_sql_pairs=None,
    database_name=None,
    user_id=None,
    dataset_id=None,
):
    logger.info(f"Task received - user_id: {user_id}, dataset_id: {dataset_id}")
    if not user_id or not dataset_id:
        logger.error("Missing user_id or dataset_id")
        return

    # Simplified: Using database-only status tracking
    # Frontend polls /dataset/{dataset_id}/ingestion_status API endpoint

    # Create synchronous database session for status updates
    engine = create_engine(str(settings.SYNC_DATABASE_URI))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def update_training_status(status: str, message: str = None):
        """
        Simplified function to update database status and log progress.

        Args:
            status: Training status (in_progress, completed, failed)
            message: Optional progress message for logging
        """
        # Map status to database ingestion_status
        status_mapping = {
            "in_progress": "processing",
            "completed": "success",
            "failed": "failed",
        }
        db_status = status_mapping.get(status, status)

        # Log the progress message for monitoring
        if message:
            logger.info(f"Training Progress: {message}")

        # Update database status
        try:
            db_session = SessionLocal()
            try:

                success = dataset_v2.update_ingestion_status_sync(
                    dataset_id=UUID(dataset_id), status=db_status, db_session=db_session
                )
                if success:
                    logger.info(
                        "Updated dataset %s ingestion_status to '%s'",
                        dataset_id,
                        db_status,
                    )
                else:
                    logger.warning(
                        "Failed to update dataset %s ingestion_status to '%s'",
                        dataset_id,
                        db_status,
                    )
            except Exception as e:
                db_session.rollback()
                logger.error(
                    "Database error updating ingestion_status to '%s': %s", db_status, e
                )
            finally:
                db_session.close()
        except Exception as e:
            logger.error("Failed to create database session for status update: %s", e)

    try:
        update_training_status("in_progress", "⚙️ Connecting to DB...")

        vn_instance = VannaConnectorManager(source_db_name=database_name)
        vn = vn_instance.vn

        DatabaseConnectionHandler.connect_database(vn, db_details, db_type)

        update_training_status("in_progress", "📄 Fetching schema info...")
        schema_query = DatabaseConnectionHandler.get_information_schema_query(db_type)
        df_information_schema = vn.run_sql(schema_query)

        update_training_status("in_progress", "📊 Generating training plan...")
        plan = vn.get_training_plan_generic(df_information_schema)

        update_training_status("in_progress", "🎯 Training on schema...")

        # Train on schema - uncommented for structured data
        try:
            vn.train(plan=plan)
            logger.info("Schema training completed successfully")
        except Exception as schema_error:
            logger.error(f"Schema training failed: {schema_error}")
            update_training_status(
                "failed", f"❌ Schema training failed: {str(schema_error)}"
            )
            raise

        if documentation:
            update_training_status("in_progress", "🧾 Training on documentation...")
            try:
                vn.train(documentation=documentation)
                logger.info("Documentation training completed successfully")
            except Exception as doc_error:
                logger.error(f"Documentation training failed: {doc_error}")
                update_training_status(
                    "failed", f"❌ Documentation training failed: {str(doc_error)}"
                )
                raise

        if question_sql_pairs:
            # Status will be updated in the loop below
            try:
                for i, pair in enumerate(question_sql_pairs, start=1):
                    vn.train(question=pair["question"], sql=pair["sql"])
                    update_training_status(
                        "in_progress",
                        f"💬 Trained Q&A pair {i}/{len(question_sql_pairs)}",
                    )
                logger.info(
                    f"Q&A training completed successfully for {len(question_sql_pairs)} pairs"
                )
            except Exception as qa_error:
                logger.error(f"Q&A training failed: {qa_error}")
                update_training_status(
                    "failed", f"❌ Q&A training failed: {str(qa_error)}"
                )
                raise

        update_training_status("completed", "✅ Model training completed!")

    except Exception as e:
        logger.error(f"Error in training task: {e}", exc_info=True)
        update_training_status("failed", f"❌ Training failed: {str(e)}")

        try:
            self.retry(exc=e, countdown=60)
        except Exception as retry_error:
            logger.error(f"Retry failed: {retry_error}")

    finally:
        # No cleanup needed - using database-only approach
        logger.info("Training task completed")
