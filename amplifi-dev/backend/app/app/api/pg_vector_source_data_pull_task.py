import os
from typing import Any, Dict, List
from uuid import UUID

import psycopg2
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.file_model import File
from app.models.pull_status_model import PullStatusEnum, SourcePullStatus
from app.schemas.file_schema import FileStatusEnum


@celery.task(name="tasks.pull_tables_from_pg_db_task", bind=True, max_retries=3)
def pull_tables_from_pg_db_task(
    self,
    workspace_id: UUID,
    user_id: UUID,
    source_id: UUID,
    username: str,
    password: str,
    host: str,
    port: str,
    database_name: str,
) -> List[Dict[str, Any]]:
    logger.info(f"Starting pull_tables_from_pg_db_task for Source ID: {source_id}")

    db_session: Session = SyncSessionLocal()
    pulled_vectors: List[Dict[str, Any]] = []
    DEFAULT_UPLOAD_FOLDER = settings.DEFAULT_UPLOAD_FOLDER

    try:
        logger.info(f"Connecting to PGVector database for source {source_id}")

        # Mark pull as STARTED
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.STARTED})
        db_session.commit()

        # Connect to PostgreSQL with pg_db credentials
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=database_name,
            user=username,
            password=password,
            sslmode="require",
        )
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            logger.info(f"Fetching data from table: {table_name}")

            # Get actual row count from the table
            cursor.execute(f"SELECT COUNT(*) FROM public.{table_name}")  # nosec
            actual_row_count = cursor.fetchone()[0]

            # Get column information (schema)
            cursor.execute(
                """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,),
            )

            schema_rows = cursor.fetchall()
            for column in schema_rows:
                column_name, data_type, is_nullable, char_max_len = column
                logger.info(
                    f"Column: {column_name}, "
                    f"Type: {data_type}, "
                    f"Nullable: {is_nullable}, "
                    f"Max Length: {char_max_len}"
                )
            columns = [desc[0] for desc in cursor.description]

            # Get actual column count from the table structure
            actual_column_count = len(schema_rows)

            unique_filename = f"{table_name}.csv"
            file_path = os.path.join(DEFAULT_UPLOAD_FOLDER, unique_filename)

            # Save table schema to CSV
            with open(file_path, "w") as f:
                f.write(",".join(columns) + "\n")
                for row in schema_rows:
                    f.write(",".join(map(str, row)) + "\n")

            # Insert file record into DB
            file = File(
                filename=unique_filename,
                mimetype="text/csv",
                size=os.path.getsize(file_path),
                file_path=file_path,
                status=FileStatusEnum.Uploaded,
                source_id=source_id,
                workspace_id=workspace_id,
                rows=actual_row_count,
                columns=actual_column_count,
            )
            db_session.add(file)
            db_session.commit()
            db_session.refresh(file)

            pulled_vectors.append(
                {
                    "filename": file.filename,
                    "size": file.size,
                    "status": file.status,
                    "id": str(file.id),
                    "table_name": table_name,
                    "row_count": file.rows,
                    "column_count": file.columns,
                }
            )

            logger.info(
                f"Stored schema from {table_name} into {unique_filename} "
                f"(Actual Table Rows: {actual_row_count}, Columns: {actual_column_count})"
            )

        cursor.close()
        conn.close()

        # Mark pull as SUCCESS
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.SUCCESS})
        db_session.commit()

        logger.info(f"Successfully pulled {len(pulled_vectors)} tables from pg_db")

    except Exception as e:
        logger.error(f"Error pulling tables from pg_db: {e}")

        # Mark pull as FAILED
        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.FAILED})
        db_session.commit()

        if self.request.retries >= self.max_retries:
            logger.error("Max retries reached for pg_db task. Status set to FAILED.")
        raise self.retry(exc=e, countdown=0)
    finally:
        db_session.close()

    return pulled_vectors
