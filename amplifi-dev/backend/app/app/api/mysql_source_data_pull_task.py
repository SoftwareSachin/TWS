import os
from typing import Any, Dict, List
from uuid import UUID

import pymysql

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.db import session
from app.models.file_model import File
from app.models.pull_status_model import PullStatusEnum, SourcePullStatus
from app.schemas.file_schema import FileStatusEnum


@celery.task(name="tasks.pull_tables_from_mysql_db_task", bind=True, max_retries=3)
def pull_tables_from_mysql_db_task(
    self,
    workspace_id: UUID,
    user_id: UUID,
    source_id: UUID,
    username: str,
    password: str,
    host: str,
    port: int,
    database_name: str,
) -> List[Dict[str, Any]]:
    logger.info(f"Starting pull_tables_from_mysql_db_task for Source ID: {source_id}")

    db_session: session = session.SyncSessionLocal()
    pulled_tables: List[Dict[str, Any]] = []
    DEFAULT_UPLOAD_FOLDER = settings.DEFAULT_UPLOAD_FOLDER

    try:
        logger.info(f"Connecting to MySQL database for source {source_id}")

        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.STARTED})
        db_session.commit()

        # Connect to MySQL
        conn = pymysql.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            database=database_name,
            ssl={"ssl": {}},  # Uses default SSL if enabled; remove if not required
        )
        cursor = conn.cursor()

        cursor.execute("SHOW TABLES;")
        tables = [row[0] for row in cursor.fetchall()]

        for table_name in tables:
            logger.info(f"Fetching data from table: {table_name}")

            try:
                # Get actual row count from the table
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")  # nosec
                    actual_row_count = cursor.fetchone()[0]
                except Exception as count_error:
                    logger.warning(
                        f"Could not get row count for {table_name}: {count_error}"
                    )
                    # Fallback: try to get approximate row count from information_schema
                    try:
                        cursor.execute(
                            """
                            SELECT table_rows
                            FROM information_schema.tables
                            WHERE table_schema = %s AND table_name = %s
                            """,
                            (database_name, table_name),
                        )
                        result = cursor.fetchone()
                        actual_row_count = (
                            result[0] if result and result[0] is not None else 0
                        )
                        logger.info(
                            f"Using approximate row count for {table_name}: {actual_row_count}"
                        )
                    except Exception:
                        logger.warning(
                            f"Could not get approximate row count for {table_name}, using 0"
                        )
                        actual_row_count = 0

                # Get column information (schema)
                cursor.execute(
                    """
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (database_name, table_name),  # nosec
                )

                schema_rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]

                # Get actual column count from the table structure
                actual_column_count = len(schema_rows)

                unique_filename = f"{table_name}.csv"
                file_path = os.path.join(DEFAULT_UPLOAD_FOLDER, unique_filename)

                with open(file_path, "w") as f:
                    f.write(",".join(columns) + "\n")
                    for row in schema_rows:
                        f.write(",".join(map(str, row)) + "\n")

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

                pulled_tables.append(
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

            except Exception as table_error:
                logger.error(f"Error processing table {table_name}: {table_error}")
                # Continue with next table instead of failing entire task
                continue

        cursor.close()
        conn.close()

        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.SUCCESS})
        db_session.commit()

        logger.info(f"Successfully pulled {len(pulled_tables)} tables from mysql_db")

    except Exception as e:
        logger.error(f"Error pulling tables from mysql_db: {e}")

        db_session.query(SourcePullStatus).filter(
            SourcePullStatus.source_id == source_id
        ).update({"pull_status": PullStatusEnum.FAILED})
        db_session.commit()

        if self.request.retries >= self.max_retries:
            logger.error("Max retries reached for mysql_db task. Status set to FAILED.")

        raise self.retry(exc=e, countdown=0)

    finally:
        db_session.close()

    return pulled_tables
