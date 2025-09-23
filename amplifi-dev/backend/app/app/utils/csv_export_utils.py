import os
from datetime import datetime
from typing import Optional
from uuid import UUID

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.dataset_crud_v2 import dataset_v2
from app.models.file_model import File
from app.schemas.file_schema import FileStatusEnum
from app.utils.uuid6 import uuid7


async def export_large_dataframe_to_csv(
    data_frame: pd.DataFrame,
    workspace_id: UUID,
    db_session: AsyncSession,
    filename_prefix: str = "export",
) -> Optional[tuple[str, str]]:
    """
    Export a large DataFrame to CSV file and create a database record.

    Args:
        data_frame: The pandas DataFrame to export
        workspace_id: The workspace ID to associate the file with
        db_session: Database session for creating file record
        filename_prefix: Prefix for the generated filename

    Returns:
        tuple[str, str]: (file_id, filename) of the created file record, or None if export failed
    """
    try:
        # Create export directory if it doesn't exist
        os.makedirs(settings.CSV_EXPORT_FOLDER, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        csv_file_path = os.path.join(settings.CSV_EXPORT_FOLDER, filename)

        # Export DataFrame to CSV
        data_frame.to_csv(csv_file_path, index=False)
        logger.info(f"Exported DataFrame to CSV: {csv_file_path}")

        # Check file size
        file_size = os.path.getsize(csv_file_path)
        if file_size > settings.MAX_CSV_EXPORT_SIZE:
            logger.info(
                f"Generated CSV file size ({file_size} bytes) exceeds {settings.MAX_CSV_EXPORT_SIZE} bytes limit, removing file"
            )
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)
            return None

        # Create file record in database
        file_id = uuid7()
        file_record = File(
            id=file_id,
            filename=filename,
            mimetype="text/csv",
            size=file_size,
            file_path=csv_file_path,
            status=FileStatusEnum.Uploaded,
            workspace_id=workspace_id,
            source_id=UUID(settings.SYSTEM_EXPORT_SOURCE_ID),
        )

        db_session.add(file_record)
        await db_session.commit()
        await db_session.refresh(file_record)

        logger.info(f"Created file record with ID: {file_record.id}")
        return (str(file_record.id), filename)

    except Exception as e:
        logger.error(f"Failed to export DataFrame to CSV: {str(e)}")
        # Clean up file if it was created but database record failed
        if "csv_file_path" in locals() and os.path.exists(csv_file_path):
            try:
                os.remove(csv_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup CSV file: {cleanup_error}")
        return None


async def get_workspace_id_from_dataset_ids(
    dataset_ids: list[UUID], db_session: AsyncSession
) -> Optional[UUID]:
    """
    Get workspace ID from the first dataset ID in the list.

    Args:
        dataset_ids: List of dataset IDs
        db_session: Database session

    Returns:
        UUID: The workspace ID, or None if not found
    """
    if not dataset_ids:
        return None

    try:
        workspace_id = await dataset_v2.get_workspace_id_of_dataset(
            dataset_id=dataset_ids[0], db_session=db_session
        )
        logger.info(
            f"Retrieved workspace_id: {workspace_id} for dataset: {dataset_ids[0]}"
        )
        return workspace_id
    except Exception as e:
        logger.warning(f"Failed to get workspace_id from dataset: {e}")
        return None
