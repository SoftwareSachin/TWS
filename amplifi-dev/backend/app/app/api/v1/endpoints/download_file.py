from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.crud.source_connector_crud import CRUDSource
from app.models.source_model import Source
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData

router = APIRouter()

crud = CRUDSource(Source)


@router.get("/files/{file_id}/download")
async def download_file(
    file_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    db: AsyncSession = Depends(deps.get_db),
):
    """
    Downloads a file based on the given file_id.

    Required roles:
    - admin
    - developer

    Parameters:
    - file_id (UUID): The ID of the file to be downloaded.

    Returns:
    - A file response if the file exists, otherwise an error response.
    """
    file_record = await crud.get_file_by_fileid(file_id=file_id, db_session=db)

    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = file_record.file_path

    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found on server")

    return FileResponse(
        file_path, filename=file_record.filename, media_type="application/octet-stream"
    )
