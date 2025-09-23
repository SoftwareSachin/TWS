import io
from test.test_files.aristotle import aristotle_bytes

import pytest

from app.be_core.config import settings


def file_sanity_check(file_data: dict):
    assert "filename" in file_data
    assert "id" in file_data
    assert "size" in file_data
    assert file_data["status"] in ["Uploaded", "Processing"]


# @pytest.mark.asyncio
# async def test_file_upload(upload_file_to_workspace):
#     file_sanity_check(upload_file_to_workspace)
#     assert upload_file_to_workspace["filename"].endswith(".txt")


@pytest.mark.asyncio
async def test_pdf_file_upload(upload_pdf_to_workspace):

    file_sanity_check(upload_pdf_to_workspace)
    assert upload_pdf_to_workspace["filename"].endswith(".pdf")


# @pytest.mark.asyncio
# async def test_multi_file_upload(test_client_admin, workspace_id):
#     with open("test/test_files/Socrates_file.pdf", "rb") as pdf_file:
#         file_content = aristotle_bytes
#         file_like_object = io.BytesIO(file_content)
#         file_like_object.name = "aristotle.txt"

#         files = [("files", file_like_object), ("files", pdf_file)]

#         response = await test_client_admin.post(
#             f"/workspace/{workspace_id}/file_upload",
#             files=files,
#         )
#         response.raise_for_status()
#         response_data = response.json()["data"]
#         assert len(response_data) == 2
#         for file_data in response_data:
#             file_sanity_check(file_data)


@pytest.mark.asyncio
async def test_multi_file_upload_fail(test_client_admin, workspace_id):
    with open("test/test_files/Socrates_file.pdf", "rb") as pdf_file:
        file_content = aristotle_bytes
        file_like_object = io.BytesIO(file_content)
        file_like_object.name = "aristotle.unsupportedtype"

        files = [("files", file_like_object), ("files", pdf_file)]

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/file_upload",
            files=files,
        )
        response.raise_for_status()
        response_data = response.json()["data"]
        assert len(response_data) == 2
        assert response_data[0]["status"] == "Failed"
        assert response_data[1]["status"] in ["Uploaded", "Processing"]
        file_sanity_check(response_data[1])
        assert "Invalid file type" in response_data[0]["error"]


@pytest.mark.skip(reason="Takes too long")
@pytest.mark.asyncio
async def test_single_large_file_upload_fail(test_client_admin, workspace_id):
    """Test uploading a single large file, too large to be uploaded."""
    size_in_bytes = 5 * settings.MAX_FILE_SIZE_UPLOADED_FILES // 4
    large_file = io.BytesIO(b"\x00" * size_in_bytes)
    large_file.name = "large_file.txt"
    files = [("files", large_file)]
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/file_upload",
        files=files,
    )
    assert response.status_code == 413


@pytest.mark.skip(reason="Takes too long")
@pytest.mark.asyncio
async def test_multi_large_file_upload_fail(test_client_admin, workspace_id):
    """Test uploading multiple large files, individually small enough but together too large."""
    size_in_bytes = settings.MAX_FILE_SIZE_UPLOADED_FILES // 9
    files = [
        ("files", (f"large_file_{i}.txt", io.BytesIO(b"\x00" * size_in_bytes)))
        for i in range(10)
    ]
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/file_upload",
        files=files,
    )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_no_files(test_client_admin, workspace_id):
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/file_upload",
    )
    assert response.status_code == 400
