"""Tests for download file endpoint."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio


class TestDownloadFile:
    """Test cases for file download functionality."""

    @pytest.mark.asyncio
    async def test_download_file_success(self, test_client_admin):
        """Test successful file download."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".txt"
        ) as temp_file:
            temp_file.write("This is test file content for download testing.")
            temp_file_path = temp_file.name
            temp_filename = Path(temp_file_path).name

        try:
            file_id = str(uuid4())

            # Mock the file record
            mock_file_record = MagicMock()
            mock_file_record.file_path = temp_file_path
            mock_file_record.filename = temp_filename

            # Mock the CRUD method
            with patch(
                "app.api.v1.endpoints.download_file.crud.get_file_by_fileid",
                new_callable=AsyncMock,
            ) as mock_get_file:
                mock_get_file.return_value = mock_file_record

                response = await test_client_admin.get(f"/files/{file_id}/download")

                assert response.status_code == 200
                assert response.headers["content-type"].startswith(
                    "application/octet-stream"
                )
                # Check if filename is in content-disposition header
                content_disposition = response.headers.get("content-disposition", "")
                assert temp_filename in content_disposition

                # Verify content
                content = response.content.decode()
                assert "This is test file content for download testing." in content

                # Verify the CRUD method was called with correct parameters
                mock_get_file.assert_called_once()

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_download_file_not_found_in_database(self, test_client_admin):
        """Test download when file record doesn't exist in database."""
        file_id = str(uuid4())

        # Mock the CRUD method to return None (file not found in DB)
        with patch(
            "app.api.v1.endpoints.download_file.crud.get_file_by_fileid",
            new_callable=AsyncMock,
        ) as mock_get_file:
            mock_get_file.return_value = None

            response = await test_client_admin.get(f"/files/{file_id}/download")

            assert response.status_code == 404
            data = response.json()
            assert data["detail"] == "File not found"

            # Verify the CRUD method was called
            mock_get_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_not_found_on_server(self, test_client_admin):
        """Test download when file record exists but file is missing from server."""
        file_id = str(uuid4())
        non_existent_path = "/path/to/non/existent/file.txt"

        # Mock the file record with non-existent path
        mock_file_record = MagicMock()
        mock_file_record.file_path = non_existent_path
        mock_file_record.filename = "missing_file.txt"

        # Mock the CRUD method
        with patch(
            "app.api.v1.endpoints.download_file.crud.get_file_by_fileid",
            new_callable=AsyncMock,
        ) as mock_get_file:
            mock_get_file.return_value = mock_file_record

            response = await test_client_admin.get(f"/files/{file_id}/download")

            assert response.status_code == 404
            data = response.json()
            assert data["detail"] == "File not found on server"

    @pytest.mark.asyncio
    async def test_download_file_invalid_uuid(self, test_client_admin):
        """Test download with invalid UUID format."""
        invalid_file_id = "invalid-uuid-format"

        response = await test_client_admin.get(f"/files/{invalid_file_id}/download")

        assert response.status_code == 422  # Validation error for invalid UUID
