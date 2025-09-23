import uuid

import pytest


@pytest.fixture(scope="module")
def mock_dataset_id():
    """
    Mock dataset ID for testing document endpoints.
    Dataset creation typically uses V2 API fixtures (dataset_id_v2)
    but we need to test V1 document endpoints, so we use a mock to avoid
    the R2R infrastructure issues with real dataset creation.
    """
    return str(uuid.uuid4())


@pytest.fixture(scope="module")
def mock_document_id():
    """
    Mock document ID for testing document endpoints, since we're using a mock dataset ID.
    """
    return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_get_documents(test_client_admin, mock_dataset_id):

    response = await test_client_admin.get(f"/dataset/{mock_dataset_id}/documents")

    assert response.status_code == 404
    response_data = response.json()
    assert "detail" in response_data


@pytest.mark.asyncio
async def test_get_document_chunks(
    test_client_admin, mock_dataset_id, mock_document_id
):

    response = await test_client_admin.get(
        f"/dataset/{mock_dataset_id}/documents/{mock_document_id}/chunks"
    )

    # Should return 404 since mock document doesn't exist
    assert response.status_code == 404
    response_data = response.json()
    assert "detail" in response_data
    assert response_data["detail"] == "Not Found"


@pytest.mark.asyncio
async def test_get_document_status(
    test_client_admin, mock_dataset_id, mock_document_id
):
    response = await test_client_admin.get(
        f"/dataset/{mock_dataset_id}/documents/{mock_document_id}/status"
    )

    # Should return 404 since mock document doesn't exist
    assert response.status_code == 404
    response_data = response.json()
    assert "detail" in response_data
    assert response_data["detail"] == "Not Found"


@pytest.mark.asyncio
async def test_delete_document(test_client_admin, mock_dataset_id, mock_document_id):

    response = await test_client_admin.delete(
        f"/dataset/{mock_dataset_id}/documents/{mock_document_id}"
    )

    # Should return 404 since mock document doesn't exist
    assert response.status_code == 404
    response_data = response.json()
    assert "detail" in response_data
    assert response_data["detail"] == "Not Found"
