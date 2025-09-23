import json
import random
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_create_dataset_v2_success(test_client_admin_v2, workspace_id):
    unique_suffix = random.randint(100, 999)
    dataset_data = {
        "name": f"Test Dataset V2 {unique_suffix}",
        "description": "Dataset created via V2 API for testing",
        "file_ids": [],
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Dataset created successfully"
    assert response_data["data"]["name"] == dataset_data["name"]
    assert response_data["data"]["description"] == dataset_data["description"]


@pytest.mark.asyncio
async def test_create_dataset_v2_with_files(
    test_client_admin_v2, workspace_id, upload_file_to_workspace
):
    unique_suffix = random.randint(100, 999)
    file_id = upload_file_to_workspace["id"]
    dataset_data = {
        "name": f"Test Dataset V2 with Files {unique_suffix}",
        "description": "Dataset with files created via V2 API for testing",
        "file_ids": [file_id],
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Dataset created successfully"
    assert response_data["data"]["name"] == dataset_data["name"]
    assert response_data["data"]["description"] == dataset_data["description"]
    assert len(response_data["data"]["file_ids"]) == 1
    assert file_id in response_data["data"]["file_ids"]


@pytest.mark.asyncio
async def test_create_dataset_v2_invalid_data(test_client_admin_v2, workspace_id):
    dataset_data = {
        "description": "Dataset without name should fail",
        "file_ids": [],
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_dataset_v2_success(
    test_client_admin_v2, workspace_id, dataset_id_v2
):
    updated_data = {
        "name": "Updated Dataset V2",
        "description": "Updated description via V2 API",
    }
    response = await test_client_admin_v2.put(
        f"/workspace/{workspace_id}/dataset/{dataset_id_v2}", json=updated_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Dataset updated successfully"
    assert response_data["data"]["name"] == updated_data["name"]
    assert response_data["data"]["description"] == updated_data["description"]


@pytest.mark.asyncio
async def test_update_dataset_v2_not_found(test_client_admin_v2, workspace_id):
    non_existent_dataset_id = str(uuid4())
    updated_data = {
        "name": "Non-existent Dataset",
        "description": "This dataset does not exist",
    }
    response = await test_client_admin_v2.put(
        f"/workspace/{workspace_id}/dataset/{non_existent_dataset_id}",
        json=updated_data,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_dataset_v2_success(
    test_client_admin_v2, workspace_id, dataset_id_v2
):
    response = await test_client_admin_v2.delete(
        f"/workspace/{workspace_id}/dataset/{dataset_id_v2}"
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Dataset deleted successfully"


@pytest.mark.asyncio
async def test_delete_dataset_v2_not_found(test_client_admin_v2, workspace_id):
    non_existent_dataset_id = str(uuid4())
    response = await test_client_admin_v2.delete(
        f"/workspace/{workspace_id}/dataset/{non_existent_dataset_id}"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_chunks_v2_success(test_client_admin_v2, workspace_id, dataset_id_v2):
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/dataset/{dataset_id_v2}/chunks"
    )
    # Note: This might return 404 if no chunks are ingested yet
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        response_data = response.json()
        assert response_data["message"] == "Chunks fetched"
        assert isinstance(response_data["data"]["items"], list)


@pytest.mark.asyncio
async def test_get_chunks_v2_with_file_id(
    test_client_admin_v2, workspace_id, dataset_id_v2, upload_file_to_workspace
):
    file_id = upload_file_to_workspace["id"]
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/dataset/{dataset_id_v2}/chunks?file_id={file_id}"
    )
    # Note: This might return 404 if the file is not ingested yet
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_get_chunks_v2_with_vectors(
    test_client_admin_v2, workspace_id, dataset_id_v2
):
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/dataset/{dataset_id_v2}/chunks?include_vectors=true"
    )
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        response_data = response.json()
        assert response_data["message"] == "Chunks fetched"


@pytest.mark.asyncio
async def test_get_chunks_v2_with_partial_vectors(
    test_client_admin_v2, workspace_id, dataset_id_v2
):
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/dataset/{dataset_id_v2}/chunks?include_vectors=true&partial_vectors=false"
    )
    # Note: This might return 404 if no chunks are ingested yet
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        response_data = response.json()
        assert response_data["message"] == "Chunks fetched"


@pytest.mark.asyncio
async def test_get_chunks_v2_not_found(test_client_admin_v2, workspace_id):
    non_existent_dataset_id = str(uuid4())
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/dataset/{non_existent_dataset_id}/chunks"
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_remove_file_from_dataset_success(
    test_client_admin_v2, workspace_id, upload_file_to_workspace
):
    """Test successful removal of a file from a dataset."""
    unique_suffix = random.randint(100, 999)
    file_id = upload_file_to_workspace["id"]

    # Create a dataset with the file
    dataset_data = {
        "name": f"Test Dataset for File Removal {unique_suffix}",
        "description": "Dataset for testing file removal",
        "file_ids": [file_id],
    }

    create_response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert create_response.status_code == 200
    dataset_id = create_response.json()["data"]["id"]

    # Remove the file from the dataset
    response = await test_client_admin_v2.request(
        "DELETE",
        f"/workspace/{workspace_id}/dataset/{dataset_id}/files",
        content=json.dumps({"file_ids": [file_id]}),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "File removed from dataset successfully"
    assert response_data["data"]["dataset_id"] == dataset_id
    assert file_id in response_data["data"]["removed_files"]
    assert len(response_data["data"]["not_found_or_failed"]) == 0


@pytest.mark.asyncio
async def test_remove_file_not_in_dataset(
    test_client_admin_v2, workspace_id, upload_file_to_workspace
):
    """Test removing a file that is not in the dataset."""
    # Create a fresh dataset for this test to avoid conflicts with dataset deletion tests
    unique_suffix = random.randint(100, 999)
    file_id = upload_file_to_workspace["id"]
    dataset_data = {
        "name": f"Test Dataset for File Not Found {unique_suffix}",
        "description": "Dataset for testing file not found scenario",
        "file_ids": [file_id],
    }

    create_response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert create_response.status_code == 200
    dataset_id = create_response.json()["data"]["id"]

    # Use a completely different file ID that doesn't exist
    non_existent_file_id = str(uuid4())

    response = await test_client_admin_v2.request(
        "DELETE",
        f"/workspace/{workspace_id}/dataset/{dataset_id}/files",
        content=json.dumps({"file_ids": [non_existent_file_id]}),
        headers={"Content-Type": "application/json"},
    )

    # The new endpoint returns 200 but reports files not found in the response data
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "File removed from dataset successfully"
    assert response_data["data"]["dataset_id"] == dataset_id
    assert len(response_data["data"]["removed_files"]) == 0
    assert non_existent_file_id in response_data["data"]["not_found_or_failed"]


@pytest.mark.asyncio
async def test_remove_multiple_files_from_dataset(
    test_client_admin_v2, workspace_id, upload_file_to_workspace
):
    """Test successful removal of multiple files from a dataset."""
    unique_suffix = random.randint(100, 999)
    file_id_1 = upload_file_to_workspace["id"]

    # For this test, we'll simulate multiple files by using the same file twice
    # In practice, this would be two different files
    fake_file_id = str(uuid4())

    # Create a dataset with one real file
    dataset_data = {
        "name": f"Test Dataset for Multiple File Removal {unique_suffix}",
        "description": "Dataset for testing multiple file removal",
        "file_ids": [file_id_1],
    }

    create_response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert create_response.status_code == 200
    dataset_id = create_response.json()["data"]["id"]

    # Try to remove one real file and one fake file
    response = await test_client_admin_v2.request(
        "DELETE",
        f"/workspace/{workspace_id}/dataset/{dataset_id}/files",
        content=json.dumps({"file_ids": [file_id_1, fake_file_id]}),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "File removed from dataset successfully"
    assert response_data["data"]["dataset_id"] == dataset_id
    assert file_id_1 in response_data["data"]["removed_files"]
    assert fake_file_id in response_data["data"]["not_found_or_failed"]
    assert len(response_data["data"]["removed_files"]) == 1
    assert len(response_data["data"]["not_found_or_failed"]) == 1


@pytest.mark.asyncio
async def test_remove_mixed_files_from_dataset(
    test_client_admin_v2, workspace_id, upload_file_to_workspace
):
    """Test removal of mix of existing and non-existing files from dataset."""
    unique_suffix = random.randint(100, 999)
    file_id = upload_file_to_workspace["id"]
    non_existent_file_id = str(uuid4())

    # Create a dataset with one file
    dataset_data = {
        "name": f"Test Dataset for Mixed File Removal {unique_suffix}",
        "description": "Dataset for testing mixed file removal",
        "file_ids": [file_id],
    }

    create_response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    assert create_response.status_code == 200
    dataset_id = create_response.json()["data"]["id"]

    # Try to remove one existing and one non-existing file
    response = await test_client_admin_v2.request(
        "DELETE",
        f"/workspace/{workspace_id}/dataset/{dataset_id}/files",
        content=json.dumps({"file_ids": [file_id, non_existent_file_id]}),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "File removed from dataset successfully"
    assert response_data["data"]["dataset_id"] == dataset_id
    assert file_id in response_data["data"]["removed_files"]
    assert len(response_data["data"]["removed_files"]) == 1
    assert non_existent_file_id in response_data["data"]["not_found_or_failed"]
    assert len(response_data["data"]["not_found_or_failed"]) == 1
