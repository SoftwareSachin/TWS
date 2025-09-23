import random
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_ingest_files_v2_success(
    test_client_admin_v2, dataset_id_v2, upload_file_to_workspace
):
    unique_suffix = random.randint(100, 999)
    file_id = upload_file_to_workspace["id"]
    ingest_data = {
        "name": f"Test Ingestion V2 {unique_suffix}",
        "file_ids": [file_id],
        "chunking_config": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking_strategy": "recursive",
        },
        "metadata": {"source": "test", "version": "v2"},
    }
    response = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2}/ingest", json=ingest_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Ingestion process initiated"
    assert "data" in response_data
    assert isinstance(response_data["data"], list)


@pytest.mark.asyncio
async def test_ingest_files_v2_invalid_data(test_client_admin_v2, dataset_id_v2):

    response = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2}/ingest", json={}
    )
    # V2 endpoint accepts empty body and returns 200 with warning about processing
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_ingest_files_v2_invalid_dataset(
    test_client_admin_v2, upload_file_to_workspace
):
    unique_suffix = random.randint(100, 999)
    fake_dataset_id = str(uuid4())
    file_id = upload_file_to_workspace["id"]
    ingest_data = {
        "name": f"Invalid Dataset Ingestion V2 {unique_suffix}",
        "file_ids": [file_id],
        "chunking_config": {"chunk_size": 1000, "chunk_overlap": 200},
    }
    response = await test_client_admin_v2.post(
        f"/dataset/{fake_dataset_id}/ingest", json=ingest_data
    )
    assert response.status_code == 404
