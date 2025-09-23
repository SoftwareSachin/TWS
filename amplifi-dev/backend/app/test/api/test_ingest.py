import pytest


@pytest.mark.skip("Uses R2R")
@pytest.mark.asyncio
async def test_ingest_files(test_client_admin, dataset_id):

    ingest_body = {"name": "Test Ingestion"}

    response = await test_client_admin.post(
        f"/dataset/{dataset_id}/ingest", json=ingest_body
    )
    print(response.json())

    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"
    response_data = response.json()

    assert response_data["message"] == "Ingestion process initiated"
    assert "data" in response_data
    assert isinstance(response_data["data"], list)

    first_item = response_data["data"][0]
    required_keys = ["chunking_config_id", "created_at", "file_id", "filename"]
    for key in required_keys:
        assert key in first_item

    print(f"Ingestion initiation response: {response_data}")
