import asyncio
from uuid import uuid4

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_most_recent_graph(test_client_admin_v2, dataset_id_v2_empty):
    response = await test_client_admin_v2.get(
        f"/dataset/{dataset_id_v2_empty}/graph",
    )
    assert response.status_code == 404
    create_resp = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2_empty}/graph",
        json={"entity_types": ["Person", "Organization", "Location"]},
    )
    graph_id = create_resp.json()["data"]["id"]
    assert create_resp.status_code == 200
    response = await test_client_admin_v2.get(
        f"/dataset/{dataset_id_v2_empty}/graph",
    )
    data = response.json()
    assert data["data"]["id"] == graph_id
    assert data["data"]["dataset_id"] == dataset_id_v2_empty


@pytest.mark.asyncio
async def test_create_graph_for_dataset(test_client_admin_v2, dataset_id_v2_empty):
    response = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2_empty}/graph",
        json={"entity_types": ["Person", "Organization", "Location"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"].startswith("Graph created")
    assert "data" in data
    assert data["data"]["dataset_id"] == dataset_id_v2_empty
    # Verify that newly created graphs start with NOT_STARTED status
    assert data["data"]["entities_status"] == "not_started"
    assert data["data"]["relationships_status"] == "not_started"


@pytest.mark.asyncio
async def test_get_graph_by_id(test_client_admin_v2, dataset_id_v2_empty):
    # First, create a graph to get a real graph_id
    create_resp = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2_empty}/graph",
        json={"entity_types": ["Person", "Organization", "Location"]},
    )
    assert create_resp.status_code == 200
    graph_id = create_resp.json()["data"]["id"]
    # Now, get the graph by ID
    response = await test_client_admin_v2.get(
        f"/dataset/{dataset_id_v2_empty}/graph/{graph_id}",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["id"] == graph_id
    assert data["data"]["dataset_id"] == dataset_id_v2_empty


@pytest.mark.asyncio
async def test_delete_graph(test_client_admin_v2, dataset_id_v2_empty):
    # First, create a graph to get a real graph_id
    create_resp = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2_empty}/graph",
        json={"entity_types": ["Person", "Organization", "Location"]},
    )
    assert create_resp.status_code == 200
    graph_id = create_resp.json()["data"]["id"]

    # Verify the graph exists before deletion
    get_resp = await test_client_admin_v2.get(
        f"/dataset/{dataset_id_v2_empty}/graph/{graph_id}",
    )
    assert get_resp.status_code == 200

    # Delete the graph
    delete_resp = await test_client_admin_v2.delete(
        f"/dataset/{dataset_id_v2_empty}/graph/{graph_id}",
    )
    assert delete_resp.status_code == 200
    data = delete_resp.json()
    assert data["message"] == "Graph soft deleted successfully"
    assert data["data"]["id"] == graph_id
    assert data["data"]["dataset_id"] == dataset_id_v2_empty

    # Verify the graph is no longer accessible (soft deleted)
    get_resp_after_delete = await test_client_admin_v2.get(
        f"/dataset/{dataset_id_v2_empty}/graph/{graph_id}",
    )
    assert get_resp_after_delete.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_graph(test_client_admin_v2, dataset_id_v2_empty):
    # Try to delete a non-existent graph
    fake_graph_id = str(uuid4())
    response = await test_client_admin_v2.delete(
        f"/dataset/{dataset_id_v2_empty}/graph/{fake_graph_id}",
    )
    assert response.status_code == 404


@pytest.mark.skip("Polling takes too long.")
@pytest.mark.asyncio
async def test_get_entities_relationships(test_client_admin_v2, dataset_id_v2_empty):
    # First, create a graph to get a real graph_id
    create_resp = await test_client_admin_v2.post(
        f"/dataset/{dataset_id_v2_empty}/graph",
        json={"entity_types": ["Person", "Organization", "Location"]},
    )
    assert create_resp.status_code == 200
    graph_id = create_resp.json()["data"]["id"]

    # Poll the graph status until it's either 'success' or 'failed'
    max_retries = 6
    delay_seconds = 10
    final_status = None

    for _ in range(max_retries):
        status_resp = await test_client_admin_v2.get(
            f"/dataset/{dataset_id_v2_empty}/graph/{graph_id}"
        )
        assert status_resp.status_code == 200
        entities_status = status_resp.json()["data"]["entities_status"]
        relationships_status = status_resp.json()["data"]["relationships_status"]
        # Check if both entities and relationships are in a final state
        if entities_status in {"success", "failed"} and relationships_status in {
            "success",
            "failed",
        }:
            final_status = {
                "entities": entities_status,
                "relationships": relationships_status,
            }
            break
        # Verify intermediate states are valid
        assert entities_status in {"not_started", "pending", "success", "failed"}
        assert relationships_status in {"not_started", "pending", "success", "failed"}
        await asyncio.sleep(delay_seconds)

    assert (
        final_status is not None
        and final_status["entities"] == "success"
        and final_status["relationships"] == "success"
    ), f"Graph construction failed or timed out (final status: {final_status})"

    response = await test_client_admin_v2.get(
        f"/graph/{graph_id}/entities-relationships",
    )
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "entities" in data["data"]
    assert "relationships" in data["data"]
