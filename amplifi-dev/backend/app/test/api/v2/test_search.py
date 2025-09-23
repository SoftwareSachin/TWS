import random
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_perform_search_v2_success(
    test_client_admin_v2, workspace_id, dataset_id_v2
):
    search_data = {
        "query": "summarize the document",
        "dataset_ids": [dataset_id_v2],
        "vector_search_settings": {"search_limit": 5},
        "perform_aggregate_search": True,
        "perform_graph_search": False,
        "perform_hybrid_search": False,
        "calculate_metrics": False,
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/search", json=search_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Finished Query & Eval"
    assert "data" in response_data
    assert "meta" in response_data
    assert response_data["meta"]["workspace_id"] == workspace_id
    assert response_data["meta"]["dataset_ids"] == [dataset_id_v2]


@pytest.mark.asyncio
async def test_perform_search_v2_no_aggregate(
    test_client_admin_v2, workspace_id, dataset_id_v2
):
    search_data = {
        "query": "analyze the content",
        "dataset_ids": [dataset_id_v2],
        "vector_search_settings": {"search_limit": 3},
        "perform_aggregate_search": False,
        "perform_graph_search": False,
        "perform_hybrid_search": False,
        "calculate_metrics": False,
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/search", json=search_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Finished Query & Eval"
    assert "data" in response_data
    assert "meta" in response_data
    assert response_data["meta"]["workspace_id"] == workspace_id
    assert response_data["meta"]["dataset_ids"] == [dataset_id_v2]
