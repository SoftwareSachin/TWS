import time
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_assign_tool_to_workspace_success(
    test_client_admin, workspace_id, tool_id
):
    request_data = {
        "name": "Test Workspace Tool",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Tools assigned successfully"
    assert tool_id in response_data["data"]["tool_ids"]


@pytest.mark.asyncio
async def test_assign_tool_to_workspace_invalid_tool(test_client_admin, workspace_id):
    invalid_tool_id = str(uuid4())
    request_data = {
        "tool_ids": [invalid_tool_id],
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response.status_code == 400
    response_data = response.json()
    assert "Tools not found" in response_data["detail"]


@pytest.mark.asyncio
async def test_assign_tool_to_workspace_empty_tool_ids(test_client_admin, workspace_id):
    request_data = {
        "name": "Test Tool with Empty IDs",
        "tool_ids": [],
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_assign_tool_to_workspace_duplicate_name(
    test_client_admin, workspace_id, tool_id
):
    # First assignment
    request_data = {
        "name": "Duplicate Tool Name",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }
    response1 = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response1.status_code == 200

    # Second assignment with same name should fail
    response2 = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response2.status_code == 400
    response_data = response2.json()
    assert "already exists" in response_data["detail"]


@pytest.mark.asyncio
async def test_assign_tool_to_workspace_invalid_workspace(test_client_admin, tool_id):
    """Test assign tool to non-existent workspace"""
    invalid_workspace_id = str(uuid4())
    request_data = {
        "name": "Test Tool",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{invalid_workspace_id}/tool", json=request_data
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "not authorized" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_get_workspace_tool_by_id_success(
    test_client_admin, workspace_id, workspace_tool_id
):
    response = await test_client_admin.get(
        f"/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}"
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Workspace tool retrieved successfully."
    assert response_data["data"]["id"] == str(workspace_tool_id)


@pytest.mark.asyncio
async def test_get_workspace_tool_by_id_not_found(test_client_admin, workspace_id):
    non_existent_workspace_tool_id = str(uuid4())
    response = await test_client_admin.get(
        f"/workspace/{workspace_id}/workspace_tool/{non_existent_workspace_tool_id}"
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "Workspace tool not found" in response_data["detail"]


@pytest.mark.asyncio
async def test_get_workspace_tools_success(test_client_admin, workspace_id):
    response = await test_client_admin.get(f"/workspace/{workspace_id}/tool")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Data Got correctly"
    assert isinstance(response_data["data"]["items"], list)


@pytest.mark.asyncio
async def test_get_workspace_tools_invalid_workspace(test_client_admin):
    invalid_workspace_id = str(uuid4())
    response = await test_client_admin.get(f"/workspace/{invalid_workspace_id}/tool")
    assert response.status_code == 404
    response_data = response.json()
    assert "not authorized" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_update_workspace_tool_success(
    test_client_admin, workspace_id, workspace_tool_id, tool_id
):
    unique_suffix = str(int(time.time()))

    update_data = {
        "name": f"Updated Workspace Tool {unique_suffix}",
        "description": "Updated description for testing",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }

    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
        json=update_data,
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Workspace tool updated successfully"
    assert tool_id in response_data["data"]["tool_ids"]


@pytest.mark.asyncio
async def test_update_workspace_tool_not_found(
    test_client_admin, workspace_id, tool_id
):
    non_existent_workspace_tool_id = str(uuid4())
    update_data = {
        "name": "Non-existent Workspace Tool",
        "description": "This should fail",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }

    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/workspace_tool/{non_existent_workspace_tool_id}",
        json=update_data,
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "Workspace tool not found" in response_data["detail"]


@pytest.mark.asyncio
async def test_update_workspace_tool_empty_tool_ids(
    test_client_admin, workspace_id, workspace_tool_id
):
    update_data = {
        "name": "Test Workspace Tool",
        "description": "Testing with empty tool IDs",
        "tool_ids": [],
        "dataset_ids": [],
    }

    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
        json=update_data,
    )
    assert response.status_code == 400
    response_data = response.json()
    assert "tool_ids must be provided" in response_data["detail"]


@pytest.mark.asyncio
async def test_update_workspace_tool_invalid_workspace(
    test_client_admin, workspace_tool_id, tool_id
):
    invalid_workspace_id = str(uuid4())
    update_data = {
        "name": "Test Workspace Tool",
        "description": "Testing with invalid workspace",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }

    response = await test_client_admin.put(
        f"/workspace/{invalid_workspace_id}/workspace_tool/{workspace_tool_id}",
        json=update_data,
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "not authorized" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_update_workspace_tool_duplicate_name(
    test_client_admin, workspace_id, workspace_tool_id, tool_id
):
    # First create another workspace tool with a specific name
    existing_name = f"Existing Tool {int(time.time())}"

    request_data = {
        "name": existing_name,
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }
    response1 = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response1.status_code == 200

    # Now try to update another workspace tool to the same name
    update_data = {
        "name": existing_name,
        "description": "This should fail due to duplicate name",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }

    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
        json=update_data,
    )
    assert response.status_code == 400
    response_data = response.json()
    assert "already exists" in response_data["detail"]


@pytest.mark.asyncio
async def test_unassign_tool_from_workspace_success(
    test_client_admin, workspace_id, workspace_tool_id
):
    response = await test_client_admin.delete(
        f"/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}"
    )
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_unassign_tool_from_workspace_not_found(test_client_admin, workspace_id):
    non_existent_workspace_tool_id = str(uuid4())
    response = await test_client_admin.delete(
        f"/workspace/{workspace_id}/workspace_tool/{non_existent_workspace_tool_id}"
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "Workspace tool not found" in response_data["detail"]


@pytest.mark.asyncio
async def test_unassign_tool_from_workspace_invalid_workspace(
    test_client_admin, workspace_tool_id
):
    invalid_workspace_id = str(uuid4())
    response = await test_client_admin.delete(
        f"/workspace/{invalid_workspace_id}/workspace_tool/{workspace_tool_id}"
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "not authorized" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_get_system_tools_success(test_client_admin):
    response = await test_client_admin.get("/system-tools")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Data Got correctly"
    assert isinstance(response_data["data"]["items"], list)


@pytest.mark.asyncio
async def test_assign_tool_missing_required_fields(test_client_admin, workspace_id):
    request_data = {
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_workspace_tool_missing_required_fields(
    test_client_admin, workspace_id, workspace_tool_id
):
    """Test update workspace tool with missing required fields"""
    update_data = {
        "description": "Missing tool_ids field",
    }

    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/workspace_tool/{workspace_tool_id}",
        json=update_data,
    )
    assert response.status_code == 422
