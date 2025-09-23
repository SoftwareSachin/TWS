import json
import random
import uuid
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_create_workspace(test_client_admin, organization_id):
    unique_suffix = random.randint(100, 999)  # 3-digit random number
    unique_name = f"TestWS-{unique_suffix}"
    workspace_data = {
        "name": unique_name,
        "description": "Workspace for testing",
        "is_active": True,
    }
    response = await test_client_admin.post(
        f"/organization/{organization_id}/workspace", json=workspace_data
    )
    assert response.status_code == 200
    assert response.json()["data"]["name"] == unique_name
    assert response.json()["message"] == "Workspace created successfully"


@pytest.mark.asyncio
async def test_get_workspace_by_id(test_client_admin, organization_id, workspace_id):
    response = await test_client_admin.get(
        f"/organization/{organization_id}/workspace/{workspace_id}"
    )
    assert response.status_code == 200
    assert response.json()["data"]["id"] == workspace_id


@pytest.mark.asyncio
async def test_get_all_workspaces(test_client_admin, organization_id):
    response = await test_client_admin.get(f"/organization/{organization_id}/workspace")
    assert response.status_code == 200
    assert "data" in response.json()
    assert "items" in response.json()["data"]
    assert isinstance(response.json()["data"]["items"], list)


@pytest.mark.asyncio
async def test_add_users_in_workspace(
    test_client_admin, user_id, organization_id, workspace_id
):
    user_data = {"user_ids": [str(user_id)]}
    response = await test_client_admin.post(
        f"/organization/{organization_id}/workspace/{workspace_id}/add_users",
        json=user_data,
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Added users to the workspace successfully"
    assert "data" in response.json()
    assert set(response.json()["data"]["user_ids"]) == set(user_data["user_ids"])


@pytest.mark.asyncio
async def test_add_users_in_workspace_fail_not_belong_to_org(
    test_client_admin, organization_id, workspace_id
):
    # Payload with an invalid user_id that doesn't belong to the organization
    user_data = {"user_ids": [str(uuid4())]}

    # Send the request to add the user to the workspace
    response = await test_client_admin.post(
        f"/organization/{organization_id}/workspace/{workspace_id}/add_users",
        json=user_data,
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == f"User {user_data['user_ids'][0]} does not belong to organization and cannot be added to workspace {workspace_id}."
    )


@pytest.mark.asyncio
async def test_get_users_in_workspace(test_client_admin, organization_id, workspace_id):
    response = await test_client_admin.get(
        f"/organization/{organization_id}/workspace/{workspace_id}/get_users"
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "data" in json_response
    assert "items" in json_response["data"]
    assert isinstance(json_response["data"]["items"], list)


@pytest.mark.asyncio
async def test_get_users_in_workspace_forbidden(test_client_admin, organization_id):
    fake_workspace_id = uuid4()
    response = await test_client_admin.get(
        f"/organization/{organization_id}/workspace/{fake_workspace_id}/get_users"
    )
    assert response.status_code == 404
    json_response = response.json()
    assert json_response["detail"] == "You are not authorized to perform this action"


@pytest.mark.asyncio
async def test_remove_users_in_workspace(
    test_client_admin, user_id, organization_id, workspace_id
):
    user_data = {"user_ids": [str(user_id)]}
    response = await test_client_admin.post(
        f"/organization/{organization_id}/workspace/{workspace_id}/remove_users",
        json=user_data,
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Users deleted successfully"
    assert "data" in response.json()
    assert set(response.json()["data"]["user_ids"]) == set(user_data["user_ids"])


@pytest.mark.asyncio
async def test_remove_users_in_workspace_fail_no_matching_users(
    test_client_admin, organization_id, workspace_id
):
    # Payload with user_ids that are not in the workspace
    user_data = {"user_ids": [str(uuid4()), str(uuid4())]}  # Invalid user UUIDs

    # Send the request to remove users from the workspace
    response = await test_client_admin.post(
        f"/organization/{organization_id}/workspace/{workspace_id}/remove_users",
        json=user_data,
    )

    # Assertions
    assert response.status_code == 404
    assert response.json()["detail"] == "No matching users found in workspace"
