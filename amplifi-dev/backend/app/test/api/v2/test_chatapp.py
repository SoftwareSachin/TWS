import random
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_create_chatapp_v2_success(
    test_client_admin_v2, workspace_id, agent_id_v2
):
    unique_suffix = random.randint(100, 999)
    chatapp_data = {
        "name": f"Test ChatApp V2 {unique_suffix}",
        "description": "ChatApp created via V2 API for testing",
        "agent_id": agent_id_v2,
        "voice_enabled": False,
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/chat_app", json=chatapp_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "ChatApp created successfully"
    assert response_data["data"]["name"] == chatapp_data["name"]
    assert response_data["data"]["description"] == chatapp_data["description"]


@pytest.mark.asyncio
async def test_create_chatapp_v2_invalid_data(test_client_admin_v2, workspace_id):
    chatapp_data = {
        "description": "ChatApp without name should fail",
        "voice_enabled": False,
    }
    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/chat_app", json=chatapp_data
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_chatapp_v2_success(
    test_client_admin_v2, workspace_id, chatapp_id_v2
):
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/chat_app/{chatapp_id_v2}"
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "ChatApp retrieved successfully"
    assert response_data["data"]["id"] == chatapp_id_v2
    assert response_data["data"]["name"] is not None


@pytest.mark.asyncio
async def test_get_chatapp_v2_not_found(test_client_admin_v2, workspace_id):
    fake_chatapp_id = str(uuid4())
    response = await test_client_admin_v2.get(
        f"/workspace/{workspace_id}/chat_app/{fake_chatapp_id}"
    )
    assert response.status_code == 404
    response_data = response.json()
    assert (
        "not authorized" in response_data["detail"].lower()
        or "not found" in response_data["detail"].lower()
    )


@pytest.mark.asyncio
async def test_get_all_chatapps_v2(test_client_admin_v2, workspace_id):
    response = await test_client_admin_v2.get(f"/workspace/{workspace_id}/chat_apps")
    assert response.status_code == 200
    response_data = response.json()
    assert "data" in response_data
    assert "items" in response_data["data"]
    assert isinstance(response_data["data"]["items"], list)


@pytest.mark.asyncio
async def test_update_chatapp_v2_success(
    test_client_admin_v2, workspace_id, chatapp_id_v2, agent_id_v2
):
    unique_suffix = random.randint(100, 999)
    update_data = {
        "name": f"Updated ChatApp V2 {unique_suffix}",
        "description": "Updated description via V2 API for testing",
        "agent_id": agent_id_v2,
        "voice_enabled": True,
    }
    response = await test_client_admin_v2.put(
        f"/workspace/{workspace_id}/chat_app/{chatapp_id_v2}", json=update_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "ChatApp updated successfully"
    assert response_data["data"]["name"] == update_data["name"]
    assert response_data["data"]["description"] == update_data["description"]


@pytest.mark.asyncio
async def test_update_chatapp_v2_not_found(
    test_client_admin_v2, workspace_id, agent_id_v2
):
    fake_chatapp_id = str(uuid4())
    update_data = {
        "name": "Updated Non-existent ChatApp",
        "description": "This should fail",
        "agent_id": agent_id_v2,
        "voice_enabled": True,
    }
    response = await test_client_admin_v2.put(
        f"/workspace/{workspace_id}/chat_app/{fake_chatapp_id}", json=update_data
    )
    assert response.status_code == 404
    response_data = response.json()
    assert (
        "not authorized" in response_data["detail"].lower()
        or "not found" in response_data["detail"].lower()
    )
