import random
import time
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_create_agent_success(test_client_admin, workspace_id, workspace_tool_id):
    agent_data = {
        "name": "Test Agent",
        "description": "Agent for testing",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": True,
        "agent_metadata": {"key": "value"},
        "workspace_tool_ids": [workspace_tool_id],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=agent_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Agent created successfully"
    assert response_data["data"]["name"] == agent_data["name"]
    assert response_data["data"]["description"] == agent_data["description"]
    assert (
        response_data["data"]["workspace_tool_ids"] == agent_data["workspace_tool_ids"]
    )


@pytest.mark.asyncio
async def test_create_agent_invalid_data(test_client_admin, workspace_id):
    agent_data = {"invalid_field": "invalid_value"}
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=agent_data
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_agent_by_id_success(test_client_admin, workspace_id, agent_id):
    response = await test_client_admin.get(
        f"/workspace/{workspace_id}/agent/{agent_id}"
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Agent retrieved successfully"
    assert response_data["data"]["id"] == agent_id


@pytest.mark.asyncio
async def test_get_agent_by_id_not_found(test_client_admin, workspace_id):
    non_existent_agent_id = str(uuid4())
    response = await test_client_admin.get(
        f"/workspace/{workspace_id}/agent/{non_existent_agent_id}"
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "Agent not found" in response_data["detail"]


@pytest.mark.asyncio
async def test_update_agent_success(
    test_client_admin, workspace_id, agent_id, workspace_tool_id
):
    unique_suffix = str(int(time.time()))
    updated_data = {
        "name": f"Updated Agent {unique_suffix}",
        "description": "Updated description",
        "prompt_instructions": "Updated instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.8,
        "system_prompt": "Updated system prompt",
        "memory_enabled": False,
        "agent_metadata": {"updated_key": "updated_value"},
        "workspace_tool_ids": [workspace_tool_id],
    }
    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/agent/{agent_id}", json=updated_data
    )
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Agent updated successfully"
    assert response_data["data"]["name"] == updated_data["name"]
    assert response_data["data"]["description"] == updated_data["description"]


@pytest.mark.asyncio
async def test_update_agent_not_found(test_client_admin, workspace_id):
    non_existent_agent_id = str(uuid4())
    updated_data = {
        "name": "Non-existent Agent",
        "description": "This agent does not exist",
    }
    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/agent/{non_existent_agent_id}", json=updated_data
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "Agent not found" in response_data["detail"]


@pytest.mark.asyncio
async def test_delete_agent_success(test_client_admin, workspace_id, agent_id):
    response = await test_client_admin.delete(
        f"/workspace/{workspace_id}/agent/{agent_id}"
    )
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_agent_not_found(test_client_admin, workspace_id):
    non_existent_agent_id = str(uuid4())
    response = await test_client_admin.delete(
        f"/workspace/{workspace_id}/agent/{non_existent_agent_id}"
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "Unable to find the WorkspaceAgent" in response_data["detail"]


@pytest.mark.asyncio
async def test_list_agents_success(test_client_admin, workspace_id, workspace_tool_id):
    agent_data_1 = {
        "name": "Agent 1",
        "description": "First agent",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": True,
        "agent_metadata": {"key": "value"},
        "workspace_tool_ids": [workspace_tool_id],
    }
    agent_data_2 = {
        "name": "Agent 2",
        "description": "Second agent",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": False,
        "agent_metadata": {"key": "value2"},
        "workspace_tool_ids": [workspace_tool_id],
    }

    await test_client_admin.post(f"/workspace/{workspace_id}/agent", json=agent_data_1)
    await test_client_admin.post(f"/workspace/{workspace_id}/agent", json=agent_data_2)

    response = await test_client_admin.get(f"/workspace/{workspace_id}/agent")
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "Agents retrieved successfully."
    assert len(response_data["data"]["items"]) >= 2
    assert any(agent["name"] == "Agent 1" for agent in response_data["data"]["items"])
    assert any(agent["name"] == "Agent 2" for agent in response_data["data"]["items"])


@pytest.mark.asyncio
async def test_create_agent_invalid_workspace(test_client_admin, workspace_tool_id):
    invalid_workspace_id = str(uuid4())
    agent_data = {
        "name": "Test Agent Invalid Workspace",
        "description": "Agent for testing invalid workspace",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": True,
        "agent_metadata": {"key": "value"},
        "workspace_tool_ids": [workspace_tool_id],
    }
    response = await test_client_admin.post(
        f"/workspace/{invalid_workspace_id}/agent", json=agent_data
    )
    assert response.status_code == 404
    response_data = response.json()
    assert "not authorized" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_create_agent_duplicate_name(
    test_client_admin, workspace_id, workspace_tool_id
):
    duplicate_name = f"Duplicate Agent {int(time.time())}"

    agent_data = {
        "name": duplicate_name,
        "description": "First agent with this name",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": True,
        "agent_metadata": {"key": "value"},
        "workspace_tool_ids": [workspace_tool_id],
    }

    # First creation should succeed
    response1 = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=agent_data
    )
    assert response1.status_code == 200

    # Second creation with same name should fail
    response2 = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=agent_data
    )
    assert response2.status_code == 400
    response_data = response2.json()
    assert "already exists" in response_data["detail"]


@pytest.mark.asyncio
async def test_update_agent_invalid_data(test_client_admin, workspace_id, agent_id):
    invalid_data = {
        "invalid_field": "invalid_value",
        "temperature": "invalid_temperature",  # Should be float, not string
    }
    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/agent/{agent_id}", json=invalid_data
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_agent_duplicate_name(
    test_client_admin, workspace_id, agent_id, workspace_tool_id
):
    existing_name = f"Existing Agent {int(time.time())}"

    # First create another agent with a specific name
    existing_agent_data = {
        "name": existing_name,
        "description": "Existing agent",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": True,
        "agent_metadata": {"key": "value"},
        "workspace_tool_ids": [workspace_tool_id],
    }

    response1 = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=existing_agent_data
    )
    assert response1.status_code == 200

    # Now try to update the original agent to have the same name
    updated_data = {
        "name": existing_name,
        "description": "Updated description - should fail",
        "prompt_instructions": "Updated instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.8,
        "system_prompt": "Updated system prompt",
        "memory_enabled": False,
        "agent_metadata": {"updated_key": "updated_value"},
        "workspace_tool_ids": [workspace_tool_id],
    }

    response = await test_client_admin.put(
        f"/workspace/{workspace_id}/agent/{agent_id}", json=updated_data
    )
    assert response.status_code == 400
    response_data = response.json()
    assert "already exists" in response_data["detail"]
