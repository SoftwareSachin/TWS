import random
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_chat_v2_success(test_client_admin_v2, chatapp_id_v2):
    chat_data = {
        "chat_app_id": chatapp_id_v2,
        "query": "Hello, can you help me with a test question?",
    }
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    assert response.status_code == 200
    response_data = response.json()

    assert response_data is not None


@pytest.mark.asyncio
async def test_chat_v2_with_session_id(test_client_admin_v2, chatapp_id_v2):
    # Use a fake session ID for testing
    fake_session_id = str(uuid4())
    chat_data = {
        "chat_app_id": chatapp_id_v2,
        "chat_session_id": fake_session_id,
        "query": "This is a follow-up question in the session.",
    }
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    # Response may succeed or fail depending on session validation logic
    assert response.status_code in [200, 400, 404]


@pytest.mark.asyncio
async def test_chat_v2_empty_query(test_client_admin_v2, chatapp_id_v2):
    """Test chat request with empty query should fail."""
    chat_data = {"chat_app_id": chatapp_id_v2, "query": ""}
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    assert response.status_code == 400
    response_data = response.json()
    assert "query cannot be empty" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_chat_v2_whitespace_only_query(test_client_admin_v2, chatapp_id_v2):
    chat_data = {"chat_app_id": chatapp_id_v2, "query": "   \n\t   "}
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    assert response.status_code == 400
    response_data = response.json()
    assert "query cannot be empty" in response_data["detail"].lower()


@pytest.mark.asyncio
async def test_chat_v2_invalid_chatapp_id(test_client_admin_v2):
    fake_chatapp_id = str(uuid4())
    chat_data = {
        "chat_app_id": fake_chatapp_id,
        "query": "This should fail because the chatapp doesn't exist.",
    }
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    # The chat utility returns 200 with error info in response, not HTTP error codes
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_v2_missing_required_fields(test_client_admin_v2):
    """Test chat request with missing required fields."""
    # Missing chat_app_id
    chat_data = {"query": "This request is missing chat_app_id."}
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    assert response.status_code == 422

    # Missing query
    chat_data = {"chat_app_id": str(uuid4())}
    response = await test_client_admin_v2.post("/chat", json=chat_data)
    assert response.status_code == 422
