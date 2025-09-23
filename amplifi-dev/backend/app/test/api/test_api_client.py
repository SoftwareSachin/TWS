from datetime import datetime, timedelta

import pytest


@pytest.mark.asyncio
async def test_update_api_client_success(test_client_admin, organization_id):
    # Step 1: Create an API client to update
    create_data = {
        "name": "Test API Client",
        "description": "API client for update test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert create_response.status_code == 200
    api_client = create_response.json()["data"]
    api_client_id = api_client["id"]

    # Step 2: Update the API client
    update_data = {
        "name": "Updated API Client Name",
        "description": "Updated description",
        "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
    }
    update_response = await test_client_admin.put(
        f"/organization/{organization_id}/api_client/{api_client_id}",
        json=update_data,
    )
    assert update_response.status_code == 200
    response_data = update_response.json()["data"]
    assert response_data["name"] == "Updated API Client Name"
    assert response_data["description"] == "Updated description"
    assert "expires_at" in response_data
    assert update_response.json()["message"] == "API client updated successfully"


@pytest.mark.asyncio
async def test_create_api_client_success(test_client_admin, organization_id):
    create_data = {
        "name": "Test API Client Create",
        "description": "API client for create test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["name"] == "Test API Client Create"
    assert data["description"] == "API client for create test"
    assert "client_id" in data
    assert "client_secret" in data
    assert "expires_at" in data
    assert response.json()["message"] == "API client created successfully"
    return data["id"]


@pytest.mark.asyncio
async def test_delete_api_client_success(test_client_admin, organization_id):
    # Create a client to delete
    create_data = {
        "name": "Test API Client Delete",
        "description": "API client for delete test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert create_response.status_code == 200
    api_client_id = create_response.json()["data"]["id"]
    # Delete
    delete_response = await test_client_admin.delete(
        f"/organization/{organization_id}/api_client/{api_client_id}"
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "API client deleted successfully"
    # Verify not found after delete
    get_response = await test_client_admin.get(
        f"/organization/{organization_id}/api_client/{api_client_id}"
    )
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_get_api_clients_list(test_client_admin, organization_id):
    # Create a client to ensure at least one exists
    create_data = {
        "name": "Test API Client List",
        "description": "API client for list test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert create_response.status_code == 200
    api_client_id = create_response.json()["data"]["id"]
    # List
    list_response = await test_client_admin.get(
        f"/organization/{organization_id}/api_client?page=1&size=50"
    )
    assert list_response.status_code == 200
    data = list_response.json()["data"]
    assert "items" in data
    assert any(client["id"] == api_client_id for client in data["items"])


@pytest.mark.asyncio
async def test_regenerate_api_client_secret(test_client_admin, organization_id):
    # Create a client
    create_data = {
        "name": "Test API Client Regenerate",
        "description": "API client for regenerate secret test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert create_response.status_code == 200
    api_client_id = create_response.json()["data"]["id"]
    # Regenerate secret
    regen_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client/{api_client_id}/regenerate-secret"
    )
    assert regen_response.status_code == 200
    data = regen_response.json()["data"]
    assert "client_secret" in data
    assert (
        regen_response.json()["message"] == "API client secret regenerated successfully"
    )


# --- Error Scenarios ---


@pytest.mark.asyncio
async def test_update_api_client_invalid_expires_at(test_client_admin, organization_id):
    # Create a client
    create_data = {
        "name": "Test API Client Invalid Update",
        "description": "API client for invalid update test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    api_client_id = create_response.json()["data"]["id"]
    # Try to update with past expires_at
    update_data = {
        "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
    }
    response = await test_client_admin.put(
        f"/organization/{organization_id}/api_client/{api_client_id}",
        json=update_data,
    )
    assert response.status_code == 422
    assert "expires_at must be in the future" in response.text


@pytest.mark.asyncio
async def test_update_api_client_not_found(test_client_admin, organization_id):
    fake_id = "00000000-0000-0000-0000-000000000000"
    update_data = {"name": "Should Not Exist"}
    response = await test_client_admin.put(
        f"/organization/{organization_id}/api_client/{fake_id}",
        json=update_data,
    )
    assert response.status_code == 404 or response.status_code == 400
    assert "not found" in response.text.lower() or "not exist" in response.text.lower()


@pytest.mark.asyncio
async def test_delete_api_client_not_found(test_client_admin, organization_id):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await test_client_admin.delete(
        f"/organization/{organization_id}/api_client/{fake_id}"
    )
    assert response.status_code == 404
    assert "not found" in response.text.lower()


@pytest.mark.asyncio
async def test_regenerate_api_client_secret_not_found(
    test_client_admin, organization_id
):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client/{fake_id}/regenerate-secret"
    )
    assert response.status_code == 404
    assert "not found" in response.text.lower()


@pytest.mark.asyncio
async def test_create_api_client_missing_required_fields(
    test_client_admin, organization_id
):
    create_data = {"description": "Missing name"}  # name is required
    response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert response.status_code == 422
    assert "name" in response.text


@pytest.mark.asyncio
async def test_create_api_client_invalid_expires_at(test_client_admin, organization_id):
    create_data = {
        "name": "Test API Client Invalid Expires",
        "description": "API client with past expires_at",
        "expires_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
    }
    response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert response.status_code == 422
    assert "expires_at must be in the future" in response.text


@pytest.mark.asyncio
async def test_get_api_client_by_id_not_found(test_client_admin, organization_id):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = await test_client_admin.get(
        f"/organization/{organization_id}/api_client/{fake_id}"
    )
    assert response.status_code == 404
    assert "not found" in response.text.lower()


@pytest.mark.asyncio
async def test_get_api_client_by_id_success(test_client_admin, organization_id):
    # Create an API client
    create_data = {
        "name": "Test API Client Get By ID",
        "description": "API client for get by id test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert create_response.status_code == 200
    api_client = create_response.json()["data"]
    api_client_id = api_client["id"]
    # Fetch by ID
    get_response = await test_client_admin.get(
        f"/organization/{organization_id}/api_client/{api_client_id}"
    )
    assert get_response.status_code == 200
    data = get_response.json()["data"]
    assert data["id"] == api_client_id
    assert data["name"] == "Test API Client Get By ID"
    assert data["description"] == "API client for get by id test"
    assert "expires_at" in data
    assert get_response.json()["message"] == "API client retrieved successfully"


@pytest.mark.asyncio
async def test_get_api_client_token_success(test_client_admin, organization_id):
    # Create an API client
    create_data = {
        "name": "Test API Client Token",
        "description": "API client for token test",
        "expires_at": (datetime.utcnow() + timedelta(days=10)).isoformat(),
    }
    create_response = await test_client_admin.post(
        f"/organization/{organization_id}/api_client",
        json=create_data,
    )
    assert create_response.status_code == 200
    api_client = create_response.json()["data"]
    client_id = api_client["client_id"]
    client_secret = api_client["client_secret"]
    # Get token
    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
    }
    token_response = await test_client_admin.post(
        "/api_client/token",
        json=token_data,
    )
    assert token_response.status_code == 200
    data = token_response.json()["data"]
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "expires_in" in data
    assert data["client_id"] == client_id
    assert token_response.json()["message"] == "API client authenticated successfully"
