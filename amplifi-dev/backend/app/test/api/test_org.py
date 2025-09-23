from uuid import uuid4

import pytest


@pytest.mark.skip("Uses R2R")
@pytest.mark.asyncio
async def test_create_organization(test_client_admin):
    org_data = {
        "name": "Test Organization",
        "description": "Organization for testing",
        "domain": "testorg.com",
    }
    response = await test_client_admin.post("/organization", json=org_data)
    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Test Organization"
    assert response.json()["message"] == "Organization created successfully"


@pytest.mark.asyncio
async def test_get_organizations(test_client_admin):
    response = await test_client_admin.get("/organization")
    assert response.status_code == 200
    assert "data" in response.json()
    data = response.json()["data"]
    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.asyncio
async def test_get_organization_by_id(test_client_admin, organization_id):
    response = await test_client_admin.get(f"/organization/{organization_id}")
    assert response.status_code == 200
    assert response.json()["data"]["id"] == organization_id


@pytest.mark.asyncio
async def test_update_organization(test_client_admin, organization_id):
    update_data = {
        "name": "Updated Organization Name",
        "description": "Updated description",
        "domain": "updatedorg.com",
    }
    response = await test_client_admin.put(
        f"/organization/{organization_id}", json=update_data
    )
    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Updated Organization Name"
    assert response.json()["data"]["description"] == "Updated description"
    assert response.json()["data"]["domain"] == "example.com"
