import test.api.utils as utils
import uuid

import pytest

pg_vec = {
    "host": "database",
    "port": 5432,
    "database_name": "amplifi_db",
    "table_name": "sample_table",
    "username": "postgres",
    "password": "postgres",
}
new_dest_body = {
    "name": "test_destination",
    "description": "destination for testing",
    "is_active": True,
    "pg_vector": pg_vec,
}


@pytest.mark.asyncio
async def test_destination(test_client_admin, organization_id):
    client = test_client_admin
    # Finish setup
    post_response = await client.post(
        f"/organization/{organization_id}/destination",
        json=new_dest_body,
    )
    assert post_response.status_code == 200
    post_data = post_response.json()
    assert post_data["message"] == "Destination created successfully"
    assert "id" in post_data["data"]
    assert post_data["data"]["name"] == "test_destination"
    assert post_data["data"]["description"] == "destination for testing"
    assert post_data["data"]["is_active"] is True
    dest_id = post_data["data"]["id"]

    response = await client.get(
        f"/organization/{organization_id}/destination/{dest_id}",
    )
    assert response.status_code == 200
    json = response.json()
    assert json["message"] == "Data got correctly"
    assert json["data"]["name"] == "test_destination"
    assert json["data"]["description"] == "destination for testing"
    assert json["data"]["is_active"] is True

    response = await client.delete(
        f"/organization/{organization_id}/destination/{dest_id}",
    )
    assert response.status_code == 204
    # json = response.json()
    # assert json["message"] == "Destination deleted successfully"
    # assert json["data"]["name"] == "test_destination"
    # assert json["data"]["description"] == "destination for testing"
    # assert json["data"]["is_active"] is True

    response = await client.get(
        f"/organization/{organization_id}/destination/{dest_id}",
    )
    assert response.status_code == 404


random_uuid = uuid.uuid4()


@pytest.mark.asyncio
async def test_destination_post_valid_nonexistent_uuid(test_client_admin):
    client = test_client_admin
    response = await client.post(
        f"/organization/{random_uuid}/destination",
        json=new_dest_body,
    )
    # TODO Probably should return some other error if org not found?
    assert response.status_code == 403
    # assert response.json() == {"detail": "Failed to create destination"}


@pytest.mark.asyncio
async def test_destination_get_valid_nonexistent_uuid(test_client_admin):
    client = test_client_admin
    response = await client.get(
        f"/organization/{random_uuid}/destination",
    )
    assert response.status_code == 403
    # TODO Probably shouldn't just return nothing if orgID doesn't exist?
    # utils.is_empty_response(response)


@pytest.mark.asyncio
async def test_destination_get_invalid_uuid(test_client_admin):
    client = test_client_admin

    response = await client.get(
        f"/organization/not_valid_uuid/destination",
    )
    utils.is_invalid_uuid_response(response)
