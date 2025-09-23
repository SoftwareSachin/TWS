import test.api.utils as utils
import uuid
from uuid import UUID

import pytest


@pytest.mark.skip("Uses R2R")
@pytest.mark.asyncio
async def test_add_workflow_into_organization(
    test_client_admin, organization_id, destination_id, dataset_id
):
    client = test_client_admin
    new_workflow_body = {
        "name": "test_workflow",
        "description": "workflow for testing",
        "is_active": True,
        "destination_id": destination_id,
        "dataset_id": dataset_id,
        "schedule_config": {
            "cron_expression": ""
        },  # TODO: test specific cron expressions
    }
    print("NEW UPLOAD BODY !! ----- --------------- \n\n")
    print(new_workflow_body)
    # Test initial state (no workflows)
    response = await client.get(f"/organization/{organization_id}/workflow")
    utils.is_empty_response(response)

    # Create a new workflow
    post_response = await client.post(
        f"/organization/{organization_id}/workflow",
        json=new_workflow_body,
    )
    assert post_response.status_code == 200
    post_data = post_response.json()
    assert post_data["message"] == "Data created correctly"
    assert "id" in post_data["data"]
    assert post_data["data"]["name"] == "test_workflow"
    assert post_data["data"]["description"] == "workflow for testing"
    assert post_data["data"]["is_active"] is True
    assert post_data["data"]["destination_id"] == destination_id
    assert post_data["data"]["dataset_id"] == dataset_id
    assert "dataset_name" in post_data["data"]
    assert "destination_name" in post_data["data"]

    workflow_id = post_data["data"]["id"]

    # Retrieve the created workflow
    response = await client.get(
        f"/organization/{organization_id}/workflow/{workflow_id}"
    )
    assert response.status_code == 200
    json = response.json()
    assert json["message"] == "Data got correctly"
    assert json["data"] == post_data["data"]

    # Delete the workflow
    response = await client.delete(
        f"/organization/{organization_id}/workflow/{workflow_id}"
    )
    assert response.status_code == 200
    json = response.json()
    assert json["message"] == "Data deleted correctly"
    assert json["data"]["id"] == workflow_id  # Ensure the correct workflow was deleted

    # Verify the workflow is deleted
    response = await client.get(
        f"/organization/{organization_id}/workflow/{workflow_id}"
    )
    assert response.status_code == 404


random_workflow_body = {
    "name": "test_workflow",
    "description": "workflow for testing",
    "is_active": True,
    "destination_id": str(uuid.uuid4()),
    "dataset_id": str(uuid.uuid4()),
    "schedule_config": {"cron_expression": "cron_expression str in schedule_config"},
}
random_uuid = uuid.uuid4()


@pytest.mark.skip(reason="Currently will InternalError, need to fix sometime")
@pytest.mark.asyncio
async def test_workflow_post_valid_nonexistent_uuid(test_client_admin):
    client = test_client_admin
    response = await client.post(
        f"/organization/{random_uuid}/workflow",
        json=random_workflow_body,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to create workflow"}


@pytest.mark.asyncio
async def test_workflow_get_valid_nonexistent_uuid(test_client_admin):
    client = test_client_admin
    response = await client.get(
        f"/organization/{random_uuid}/workflow",
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_workflow_get_invalid_uuid(test_client_admin):
    client = test_client_admin
    response = await client.get(
        f"/organization/not_valid_uuid/workflow",
    )
    utils.is_invalid_uuid_response(response)
