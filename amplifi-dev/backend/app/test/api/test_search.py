import random

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.schemas.eval_schema import IWorkspaceSearchEvalResponse
from app.schemas.search_schema import IWorkspacePerformSearchRequest


def check_search_response(
    response,
    workspace_id,
    search_request: IWorkspacePerformSearchRequest,
) -> None:
    assert response.json()["meta"]["dataset_ids"] == search_request.dataset_ids
    assert response.json()["meta"]["workspace_id"] == workspace_id
    response_object = IWorkspaceSearchEvalResponse.model_validate(
        response.json()["data"]
    )
    response_aggregate = response_object.aggregate_results
    response_dataset = response_object.dataset_results
    if search_request.perform_aggregate_search:
        assert response_aggregate.query == search_request.query
        assert [  # Using UUID in response but str in request since celery can't serialize UUID
            str(id) for id in response_aggregate.dataset_ids
        ] == search_request.dataset_ids
        assert (
            len(response_aggregate.vector_search_results)
            == search_request.vector_search_settings.search_limit
        )

        assert response_dataset == None
    else:
        assert len(response_dataset) == len(search_request.dataset_ids)
        for result in response_dataset:
            assert result.query == search_request.query
            # Using UUID in response but str in request since celery can't serialize UUID
            assert str(result.dataset_id) in search_request.dataset_ids
            assert (
                len(result.vector_search_results)
                == search_request.vector_search_settings.search_limit
            )
        assert response_aggregate is None


@pytest.fixture(scope="function")
def search_request():
    return IWorkspacePerformSearchRequest(
        query="What is Plato known for?", dataset_ids=[]
    )


@pytest_asyncio.fixture(scope="module")
async def upload_1_file(dataset_id, ingest_dataset_by_id):
    await ingest_dataset_by_id(dataset_id)
    return dataset_id


@pytest.mark.skip(reason="Ingest error for same file multi dataset")
@pytest.mark.asyncio
async def test_search_three_datasets_aggregate(
    test_client_admin: AsyncClient,
    workspace_id,
    search_request,
    ingest_5_datasets,
):
    dataset_ids = random.sample(ingest_5_datasets, 3)
    search_request.dataset_ids = dataset_ids
    search_request.perform_aggregate_search = True
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/search",
        json=search_request.model_dump(),
    )
    response.raise_for_status()
    check_search_response(
        response=response,
        workspace_id=workspace_id,
        dataset_ids=dataset_ids,
        search_request=search_request,
    )


@pytest.mark.skip(reason="Ingest error for same file multi dataset")
@pytest.mark.asyncio
async def test_search_three_datasets_dataset_scoped(
    test_client_admin: AsyncClient,
    workspace_id,
    search_request,
    ingest_5_datasets,
):
    dataset_ids = random.sample(ingest_5_datasets, 3)
    search_request.dataset_ids = dataset_ids
    search_request.perform_dataset_search = True
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/search",
        json=search_request.model_dump(),
    )
    response.raise_for_status()
    check_search_response(
        response=response,
        workspace_id=workspace_id,
        dataset_ids=dataset_ids,
        search_request=search_request,
    )


@pytest.mark.skip(reason="Ingest error for same file multi dataset")
@pytest.mark.asyncio
async def test_search_three_datasets_both_scopes(
    test_client_admin: AsyncClient,
    workspace_id,
    search_request,
    ingest_5_datasets,
):
    dataset_ids = random.sample(ingest_5_datasets, 3)
    search_request.dataset_ids = dataset_ids
    search_request.perform_dataset_search = True
    search_request.perform_aggregate_search = True
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/search",
        json=search_request.model_dump(),
    )
    response.raise_for_status()
    check_search_response(
        response=response,
        workspace_id=workspace_id,
        dataset_ids=dataset_ids,
        search_request=search_request,
    )


@pytest.mark.skip("Uses R2R")
@pytest.mark.asyncio
async def test_search_one_dataset_aggregate(
    test_client_admin: AsyncClient, workspace_id, search_request, upload_1_file
):
    search_request.dataset_ids = [upload_1_file]
    search_request.perform_aggregate_search = True
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/search",
        json=search_request.model_dump(),
    )
    response.raise_for_status()
    check_search_response(
        response=response,
        workspace_id=workspace_id,
        search_request=search_request,
    )


@pytest.mark.skip("Uses R2R")
@pytest.mark.asyncio
async def test_search_one_dataset_dataset_scoped(
    test_client_admin: AsyncClient, workspace_id, search_request, upload_1_file
):
    search_request.dataset_ids = [upload_1_file]
    search_request.perform_aggregate_search = False
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/search",
        json=search_request.model_dump(),
    )
    response.raise_for_status()
    check_search_response(
        response=response,
        workspace_id=workspace_id,
        search_request=search_request,
    )


@pytest.mark.skip(reason="currently doesn't support both dataset and aggregate")
@pytest.mark.asyncio
async def test_search_one_dataset_both_scopes(
    test_client_admin: AsyncClient, workspace_id, search_request, upload_1_file
):
    search_request.dataset_ids = [upload_1_file]
    search_request.perform_dataset_search = True
    search_request.perform_aggregate_search = True
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/search",
        json=search_request.model_dump(),
    )
    response.raise_for_status()
    check_search_response(
        response=response,
        workspace_id=workspace_id,
        search_request=search_request,
    )
