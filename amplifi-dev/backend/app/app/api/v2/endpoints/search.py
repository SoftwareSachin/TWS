import time
from typing import Dict, List, Tuple, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

# from celery.result import AsyncResult
from app.api.deps import (  # gpt4o_client_batch,; gpt_deployment,; azure_endpoint,; azure_api_version,; blob_sas_token_rag,; blob_url_rag,
    get_current_user,
    workspace_check,
)

# from phoenix.evals import llm_generate, OpenAIModel, RelevanceEvaluator
# from phoenix.evals import llm_generate, OpenAIModel, RelevanceEvaluator
from app.be_core.logger import logger  # # for logging
from app.crud.image_media_crud import CRUDImageMedia
from app.crud.search_crud import search_crud
from app.models import ImageMedia
from app.schemas.eval_schema import (
    AggregateSearchEvalResult,
    DatasetSearchEvalResult,
    GraphSearchEvalResult,
    IWorkspaceSearchEvalResponse,
    SearchMetrics,
)
from app.schemas.response_schema import IPostResponseBase, create_response
from app.schemas.role_schema import IRoleEnum
from app.schemas.search_schema import (
    ImageSearchResult,
    IWorkspacePerformSearchRequest,
    R2RVectorSearchResult,
)
from app.schemas.user_schema import UserData
from app.utils.azure_fns.get_scores import get_precision
from app.utils.openai_utils import generate_embedding_async

router = APIRouter()
crudImage = CRUDImageMedia(ImageMedia)


async def search_files(
    query: str, dataset_ids: List[UUID], top_k: int, file_ids: List[UUID] | None = None
) -> List[ImageSearchResult]:
    logger.debug("Starting image search...")
    start_time = time.time()

    # Step 1: Get the query embedding
    query_embedding = await generate_embedding_async(query)

    # Step 2: Perform HNSW vector search using pgvector on chunks
    all_matches = await search_crud.vector_search(
        query_embedding=query_embedding,
        top_k=top_k,
        dataset_ids=dataset_ids,
        file_ids=file_ids,
    )

    logger.debug(f"Search completed in {time.time() - start_time:.2f}s")
    return all_matches


async def compute_metrics(
    query: str,
    results: List[Union[R2RVectorSearchResult, ImageSearchResult]],
    latency: float,
) -> Tuple[SearchMetrics, int]:
    precision, ndcg_score, in_tokens = get_precision(
        contexts=[r.text for r in results],
        doc_scores=[r.search_score for r in results],
        query=query,
        k=len(results),
    )
    return (
        SearchMetrics(
            precision=precision,
            ndcg_score=ndcg_score,
            latency=latency,
        ),
        in_tokens,
    )


async def build_dataset_results(
    query: str,
    dataset_ids: List[str],
    all_vector_results: List[Union[R2RVectorSearchResult, ImageSearchResult]],
    top_k: int,
    latency: float,
    calculate_metrics: bool,
) -> List[DatasetSearchEvalResult]:
    dataset_chunk_map: Dict[
        UUID, List[Union[R2RVectorSearchResult, ImageSearchResult]]
    ] = {UUID(ds): [] for ds in dataset_ids}
    for result in all_vector_results:
        dataset_chunk_map[result.dataset_id].append(result)

    dataset_results = []
    for idx, (ds_id, results) in enumerate(dataset_chunk_map.items()):
        results.sort(key=lambda x: x.search_score, reverse=True)
        results = results[:top_k]

        ds_result = DatasetSearchEvalResult(
            query=query,
            vector_search_results=results,
            precision_scores=None,
            gpt_input_tokens=-1,
            dataset_id=ds_id,
            idx=idx,
        )

        if calculate_metrics:
            metrics, in_tokens = await compute_metrics(query, results, latency)
            ds_result.precision_scores = metrics
            ds_result.gpt_input_tokens = in_tokens

        dataset_results.append(ds_result)

    return dataset_results


@router.post("/workspace/{workspace_id}/search")
async def search(
    workspace_id: UUID,
    request: IWorkspacePerformSearchRequest,
    current_user: UserData = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(workspace_check),
) -> IPostResponseBase[IWorkspaceSearchEvalResponse]:
    try:
        start_time = time.time()

        all_vector_results = await search_files(
            query=request.query,
            dataset_ids=request.dataset_ids,
            top_k=request.vector_search_settings.search_limit,
        )

        latency = time.time() - start_time
        aggregate_result = None
        dataset_results = None

        if request.perform_aggregate_search:
            results = sorted(
                all_vector_results, key=lambda x: x.search_score, reverse=True
            )[: request.vector_search_settings.search_limit]

            metrics, in_tokens = (
                await compute_metrics(request.query, results, latency)
                if request.calculate_metrics
                else (None, -1)
            )

            aggregate_result = AggregateSearchEvalResult(
                query=request.query,
                vector_search_results=results,
                precision_scores=metrics,
                gpt_input_tokens=in_tokens,
                dataset_ids=request.dataset_ids,
            )

        else:
            dataset_results = await build_dataset_results(
                query=request.query,
                dataset_ids=request.dataset_ids,
                all_vector_results=all_vector_results,
                top_k=request.vector_search_settings.search_limit,
                latency=latency,
                calculate_metrics=request.calculate_metrics,
            )

        response_data = IWorkspaceSearchEvalResponse(
            aggregate_results=aggregate_result,
            dataset_results=dataset_results,
            graph_results=(
                None
                if not request.perform_graph_search
                else GraphSearchEvalResult(query=request.query)
            ),
        )

        return create_response(
            message="Finished Query & Eval",
            meta={"dataset_ids": request.dataset_ids, "workspace_id": workspace_id},
            data=response_data,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
