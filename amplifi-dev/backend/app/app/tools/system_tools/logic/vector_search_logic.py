from typing import Dict, List

from app.api.v2.endpoints.chatsession import aggregate_file_contexts_v2
from app.api.v2.endpoints.search import search_files
from app.be_core.config import settings
from app.be_core.logger import logger
from app.schemas.rag_generation_schema import IFileContextAggregation
from app.schemas.search_schema import ImageSearchResult
from app.tools.system_tools.schemas.vector_search_schema import (
    RagContext,
    VectorSearchInput,
    VectorSearchOutput,
    VectorSearchResult,
)
from app.utils.image_processing_utils import encode_image


async def perform_vector_search(input: VectorSearchInput) -> VectorSearchOutput:
    logger.info("Performing vector search with enriched context")
    logger.info(f"Vectorsearch input: {input}")

    base64_cache: Dict[str, str] = {}
    search_results: List[ImageSearchResult] = await search_files(
        query=input.query,
        dataset_ids=input.dataset_ids,
        top_k=settings.VECTOR_SEARCH_TOOL_TOP_K,
        file_ids=input.file_ids,
    )
    logger.info(f"Vectorsearch raw results: {search_results}")

    contexts_found: List[RagContext] = []
    results = []
    for result in search_results:
        file_path = result.file_path
        is_image = result.mimetype and result.mimetype.startswith("image/")

        if result.chunk_metadata is None:
            continue  # Skip incomplete
        metadata = result.chunk_metadata
        if metadata and metadata.chunk_id:
            results.append(
                VectorSearchResult(
                    chunk_id=str(metadata.chunk_id),
                    text=result.text or "",
                    source=str(getattr(result, "dataset_id", "")),
                )
            )
        if is_image and file_path not in base64_cache:
            try:
                base64_cache[file_path] = await encode_image(file_path)
                result.chunk_metadata.base64 = base64_cache[file_path]
            except Exception as e:
                logger.warning(f"Failed to encode image {file_path}: {e}")
                result.chunk_metadata.base64 = None
        else:
            result.chunk_metadata.base64 = None

        contexts_found.append(
            RagContext(
                text=result.text or "",
                file={
                    "filename": file_path.split("/")[-1],
                    "filepath": str(file_path),
                    "mime_type": result.mimetype or "",
                    "file_id": str(result.file_id),
                    "dataset_id": str(result.dataset_id),
                },
                download_url="",
                chunk_id=result.chunk_metadata.chunk_id,
            )
        )

    # Sort and filter
    search_results.sort(key=lambda x: x.search_score or 0, reverse=True)
    top_results = search_results[: settings.VECTOR_SEARCH_TOOL_TOP_K]

    # Filter contexts by search score threshold
    filtered_top_results = [
        result
        for result in top_results
        if (result.search_score or 0)
        >= settings.CONTEXT_RETRIEVED_SEARCH_SCORE_THRESHOLD
    ]
    filtered_file_ids = {str(result.file_id) for result in filtered_top_results}
    # Filter contexts by pre-filtered file_ids
    filtered_contexts = [
        context
        for context in contexts_found
        if context.file.get("file_id", "") in filtered_file_ids
    ]

    aggregated_file_contexts: List[IFileContextAggregation] = (
        await aggregate_file_contexts_v2(filtered_contexts, filtered_top_results)
    )

    return VectorSearchOutput(
        results=results,
        # rag_contexts=filtered_contexts,
        # raw_results=top_results,
        aggregated=aggregated_file_contexts,
    )
