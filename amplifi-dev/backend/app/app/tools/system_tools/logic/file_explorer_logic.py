from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from rapidfuzz import fuzz, process
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.be_core.logger import logger
from app.db.session import SessionLocal
from app.models.dataset_file_link_model import DatasetFileLink
from app.models.dataset_model import Dataset
from app.models.document_model import Document, DocumentProcessingStatusEnum
from app.models.file_model import File
from app.tools.system_tools.schemas.file_explorer_schema import (
    FileExplorerInput,
    FileExplorerOutput,
    FileInfo,
)
from app.utils.openai_utils import generate_embedding_async

DEFAULT_FILE_LIMIT = 50
MAX_FILE_LIMIT = 200  # Maximum limit for file operations


class FileInfoType(Enum):
    """Types of file info building for different operations."""

    BASIC = "basic"
    DETAILED = "detailed"
    DESCRIPTION = "description"
    DATASET_AWARE = "dataset_aware"


async def perform_file_operation(input_data: FileExplorerInput) -> FileExplorerOutput:
    """
    Unified file explorer operation that returns comprehensive information in a single call.

    This function eliminates the need for multiple API calls by providing:
    - File listings with detailed metadata
    - Search results with fuzzy matching
    - Individual file descriptions and metadata
    - Dataset associations for all files
    - Comprehensive pagination information
    - All related context in one response

    Args:
        input_data: Validated input parameters

    Returns:
        FileExplorerOutput with comprehensive file information and metadata

    """
    db_session = None
    try:
        db_session = SessionLocal()
        # Convert UUID objects to strings for internal processing (database expects string UUIDs)
        dataset_id_strings = [str(uuid_obj) for uuid_obj in input_data.dataset_ids]

        # Execute comprehensive file operation
        result = await _execute_comprehensive_file_operation(
            db_session, input_data, dataset_id_strings
        )
        await db_session.commit()
        return result

    except Exception as e:
        logger.error(f"File explorer operation failed: {str(e)}")
        if db_session:
            try:
                await db_session.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")

        return FileExplorerOutput(
            current_path="error",
            operation_result=f"Unexpected error in file operation: {str(e)}",
            items=[],
            total_items=0,
            error=f"Unexpected error in file operation: {str(e)}",
        )
    finally:
        if db_session:
            await db_session.close()


async def _execute_comprehensive_file_operation(
    db: AsyncSession, input_data: FileExplorerInput, dataset_ids: Optional[List[str]]
) -> FileExplorerOutput:
    """
    Execute comprehensive file operation that returns all relevant information in one call.

    This unified function handles all operation types and returns comprehensive data including:
    - File listings with complete metadata
    - Search results with fuzzy matching when needed
    - Individual file detailed information when specific file requested
    - Dataset associations for all files
    - Related files and suggestions
    - Comprehensive context information
    """
    try:
        # Calculate pagination parameters
        limit = min(input_data.limit or DEFAULT_FILE_LIMIT, MAX_FILE_LIMIT)
        offset = ((input_data.page or 1) - 1) * limit

        # Initialize result variables
        files = []
        total_count = 0
        search_results = []
        used_fuzzy_matching = False
        operation_type = input_data.operation

        # Handle specific file operations (get_metadata, get_description, find_datasets_for_file, find_file_by_name)
        if (
            operation_type
            in [
                "get_metadata",
                "get_description",
                "find_datasets_for_file",
                "find_file_by_name",
            ]
            and input_data.name
        ):
            return await _handle_specific_file_comprehensive(
                db, input_data, dataset_ids
            )

        # Handle content search operations (vector search on document descriptions)
        elif operation_type == "content_search":
            files, total_count, search_results, used_fuzzy_matching = (
                await _execute_content_based_search(
                    db, input_data, dataset_ids, limit, offset
                )
            )

        # Handle search operations
        elif operation_type == "search" or input_data.search_pattern:
            files, total_count, search_results, used_fuzzy_matching = (
                await _execute_comprehensive_search(
                    db, input_data, dataset_ids, limit, offset
                )
            )

        # Handle list operations or fallback
        else:
            files, total_count = await _query_files(
                db=db,
                dataset_ids=dataset_ids,
                filename_filter=None,
                limit=limit,
                offset=offset,
            )

        # If no files found, provide helpful information
        if not files:
            context_desc = "datasets"
            search_context = (
                f" matching '{input_data.search_pattern}'"
                if input_data.search_pattern
                else ""
            )

            return FileExplorerOutput(
                current_path=_determine_context_path(input_data, dataset_ids),
                operation_result=f"No files found{search_context} in {context_desc}",
                items=[],
                total_items=0,
                total_size=None,
                search_results=[f"No results found{search_context} in {context_desc}"],
                dataset_name=(
                    await _get_primary_dataset_name(db, dataset_ids)
                    if dataset_ids
                    else None
                ),
                datasets_involved=await _get_datasets_involved(db, dataset_ids, []),
                operation_type=operation_type,
                fuzzy_matching_used=False,
                file_types_summary={},
                context_hints=_generate_context_hints(input_data, 0, 0, False),
                pagination=_build_pagination_info(input_data, 0, limit, offset),
                has_more=False,
                next_page=None,
            )

        # Build comprehensive file information with DETAILED metadata for all files
        file_infos = await _build_comprehensive_file_info_collection(
            db, files, dataset_ids
        )

        # Calculate comprehensive statistics and context
        total_size = _calculate_total_size(files)
        context_path = _determine_context_path(input_data, dataset_ids)

        # Generate file type statistics
        file_types_summary = _generate_file_types_summary(files)

        # Get all datasets involved in the operation
        datasets_involved = await _get_datasets_involved(db, dataset_ids, files)

        # Generate contextual hints
        context_hints = _generate_context_hints(
            input_data, len(files), total_count, used_fuzzy_matching
        )

        # Build pagination information
        current_page = input_data.page or 1
        _total_pages = (
            (total_count + limit - 1) // limit if not used_fuzzy_matching else 1
        )
        has_more = (offset + len(files)) < total_count and not used_fuzzy_matching
        next_page = current_page + 1 if has_more else None

        pagination_info = _build_pagination_info(
            input_data, total_count, limit, offset, used_fuzzy_matching
        )

        # Build comprehensive operation result message
        operation_result = _build_comprehensive_operation_result(
            input_data,
            len(file_infos),
            total_count,
            offset,
            has_more,
            next_page,
            used_fuzzy_matching,
        )

        # Generate default search results if not already set
        if not search_results and input_data.search_pattern:
            search_results = _build_exact_search_results(
                input_data.search_pattern, file_infos
            )

        # Get related context information
        dataset_name = (
            await _get_primary_dataset_name(db, dataset_ids) if dataset_ids else None
        )

        return FileExplorerOutput(
            current_path=context_path,
            operation_result=operation_result,
            items=file_infos,
            total_items=total_count,
            total_size=total_size,
            search_results=search_results if search_results else None,
            dataset_name=dataset_name,
            datasets_involved=datasets_involved,
            operation_type=operation_type,
            fuzzy_matching_used=used_fuzzy_matching,
            file_types_summary=file_types_summary,
            context_hints=context_hints,
            pagination=pagination_info,
            has_more=has_more,
            next_page=next_page,
        )

    except Exception as e:
        logger.error(f"Comprehensive file operation failed: {str(e)}")
        # Let the main function handle transaction rollback

        return FileExplorerOutput(
            current_path="error",
            operation_result=f"Comprehensive file operation failed: {str(e)}",
            items=[],
            total_items=0,
            error=f"Comprehensive file operation failed: {str(e)}",
        )


async def _handle_specific_file_comprehensive(
    db: AsyncSession, input_data: FileExplorerInput, dataset_ids: Optional[List[str]]
) -> FileExplorerOutput:
    """Handle specific file operations with comprehensive information."""
    if not input_data.name:
        return FileExplorerOutput(
            current_path="error",
            operation_result=f"'name' field is required for {input_data.operation} operation",
            items=[],
            total_items=0,
            error=f"'name' field is required for {input_data.operation} operation",
        )

    # Find the specific file
    file = await _find_file_by_scope(db, input_data.name, dataset_ids)

    if not file:
        context_desc = "datasets" if dataset_ids else "workspace"
        return FileExplorerOutput(
            current_path=_determine_context_path(input_data, dataset_ids),
            operation_result=f"File '{input_data.name}' not found in {context_desc}",
            items=[],
            total_items=0,
            total_size=None,
            search_results=[
                f"No file named '{input_data.name}' found in {context_desc}"
            ],
            dataset_name=(
                await _get_primary_dataset_name(db, dataset_ids)
                if dataset_ids
                else None
            ),
            datasets_involved=await _get_datasets_involved(db, dataset_ids, []),
            operation_type=input_data.operation,
            fuzzy_matching_used=False,
            file_types_summary={},
            context_hints=[
                f"💡 File '{input_data.name}' not found - check spelling or try fuzzy search"
            ],
            pagination=_build_pagination_info(input_data, 0, 50, 0),
            has_more=False,
            next_page=None,
        )

    # For find_datasets_for_file operation, get all datasets containing the file
    if input_data.operation == "find_datasets_for_file":
        containing_datasets = await _get_file_containing_datasets(db, str(file.id))
        file_info = await _build_single_file_info_optimized(
            db, file, FileInfoType.DATASET_AWARE, dataset_ids, containing_datasets
        )

        return FileExplorerOutput(
            current_path=_determine_context_path(input_data, dataset_ids),
            operation_result=f"Found file '{file.filename}' in {len(containing_datasets)} dataset(s)",
            items=[file_info],
            total_items=1,
            total_size=_format_file_size(file.size) if file.size else None,
            search_results=[
                f"File '{file.filename}' found in datasets: {', '.join([d['name'] for d in containing_datasets])}"
            ],
            dataset_name=(
                await _get_primary_dataset_name(db, dataset_ids)
                if dataset_ids
                else None
            ),
            datasets_involved=[d["name"] for d in containing_datasets],
            operation_type=input_data.operation,
            fuzzy_matching_used=file.filename != input_data.name,
            file_types_summary=(
                {Path(file.filename).suffix.lower(): 1} if file.filename else {}
            ),
            context_hints=[f"📂 File exists in {len(containing_datasets)} dataset(s)"],
            pagination=_build_pagination_info(input_data, 1, 50, 0),
            has_more=False,
            next_page=None,
        )

    # For find_file_by_name operation, return file info with dataset associations
    if input_data.operation == "find_file_by_name":
        dataset_names = await _get_file_dataset_names(db, str(file.id))
        file_info = await _build_single_file_info_optimized(
            db, file, FileInfoType.BASIC, dataset_ids
        )

        return FileExplorerOutput(
            current_path=_determine_context_path(input_data, dataset_ids),
            operation_result=f"Located file '{file.filename}' successfully",
            items=[file_info],
            total_items=1,
            total_size=_format_file_size(file.size) if file.size else None,
            search_results=[
                f"File '{file.filename}' located in datasets: {', '.join(dataset_names)}"
            ],
            dataset_name=(
                await _get_primary_dataset_name(db, dataset_ids)
                if dataset_ids
                else None
            ),
            datasets_involved=dataset_names,
            operation_type=input_data.operation,
            fuzzy_matching_used=file.filename != input_data.name,
            file_types_summary=(
                {Path(file.filename).suffix.lower(): 1} if file.filename else {}
            ),
            context_hints=[
                f"✅ File '{file.filename}' found in {len(dataset_names)} dataset(s)"
            ],
            pagination=_build_pagination_info(input_data, 1, 50, 0),
            has_more=False,
            next_page=None,
        )

    # For other operations, build comprehensive file info
    info_type = (
        FileInfoType.DETAILED
        if input_data.operation == "get_metadata"
        else FileInfoType.DESCRIPTION
    )
    file_info = await _build_single_file_info_optimized(
        db, file, info_type, dataset_ids
    )

    operation_label = (
        "metadata" if input_data.operation == "get_metadata" else "AI content analysis"
    )

    return FileExplorerOutput(
        current_path=_determine_context_path(input_data, dataset_ids),
        operation_result=f"Retrieved {operation_label} for file '{file.filename}'",
        items=[file_info],
        total_items=1,
        total_size=_format_file_size(file.size) if file.size else None,
        search_results=None,
        dataset_name=(
            await _get_primary_dataset_name(db, dataset_ids) if dataset_ids else None
        ),
        datasets_involved=await _get_datasets_involved(db, dataset_ids, [file]),
        operation_type=input_data.operation,
        fuzzy_matching_used=file.filename != input_data.name,
        file_types_summary=(
            {Path(file.filename).suffix.lower(): 1} if file.filename else {}
        ),
        context_hints=[f"📄 Retrieved {operation_label} for '{file.filename}'"],
        pagination=_build_pagination_info(input_data, 1, 50, 0),
        has_more=False,
        next_page=None,
    )


async def _execute_content_based_search(
    db: AsyncSession,
    input_data: FileExplorerInput,
    dataset_ids: Optional[List[str]],
    limit: int,
    offset: int,
) -> Tuple[List[File], int, List[str], bool]:
    """Execute hybrid content-based search using both pattern matching and semantic similarity."""
    content_query = input_data.content_query
    if not content_query:
        logger.error("Content query is required for content_search operation")
        return [], 0, [], False

    try:
        logger.info(
            f"Performing hybrid content-based file search for: '{content_query}'"
        )

        # APPROACH 1: Pattern-based search using existing search operation
        pattern_files = []
        pattern_search_results = []

        # Extract keywords from the query for pattern matching
        keywords = _extract_keywords_for_pattern_search(content_query)
        logger.info(f"Extracted keywords for pattern search: {keywords}")

        for keyword in keywords:
            try:
                # Use the existing comprehensive search operation for each keyword
                search_files, _, search_results, used_fuzzy = (
                    await _execute_comprehensive_search(
                        db=db,
                        input_data=FileExplorerInput(
                            operation="search",
                            search_pattern=f"*{keyword}*",
                            dataset_ids=input_data.dataset_ids,
                            limit=limit,
                            page=1,
                        ),
                        dataset_ids=dataset_ids,
                        limit=limit,
                        offset=0,
                    )
                )
                # Convert File objects to a list and extend our results
                pattern_files.extend(search_files)
                if search_files:
                    pattern_search_results.append(
                        f"Found {len(search_files)} files matching keyword '{keyword}'"
                    )
            except Exception as pattern_error:
                logger.warning(
                    f"Pattern search for keyword '{keyword}' failed: {pattern_error}"
                )

        # APPROACH 2: Semantic similarity search using document descriptions
        similarity_files = []
        similarity_search_results = []

        try:
            # Generate embedding for the content query
            query_embedding = await generate_embedding_async(content_query)

            # Perform vector search on document descriptions
            sim_files, _ = await _vector_search_documents(
                db=db,
                query_embedding=query_embedding,
                dataset_ids=dataset_ids,
                limit=limit,
                offset=0,
            )
            similarity_files.extend(sim_files)
            if sim_files:
                similarity_search_results.append(
                    f"Found {len(sim_files)} files via semantic similarity"
                )
        except Exception as similarity_error:
            logger.warning(f"Semantic similarity search failed: {similarity_error}")

        # COMBINE RESULTS: Merge files from both approaches and deduplicate
        all_files_dict = {}

        # Add pattern files (use file.id as key for deduplication)
        for file in pattern_files:
            all_files_dict[file.id] = file

        # Add similarity files (use file.id as key for deduplication)
        for file in similarity_files:
            all_files_dict[file.id] = file

        all_files = list(all_files_dict.values())
        total_count = len(all_files)

        # Apply pagination to combined results
        paginated_files = all_files[offset : offset + limit]

        # Build comprehensive search results
        search_results = [
            f"Hybrid search found {total_count} unique files for '{content_query}'"
        ]

        if pattern_search_results:
            search_results.extend(
                [f"📁 Pattern matches: {', '.join(pattern_search_results)}"]
            )

        if similarity_search_results:
            search_results.extend(
                [f"🧠 Semantic matches: {', '.join(similarity_search_results)}"]
            )

        if paginated_files:
            search_results.extend(
                [
                    "Top results:",
                    *[f"- {file.filename}" for file in paginated_files[:5]],
                ]
            )
            if len(paginated_files) > 5:
                search_results.append(f"... and {len(paginated_files) - 5} more files")

        logger.info(
            f"Hybrid search: {len(pattern_files)} pattern matches, {len(similarity_files)} semantic matches, {total_count} total unique files"
        )
        return paginated_files, total_count, search_results, False

    except Exception as e:
        logger.error(f"Hybrid content-based search failed: {str(e)}")
        return [], 0, [f"Hybrid content-based search failed: {str(e)}"], False


async def _execute_comprehensive_search(
    db: AsyncSession,
    input_data: FileExplorerInput,
    dataset_ids: Optional[List[str]],
    limit: int,
    offset: int,
) -> Tuple[List[File], int, List[str], bool]:
    """Execute comprehensive search with both exact and fuzzy matching."""
    search_pattern = input_data.search_pattern
    if not search_pattern:
        logger.error("Search pattern is required for search operation")
        return [], 0, [], False

    # First try exact pattern matching
    files, total_count = await _query_files(
        db=db,
        dataset_ids=dataset_ids,
        filename_filter=search_pattern,
        limit=limit,
        offset=offset,
    )

    search_results = []
    used_fuzzy_matching = False

    # If no exact matches, try fuzzy matching
    if not files:
        logger.info(
            f"No exact matches for '{search_pattern}', trying fuzzy matching..."
        )

        try:
            fuzzy_matches = await _find_best_matching_files_optimized(
                db=db,
                query=search_pattern,
                dataset_ids=dataset_ids,
                min_score=0.4,
                max_results=limit,
            )

            if fuzzy_matches:
                files = [match[0] for match in fuzzy_matches]
                total_count = len(files)
                used_fuzzy_matching = True
                search_results = _build_fuzzy_matching_results(
                    search_pattern, fuzzy_matches
                )

        except Exception as fuzzy_error:
            logger.error(f"Fuzzy matching failed: {str(fuzzy_error)}")

    # Build search results for exact matches
    if not search_results and files:
        search_results = _build_exact_search_results(
            search_pattern, []
        )  # Will be populated later

    return files, total_count, search_results, used_fuzzy_matching


async def _vector_search_documents(
    db: AsyncSession,
    query_embedding: List[float],
    dataset_ids: Optional[List[str]],
    limit: int,
    offset: int = 0,
) -> Tuple[List[File], int]:
    """
    Perform vector search on document descriptions to find semantically similar files.

    Args:
        db: Database session
        query_embedding: Query embedding vector
        dataset_ids: List of dataset IDs to search within
        limit: Maximum number of results to return
        offset: Number of results to skip

    Returns:
        Tuple of (files_list, total_count)
    """
    try:
        # Validate dataset_ids before proceeding
        if not dataset_ids:
            logger.warning("No dataset_ids provided for vector search")
            return [], 0

        # Convert dataset_ids to UUIDs with error handling
        valid_dataset_uuids = []
        for ds_id in dataset_ids:
            try:
                valid_dataset_uuids.append(UUID(ds_id))
            except (ValueError, TypeError) as uuid_error:
                logger.warning(
                    f"Invalid UUID format for dataset_id '{ds_id}': {uuid_error}"
                )
                continue

        if not valid_dataset_uuids:
            logger.warning("No valid dataset UUIDs found for vector search")
            return [], 0

        # First, let's check if there are any documents at all in this dataset
        total_docs_query = (
            select(func.count())
            .select_from(Document)
            .join(DatasetFileLink, Document.file_id == DatasetFileLink.file_id)
            .where(
                DatasetFileLink.dataset_id.in_(valid_dataset_uuids),
                Document.deleted_at.is_(None),
            )
        )

        # Then check documents with description embeddings
        debug_query = (
            select(func.count())
            .select_from(Document)
            .join(DatasetFileLink, Document.file_id == DatasetFileLink.file_id)
            .where(
                DatasetFileLink.dataset_id.in_(valid_dataset_uuids),
                Document.description_embedding.is_not(None),
                Document.processing_status == DocumentProcessingStatusEnum.Success,
                Document.deleted_at.is_(None),
            )
        )

        try:
            total_docs_result = await db.execute(total_docs_query)
            total_docs = total_docs_result.scalar() or 0

            debug_result = await db.execute(debug_query)
            docs_with_embeddings = debug_result.scalar() or 0

            logger.info(
                f"Dataset debug: {total_docs} total documents, {docs_with_embeddings} with description embeddings"
            )
        except Exception as debug_error:
            logger.error(f"Error checking for documents with embeddings: {debug_error}")

        # Build the vector search query using document description embeddings
        query = (
            select(
                File,
                Document,
                Document.description_embedding.op("<->")(query_embedding).label(
                    "similarity_score"
                ),
            )
            .join(Document, File.id == Document.file_id)
            .where(
                Document.description_embedding.is_not(None),
                Document.processing_status == DocumentProcessingStatusEnum.Success,
                Document.deleted_at.is_(None),
                File.deleted_at.is_(None),
            )
        )

        # Apply dataset filtering with validated UUIDs
        query = query.join(DatasetFileLink, File.id == DatasetFileLink.file_id).where(
            DatasetFileLink.dataset_id.in_(valid_dataset_uuids)
        )

        # Order by similarity (lower distance = higher similarity)
        # Note: We'll reorder later for DISTINCT ON compatibility

        # Execute simplified count query with error handling
        try:
            # Create a simpler count query without the complex subquery
            count_query = (
                select(func.count(func.distinct(File.id)))
                .select_from(File)
                .join(Document, File.id == Document.file_id)
                .join(DatasetFileLink, File.id == DatasetFileLink.file_id)
                .where(
                    Document.description_embedding.is_not(None),
                    Document.processing_status == DocumentProcessingStatusEnum.Success,
                    Document.deleted_at.is_(None),
                    File.deleted_at.is_(None),
                    DatasetFileLink.dataset_id.in_(valid_dataset_uuids),
                )
            )
            count_result = await db.execute(count_query)
            total_count = count_result.scalar() or 0
        except Exception as count_error:
            logger.error(f"Error getting count for vector search: {count_error}")
            total_count = 0
            # Don't rollback here - let the main function handle transaction management

        # Apply pagination with proper DISTINCT ON ordering
        query = (
            query.distinct(File.id)
            .order_by(File.id, "similarity_score")
            .limit(limit)
            .offset(offset)
        )

        # Execute main query with error handling
        try:
            result = await db.execute(query)
            rows = result.fetchall()
        except Exception as query_error:
            logger.error(f"Error executing vector search query: {query_error}")
            # Don't rollback here - let the main function handle transaction management
            return [], total_count

        # Extract unique files (in case a file has multiple matching documents)
        seen_files = set()
        files = []
        for row in rows:
            try:
                # Row structure: (File, Document, similarity_score)
                file = row.File if hasattr(row, "File") else row[0]  # File object
                if file and file.id not in seen_files:
                    files.append(file)
                    seen_files.add(file.id)
            except Exception as row_error:
                logger.warning(f"Error processing search result row: {row_error}")
                continue

        logger.info(
            f"Vector search found {len(files)} unique files from {len(rows)} document matches"
        )
        return files, total_count

    except Exception as e:
        logger.error(f"Vector search on documents failed: {str(e)}")
        # Let the main function handle transaction rollback
        return [], 0


async def _build_comprehensive_file_info_collection(
    db: AsyncSession,
    files: List[File],
    dataset_ids: Optional[List[str]] = None,
) -> List[FileInfo]:
    """
    Build comprehensive FileInfo collection with ALL metadata for each file.
    This includes descriptions, dataset associations, and detailed metadata.
    """
    if not files:
        return []

    file_ids = [str(f.id) for f in files]

    # Batch load ALL information for files
    dataset_names_map = await _get_file_dataset_names_batch(db, file_ids)
    descriptions_map = await _get_file_descriptions_batch(db, file_ids)

    # Build comprehensive FileInfo objects
    file_infos = []
    for file in files:
        file_id_str = str(file.id)
        dataset_names = dataset_names_map.get(file_id_str, [])
        description, content_summary = descriptions_map.get(file_id_str, (None, None))

        # Always use DETAILED type for comprehensive information
        final_description, final_content_summary = _customize_description_by_type(
            FileInfoType.DETAILED, description, content_summary, dataset_names, None
        )

        file_info = FileInfo(
            name=file.filename,
            size=_format_file_size(file.size) if file.size else None,
            file_extension=Path(file.filename).suffix if file.filename else None,
            file_id=file_id_str,
            content_summary=final_content_summary,
            mimetype=file.mimetype,
            # estimated_rows=getattr(file, "rows", None),
            # estimated_columns=getattr(file, "columns", None),
            dataset_names=dataset_names,
        )
        file_infos.append(file_info)

    return file_infos


def _build_pagination_info(
    input_data: FileExplorerInput,
    total_count: int,
    limit: int,
    offset: int,
    used_fuzzy_matching: bool = False,
) -> dict:
    """Build comprehensive pagination information."""
    current_page = input_data.page or 1
    total_pages = (total_count + limit - 1) // limit if not used_fuzzy_matching else 1

    return {
        "current_page": current_page,
        "total_pages": total_pages,
        "limit": limit,
        "offset": offset,
        "showing_from": offset + 1 if total_count > 0 else 0,
        "showing_to": min(offset + limit, total_count),
        "total_items": total_count,
        "fuzzy_matching_used": used_fuzzy_matching,
    }


def _build_comprehensive_operation_result(
    input_data: FileExplorerInput,
    returned_count: int,
    total_count: int,
    offset: int,
    has_more: bool,
    next_page: Optional[int],
    used_fuzzy_matching: bool,
) -> str:
    """Build comprehensive operation result message."""
    _operation_type = input_data.operation
    search_pattern = input_data.search_pattern

    if used_fuzzy_matching:
        result = f"Fuzzy search completed: found {returned_count} similar files"
        if search_pattern:
            result += f" for '{search_pattern}'"
    elif search_pattern:
        result = f"Search completed: found {returned_count} files matching '{search_pattern}'"
        if has_more:
            result += f" (showing {offset + 1}-{offset + returned_count} of {total_count} total)"
            result += f" - Use page={next_page} to see more results"
    else:
        result = f"Listed {returned_count} files"
        if has_more:
            result += f" (showing {offset + 1}-{offset + returned_count} of {total_count} total)"
            result += f" - Use page={next_page} to see more results"

    return result


async def _query_files(
    db: AsyncSession,
    dataset_ids: List[str],
    filename_filter: Optional[str] = None,
    limit: int = 50,  # OPTIMIZED: Add pagination limit
    offset: int = 0,  # OPTIMIZED: Add pagination offset
) -> Tuple[List[File], int]:
    """
    Core file querying function with dataset filtering options.
    OPTIMIZED: Added pagination and count optimization

    Args:
        db: Database session
        dataset_ids: Filter by specific datasets
        filename_filter: Pattern to match filenames
        limit: Maximum number of files to return
        offset: Number of files to skip

    Returns:
        Tuple of (files_list, total_count)
    """
    # Build base query
    query = select(File)

    # Apply dataset filtering
    query = _apply_scope_filter(query, dataset_ids)

    # Apply filename filtering with pattern matching
    if filename_filter:
        query = _apply_filename_filter(query, filename_filter)

    # Get total count before applying limit/offset
    count_query = select(func.count()).select_from(query.distinct().subquery())
    total_count = (await db.execute(count_query)).scalar()

    # Apply pagination
    query = query.distinct().limit(limit).offset(offset)

    # Execute query and return results
    result = await db.execute(query)
    files = result.scalars().all()

    return files, total_count


def _apply_scope_filter(query, dataset_ids: List[str]):
    """Apply dataset filtering to query."""
    return query.join(DatasetFileLink, File.id == DatasetFileLink.file_id).where(
        DatasetFileLink.dataset_id.in_([UUID(id) for id in dataset_ids])
    )


def _apply_filename_filter(query, filename_filter: str):
    """Apply filename filtering with pattern matching support."""
    if "*" in filename_filter or "?" in filename_filter:
        # Convert shell-style wildcards to SQL LIKE patterns
        sql_pattern = filename_filter.replace("*", "%").replace("?", "_")
        return query.where(File.filename.like(sql_pattern))
    else:
        # Partial matching for simple strings
        return query.where(File.filename.contains(filename_filter))


async def _find_file_by_exact_name_optimized(
    db: AsyncSession,
    filename: str,
    dataset_ids: List[str],
) -> Optional[File]:
    """
    Optimized exact filename lookup using indexed query.
    This is faster than fuzzy matching for exact name searches.
    """
    try:
        # Build optimized query with proper indexing hints
        query = select(File).where(File.filename == filename)

        # Apply dataset filtering with optimized joins
        query = query.join(DatasetFileLink, File.id == DatasetFileLink.file_id).where(
            DatasetFileLink.dataset_id.in_([UUID(id) for id in dataset_ids])
        )

        # Add limit for safety and performance
        query = query.limit(1)

        result = await db.execute(query)
        return result.scalars().first()

    except Exception as e:
        logger.error(f"Error in optimized exact name lookup for '{filename}': {str(e)}")
        return None


async def _find_file_by_scope(
    db: AsyncSession,
    filename: str,
    dataset_ids: List[str],
) -> Optional[File]:
    """Unified file finding function with dataset-based filtering and fuzzy matching fallback."""
    # First try optimized exact match
    exact_match = await _find_file_by_exact_name_optimized(db, filename, dataset_ids)

    if exact_match:
        logger.debug(f"Found exact match for '{filename}' using optimized lookup")
        return exact_match

    # If no exact match, try fuzzy matching
    logger.info(
        f"No exact match for '{filename}' in datasets, trying fuzzy matching..."
    )

    fuzzy_match = await _find_single_best_match_optimized(
        db=db,
        query=filename,
        dataset_ids=dataset_ids,
        min_score=0.6,
    )

    return fuzzy_match


async def _build_single_file_info_optimized(
    db: AsyncSession,
    file: File,
    info_type: FileInfoType,
    dataset_ids: Optional[List[str]] = None,
    containing_datasets: Optional[List[dict]] = None,
) -> FileInfo:
    """
    Build a single FileInfo object based on the specified type.
    OPTIMIZED: Direct single file queries instead of batch when appropriate
    """
    # Get common file information
    dataset_names = await _get_file_dataset_names(db, str(file.id))
    description, content_summary = await _get_file_description_from_documents_sync(
        db, str(file.id)
    )

    # Customize description based on info type
    final_description, final_content_summary = _customize_description_by_type(
        info_type, description, content_summary, dataset_names, containing_datasets
    )

    return FileInfo(
        name=file.filename,
        size=_format_file_size(file.size) if file.size else None,
        file_extension=Path(file.filename).suffix if file.filename else None,
        file_id=str(file.id),
        content_summary=final_content_summary,
        mimetype=file.mimetype,
        # estimated_rows=getattr(file, "rows", None),
        # estimated_columns=getattr(file, "columns", None),
        dataset_names=(
            dataset_names
            if not containing_datasets
            else [ds["name"] for ds in containing_datasets]
        ),
        ingestion_status=getattr(file, "status", None),
    )


async def _get_file_dataset_names_batch(
    db: AsyncSession, file_ids: List[str]
) -> Dict[str, List[str]]:
    """
    Get dataset names for multiple files in a single query.
    OPTIMIZED: Batch loading instead of N+1 queries
    """
    if not file_ids:
        return {}

    query = (
        select(DatasetFileLink.file_id, Dataset.name)
        .join(Dataset, DatasetFileLink.dataset_id == Dataset.id)
        .where(DatasetFileLink.file_id.in_([UUID(fid) for fid in file_ids]))
        .where(Dataset.deleted_at.is_(None))
    )

    result = await db.execute(query)

    # Group dataset names by file_id
    file_datasets_map = {}
    for file_id, dataset_name in result.fetchall():
        file_id_str = str(file_id)
        if file_id_str not in file_datasets_map:
            file_datasets_map[file_id_str] = []
        file_datasets_map[file_id_str].append(dataset_name)

    return file_datasets_map


async def _get_file_descriptions_batch(
    db: AsyncSession, file_ids: List[str]
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """
    Get file descriptions for multiple files in optimized queries.
    OPTIMIZED: Batch loading instead of N+1 queries
    """
    if not file_ids:
        return {}

    # Single query to get all descriptions
    query = (
        select(
            Document.file_id,
            Document.description,
            Document.is_split_document,
            Document.processing_status,
            Document.created_at,
        )
        .where(
            Document.file_id.in_([UUID(fid) for fid in file_ids]),
            Document.description.is_not(None),
            Document.deleted_at.is_(None),
        )
        .order_by(Document.file_id, Document.created_at.desc())
    )

    result = await db.execute(query)

    # Group documents by file_id and find best document for each file
    file_documents_map = {}
    for file_id, description, is_split, status, created_at in result.fetchall():
        file_id_str = str(file_id)
        if file_id_str not in file_documents_map:
            file_documents_map[file_id_str] = []
        file_documents_map[file_id_str].append(
            {
                "description": description,
                "is_split_document": is_split,
                "processing_status": status,
                "created_at": created_at,
            }
        )

    # Find best description for each file
    descriptions_map = {}
    for file_id_str, documents in file_documents_map.items():
        best_doc = _find_best_document_from_data(documents)
        if best_doc:
            description = best_doc["description"]
            # Always use full description - no truncation for content_summary
            content_summary = description
            descriptions_map[file_id_str] = (description, content_summary)
        else:
            descriptions_map[file_id_str] = (None, None)

    return descriptions_map


def _find_best_document_from_data(documents: List[dict]) -> Optional[dict]:
    """Find the best document from document data using priority strategy."""
    if not documents:
        return None

    # Strategy 1: Find the primary document (non-split, successful)
    for doc in documents:
        if (not doc["is_split_document"]) and doc[
            "processing_status"
        ] == DocumentProcessingStatusEnum.Success:
            return doc

    # Strategy 2: If no primary document, find the most recent successful document
    for doc in documents:
        if doc["processing_status"] == DocumentProcessingStatusEnum.Success:
            return doc

    # Strategy 3: Fallback to first document
    return documents[0]


def _customize_description_by_type(
    info_type: FileInfoType,
    description: Optional[str],
    content_summary: Optional[str],
    dataset_names: List[str],
    containing_datasets: Optional[List[dict]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Customize description and content summary based on info type."""
    if info_type == FileInfoType.BASIC:
        # Always return full description for both description and content_summary
        return description, description

    elif info_type == FileInfoType.DETAILED:
        # Full description for metadata operation - return same for both
        return description, description

    elif info_type == FileInfoType.DESCRIPTION:
        # Enhanced description for description operation - return same for both
        enhanced_description = (
            description or "No AI-generated description available for this file."
        )
        return enhanced_description, enhanced_description

    elif info_type == FileInfoType.DATASET_AWARE:
        # Enhanced description to include dataset information
        if containing_datasets:
            dataset_info = f"This file is present in {len(containing_datasets)} dataset(s): {', '.join([ds['name'] for ds in containing_datasets])}."
            enhanced_description = (
                f"{dataset_info} {description}" if description else dataset_info
            )
            # Always return the same enhanced description for both
            return enhanced_description, enhanced_description

    # Always return the same value for both description and content_summary
    return description, description


def _build_fuzzy_matching_results(
    search_pattern: str, fuzzy_matches: List[Tuple[File, float]]
) -> List[str]:
    """Build search results for fuzzy matching."""
    results = [f"Used fuzzy matching for pattern '{search_pattern}':"]
    results.extend(
        [
            f"- {match[0].filename} (similarity: {match[1]:.1%})"
            for match in fuzzy_matches[:5]  # Show top 5 with scores
        ]
    )
    if len(fuzzy_matches) > 5:
        results.append(f"... and {len(fuzzy_matches) - 5} more matches")
    return results


def _build_exact_search_results(
    search_pattern: str, file_infos: List[FileInfo]
) -> List[str]:
    """Build search results for exact matching."""
    results = [f"Found {len(file_infos)} files matching pattern '{search_pattern}'"]
    if file_infos:
        results.extend(
            [f"- {info.name}" for info in file_infos[:5]]
        )  # Show first 5 results
        if len(file_infos) > 5:
            results.append(f"... and {len(file_infos) - 5} more files")
    return results


async def _get_file_dataset_names(db: AsyncSession, file_id: str) -> List[str]:
    """Get user-friendly dataset names for a file."""
    query = (
        select(Dataset.name)
        .join(DatasetFileLink)
        .where(DatasetFileLink.file_id == UUID(file_id))
        .where(Dataset.deleted_at.is_(None))
    )
    result = await db.execute(query)
    dataset_names = result.scalars().all()

    return list(dataset_names) if dataset_names else []


async def _get_primary_dataset_name(
    db: AsyncSession, dataset_ids: Optional[List[str]]
) -> Optional[str]:
    """Get user-friendly name for primary dataset in context."""
    if not dataset_ids or not dataset_ids[0]:
        return None

    try:
        dataset_id_str = dataset_ids[0]
        logger.debug(f"Getting dataset name for ID: {dataset_id_str}")

        # Validate UUID format before querying
        try:
            dataset_uuid = UUID(dataset_id_str)
        except (ValueError, TypeError) as uuid_error:
            logger.error(
                f"Invalid UUID format for dataset ID '{dataset_id_str}': {uuid_error}"
            )
            return None

        # Execute query with proper error handling
        query = select(Dataset).where(
            Dataset.id == dataset_uuid, Dataset.deleted_at.is_(None)
        )

        try:
            result = await db.execute(query)
            dataset = result.scalars().first()
        except Exception as query_error:
            logger.error(
                f"Database query failed for dataset ID {dataset_id_str}: {query_error}"
            )
            # Let the main function handle transaction rollback
            return None

        if dataset:
            logger.debug(f"Found dataset name: {dataset.name}")
            return dataset.name
        else:
            logger.warning(f"No dataset found with ID: {dataset_id_str}")
            return None

    except Exception as e:
        logger.error(
            f"Error getting dataset name for ID {dataset_ids[0] if dataset_ids else 'None'}: {str(e)}"
        )
        # Let the main function handle transaction rollback
        return None


async def _get_file_containing_datasets(db: AsyncSession, file_id: str) -> List[dict]:
    """Get all datasets that contain a specific file with their metadata."""
    query = (
        select(Dataset.id, Dataset.name, Dataset.description)
        .join(DatasetFileLink)
        .where(DatasetFileLink.file_id == UUID(file_id), Dataset.deleted_at.is_(None))
        .order_by(Dataset.name)
    )

    result = await db.execute(query)
    datasets = []

    for row in result.fetchall():
        datasets.append(
            {
                "id": str(row[0]),
                "name": row[1],
                "description": row[2] or "No description available",
            }
        )

    return datasets


async def _get_file_description_from_documents_sync(
    db: AsyncSession, file_id: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get file description and content summary from associated documents.

    Returns:
        tuple: (description, content_summary)
    """
    logger.debug(f"Getting file description for file_id={file_id}")
    try:
        # Query documents for this file using async SQLAlchemy
        query = (
            select(Document)
            .filter(
                Document.file_id == UUID(file_id),
                Document.description.is_not(None),
                Document.deleted_at.is_(None),
            )
            .order_by(Document.created_at.desc())
        )

        logger.debug("Executing document query")
        result = await db.execute(query)
        documents = result.scalars().all()

        logger.debug(f"Found {len(documents)} documents for file_id={file_id}")

        if not documents:
            logger.debug("No documents found, returning None")
            return None, None

        # Find the best document using priority strategy
        primary_doc = _find_best_document(documents)
        description = primary_doc.description

        logger.debug(
            f"Retrieved description length: {len(description) if description else 0}"
        )

        # Always use full description - no truncation for content_summary
        content_summary = description
        logger.debug("Using full description as content summary")

        logger.debug("Successfully retrieved file description from documents")
        return description, content_summary

    except Exception as e:
        logger.error(
            f"Error getting description from documents for file {file_id}: {str(e)}"
        )
        return None, None


def _find_best_document(documents: List[Document]) -> Document:
    """Find the best document using priority strategy."""
    # Strategy 1: Find the primary document (non-split, successful)
    logger.debug("Strategy 1: Looking for primary document (non-split, successful)")
    for doc in documents:
        if (
            not doc.is_split_document
        ) and doc.processing_status == DocumentProcessingStatusEnum.Success:
            logger.debug(f"Found primary document: {doc.id}")
            return doc

    # Strategy 2: If no primary document, find the most recent successful document
    logger.debug("Strategy 2: Looking for most recent successful document")
    for doc in documents:
        if doc.processing_status == DocumentProcessingStatusEnum.Success:
            logger.debug(f"Found successful document: {doc.id}")
            return doc

    # Strategy 3: Fallback to any document with description
    logger.debug("Strategy 3: Fallback to first document with description")
    primary_doc = documents[0]
    logger.debug(f"Using fallback document: {primary_doc.id}")
    return primary_doc


def _calculate_total_size(files: List[File]) -> Optional[str]:
    """Calculate and format total size of files."""
    total_bytes = sum(file.size for file in files if file.size)
    return _format_file_size(total_bytes) if total_bytes > 0 else None


def _format_file_size(size_in_bytes: Optional[int]) -> Optional[str]:
    """
    Format file size in bytes to a human-readable string (KB, MB, GB).

    Args:
        size_in_bytes: File size in bytes (can be None)

    Returns:
        Formatted string with appropriate size unit, or None if input is None
    """
    if size_in_bytes is None:
        return None

    # Convert to appropriate unit
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        return f"{size_in_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_in_bytes / (1024 * 1024 * 1024):.1f} GB"


def _determine_context_path(
    input_data: FileExplorerInput, dataset_ids: List[str]
) -> str:
    """Determine appropriate context path for operation results."""
    if len(dataset_ids) == 1:
        return f"Dataset: {dataset_ids[0]}"
    else:
        return f"Multiple Datasets ({len(dataset_ids)} datasets)"


async def _find_best_matching_files_optimized(
    db: AsyncSession,
    query: str,
    dataset_ids: List[str],
    min_score: float = 0.6,
    max_results: int = 10,
) -> List[Tuple[File, float]]:
    """
    Find files with optimized fuzzy matching using PostgreSQL trigram similarity.
    OPTIMIZED: Uses database-level fuzzy matching instead of loading all files into memory
    """
    try:
        logger.info(f"Starting optimized fuzzy search for: '{query}'")

        # Build joins based on scope
        params = {"query": query, "min_score": min_score, "max_results": max_results}

        similarity_query = text(
            """
            SELECT f.*, similarity(f.filename, :query) as sim_score
            FROM files f
            INNER JOIN dataset_file_links dfl ON f.id = dfl.file_id
            WHERE dfl.dataset_id = ANY(:dataset_ids)
            AND similarity(f.filename, :query) > :min_score
            ORDER BY sim_score DESC
            LIMIT :max_results
        """
        )
        # Convert string dataset_ids to UUID objects for asyncpg compatibility
        params["dataset_ids"] = [UUID(id) for id in dataset_ids]

        logger.debug(f"Executing optimized fuzzy search query with params: {params}")
        result = await db.execute(similarity_query, params)

        # Convert results to File objects with scores
        matches = []
        for row in result.fetchall():
            # Create File object from row data
            file_data = dict(row._mapping)
            sim_score = file_data.pop("sim_score")

            # Create File object
            file_obj = File(**{k: v for k, v in file_data.items() if hasattr(File, k)})
            matches.append((file_obj, float(sim_score)))

        logger.info(f"Optimized fuzzy matching returned {len(matches)} results")
        return matches

    except Exception as e:
        logger.error(
            f"Optimized fuzzy matching failed, falling back to original method: {str(e)}"
        )
        # Fallback to original method if PostgreSQL trigram is not available
        return await _find_best_matching_files_fallback(
            db=db,
            query=query,
            dataset_ids=dataset_ids,
            min_score=min_score,
            max_results=max_results,
        )


async def _find_best_matching_files_fallback(
    db: AsyncSession,
    query: str,
    dataset_ids: List[str],
    min_score: float = 0.6,
    max_results: int = 10,
) -> List[Tuple[File, float]]:
    """
    Fallback fuzzy matching method using limited dataset for better performance.
    OPTIMIZED: Limited scope and early termination to avoid memory issues
    """
    try:
        logger.info(f"Using fallback fuzzy search for: '{query}'")

        # Build query with proper aliasing and limits
        files_query = (
            select(File)
            .join(DatasetFileLink, File.id == DatasetFileLink.file_id)
            .where(DatasetFileLink.dataset_id.in_([UUID(id) for id in dataset_ids]))
            .distinct()
            .limit(1000)  # OPTIMIZED: Limit to prevent memory issues
        )

        logger.debug("Executing limited fallback fuzzy search query")
        result = await db.execute(files_query)
        limited_files = result.scalars().all()

        if not limited_files:
            logger.warning(
                "No files found in specified scope for fallback fuzzy search"
            )
            return []

        logger.info(f"Found {len(limited_files)} files for fallback fuzzy matching")

        # Prepare choices with normalized names for better matching
        choices, file_mapping = _prepare_fuzzy_choices(limited_files)

        if not choices:
            logger.warning("No valid filenames found for fallback fuzzy matching")
            return []

        # Normalize query and perform fuzzy matching
        normalized_query = _normalize_filename(query)
        logger.info(f"Normalized query: '{normalized_query}'")

        matches = process.extract(
            normalized_query,
            choices,
            scorer=fuzz.token_set_ratio,
            limit=max_results,
        )

        logger.info(f"Top fallback fuzzy matches: {matches[:3]}")

        # Filter by minimum score and return File objects with scores
        results = _filter_fuzzy_matches(matches, file_mapping, min_score)

        logger.info(
            f"Fallback fuzzy matching returned {len(results)} results above threshold {min_score}"
        )
        return results

    except Exception as e:
        logger.error(f"Fallback fuzzy matching failed: {str(e)}", exc_info=True)
        return []


def _prepare_fuzzy_choices(all_files: List[File]) -> Tuple[List[str], Dict[str, File]]:
    """Prepare normalized choices and file mapping for fuzzy matching."""
    choices = []
    file_mapping = {}

    for file in all_files:
        if file.filename:
            # Use full filename for mapping but normalize for matching
            normalized = _normalize_filename(file.filename)
            choices.append(normalized)
            file_mapping[normalized] = file

    logger.debug(f"Prepared {len(choices)} choices for fuzzy matching")
    return choices, file_mapping


def _normalize_filename(filename: str) -> str:
    """Normalize filename for better fuzzy matching."""
    # Keep the original filename structure but normalize case and separators
    return filename.lower().replace("_", " ").replace("-", " ")


def _filter_fuzzy_matches(
    matches: List[Tuple[str, float, int]],
    file_mapping: Dict[str, File],
    min_score: float,
) -> List[Tuple[File, float]]:
    """Filter fuzzy matches by minimum score and return File objects with scores."""
    results = []
    score_threshold = min_score * 100

    for match_text, score, _ in matches:
        if score >= score_threshold and match_text in file_mapping:
            file_obj = file_mapping[match_text]
            normalized_score = score / 100
            results.append((file_obj, normalized_score))
            logger.debug(
                f"Added match: {file_obj.filename} (score: {normalized_score:.2f})"
            )

    return results


async def _find_single_best_match_optimized(
    db: AsyncSession,
    query: str,
    dataset_ids: List[str],
    min_score: float = 0.6,
) -> Optional[File]:
    """
    Find the single best matching file using optimized fuzzy search.
    OPTIMIZED: Uses database-level similarity matching
    """
    matches = await _find_best_matching_files_optimized(
        db=db,
        query=query,
        dataset_ids=dataset_ids,
        min_score=min_score,
        max_results=1,
    )

    return matches[0][0] if matches else None


def _generate_file_types_summary(files: List[File]) -> dict:
    """Generate summary of file types found."""
    if not files:
        return {}

    file_type_counts = {}
    for file in files:
        if file.filename:
            extension = Path(file.filename).suffix.lower()
            if extension:
                file_type_counts[extension] = file_type_counts.get(extension, 0) + 1
            else:
                file_type_counts["(no extension)"] = (
                    file_type_counts.get("(no extension)", 0) + 1
                )

    return file_type_counts


async def _get_datasets_involved(
    db: AsyncSession, dataset_ids: Optional[List[str]], files: List[File]
) -> Optional[List[str]]:
    """Get all datasets involved in this operation."""
    if dataset_ids:
        # Get names for provided dataset IDs
        query = select(Dataset.name).where(
            Dataset.id.in_([UUID(id) for id in dataset_ids]),
            Dataset.deleted_at.is_(None),
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    elif files:
        # Get all unique datasets that contain these files
        file_ids = [str(f.id) for f in files]
        dataset_names_map = await _get_file_dataset_names_batch(db, file_ids)

        # Collect all unique dataset names
        all_datasets = set()
        for datasets in dataset_names_map.values():
            all_datasets.update(datasets)

        return sorted(all_datasets)

    return None


def _extract_keywords_for_pattern_search(content_query: str) -> List[str]:
    """Extract meaningful keywords from content query for pattern-based file search."""
    import re

    # Convert to lowercase and remove common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
    }

    # Extract words and numbers, filter out stop words and short words
    words = re.findall(r"\b\w+\b", content_query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) >= 3]

    # Also extract quoted phrases and numbers/years
    quoted_phrases = re.findall(r'"([^"]*)"', content_query)
    numbers = re.findall(r"\b\d{4}\b", content_query)  # Years like 2019

    # Combine all meaningful terms
    all_keywords = keywords + quoted_phrases + numbers

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in all_keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)

    # Limit to most relevant keywords (max 5)
    return unique_keywords[:5]


def _generate_context_hints(
    input_data: FileExplorerInput,
    files_returned: int,
    total_count: int,
    used_fuzzy_matching: bool,
) -> List[str]:
    """Generate helpful context hints for the user."""
    hints = []

    if used_fuzzy_matching:
        hints.append("🔍 Used fuzzy matching to find similar files")

    if input_data.search_pattern and files_returned == 0:
        hints.append(
            "💡 Try using wildcards like *.pdf or *report* for broader searches"
        )

    if files_returned > 0 and total_count > files_returned:
        hints.append(
            f"📄 Showing {files_returned} of {total_count} total files - use pagination to see more"
        )

    if input_data.operation == "list" and not input_data.search_pattern:
        hints.append("🔎 Use search_pattern to find specific files by name")

    if files_returned > 10:
        hints.append(
            "📊 Large result set returned - consider using more specific search criteria"
        )

    if input_data.dataset_ids and len(input_data.dataset_ids) > 1:
        hints.append(
            "🗂️ Searching across multiple datasets - results may be from different sources"
        )

    return hints
