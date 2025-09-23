"""
Utility module for centralized document processing to replace R2R dependency.
Provides comprehensive PDF processing functionality including text extraction,
image extraction, table extraction, chunking, and embedding generation.
"""

import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import pandas as pd
import torch
from docling.datamodel.base_models import InputFormat

# Import PDF processing utilities
from docling_core.types.doc import ImageRefMode, PictureItem
from sqlalchemy.orm import attributes
from unstructured.documents.elements import PageBreak
from unstructured.partition.md import partition_md

from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.models.file_split_model import FileSplit
from app.utils.docling_model_manager import get_docling_converter
from app.utils.image_processing_utils import process_image_file_to_extract_data
from app.utils.ingestion_benchmark import IngestionBenchmark
from app.utils.openai_utils import (
    chat_completion_with_retry,
    generate_embedding,
    generate_embedding_with_retry,
    get_openai_client,
)


def _get_document_type_from_file_path(file_path: str) -> DocumentTypeEnum:
    """
    Determine document type based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        DocumentTypeEnum corresponding to the file type
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    type_mapping = {
        ".pdf": DocumentTypeEnum.PDF,
        ".docx": DocumentTypeEnum.DOCX,
        ".xlsx": DocumentTypeEnum.XLSX,
        ".pptx": DocumentTypeEnum.PPTX,
        ".md": DocumentTypeEnum.Markdown,
        ".html": DocumentTypeEnum.HTML,
        ".htm": DocumentTypeEnum.HTML,
        ".csv": DocumentTypeEnum.CSV,
    }

    return type_mapping.get(extension, DocumentTypeEnum.PDF)  # Default to PDF


def _get_docling_input_format(file_path: str) -> InputFormat:
    """
    Get the appropriate docling InputFormat based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        InputFormat enum for docling
    """
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()

    format_mapping = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".xlsx": InputFormat.XLSX,
        ".pptx": InputFormat.PPTX,
        ".md": InputFormat.MD,
        ".html": InputFormat.HTML,
        ".htm": InputFormat.HTML,
        ".csv": InputFormat.CSV,
    }

    return format_mapping.get(extension, InputFormat.PDF)  # Default to PDF


def process_document(
    file_path: str,
    file_id: UUID,
    document_id: UUID,
    dataset_id: UUID,
    chunking_config: Dict[str, Any] = None,
    ingestion_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    split_id: Optional[str] = None,
    skip_successful_files: bool = True,
) -> Dict[str, Any]:
    """
    Process a document completely, extracting text, images, and tables.
    Supports PDF, DOCX, XLSX, PPTX, Markdown, HTML, and CSV files.

    Args:
        file_path: Path to the document file
        file_id: ID of the original file
        document_id: Document ID to assign
        dataset_id: Dataset ID
        chunking_config: Configuration for text chunking
        ingestion_id: Optional ingestion ID for tracking
        user_id: Optional user ID who initiated the processing
        split_id: Optional ID of the split if this is a split being processed
        skip_successful_files: Whether to skip files that are already successfully processed

    Returns:
        Dictionary with processing results
    """
    logger.info(
        f"Starting document processing for {file_path} with document ID {document_id} (split_id: {split_id})"
    )

    benchmark = IngestionBenchmark(str(file_id), split_id)

    # Determine document type
    document_type = _get_document_type_from_file_path(file_path)
    input_format = _get_docling_input_format(file_path)

    logger.info(f"Processing {document_type.value} document with format {input_format}")

    # Initialize processing results and configuration
    results = _initialize_processing_results()
    chunk_size, chunk_overlap = _setup_chunking_config(chunking_config)

    if chunk_overlap > chunk_size:
        raise ValueError(
            "Chunk overlap cannot be greater than chunk size. "
            f"Received chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    # Create temporary directory for processing outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract file metadata and setup converter
            benchmark.start("total_processing")
            file_path_obj, file_size, file_name = _get_file_metadata(file_path)
            # torch.set_default_device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            doc_converter = get_docling_converter(document_type)
            benchmark.end("total_processing")

            # Convert the document once for all extractions
            logger.info("Converting document once for all extractions...")
            benchmark.start("document_conversion")
            conv_result = doc_converter.convert(file_path_obj)
            benchmark.end("document_conversion")
            doc_filename = conv_result.input.file.stem

            # Setup temporary directories and process content
            _setup_temp_directories(temp_dir)

            # Extract text content from the document
            benchmark.start("text_extraction")
            md_path = extract_text_from_document(conv_result, file_path, temp_dir)
            benchmark.end("text_extraction")

            # Process text, images, and tables
            benchmark.start("text_processing")
            text_chunks, all_text = process_text(
                md_path=md_path,
                document_id=document_id,
                dataset_id=dataset_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_config=chunking_config,
                split_id=split_id,
            )
            benchmark.end("text_processing")

            benchmark.start("table_processing")
            has_tables, table_chunks = process_tables(
                conv_result,
                document_id,
                dataset_id,
                split_id=split_id,
            )
            benchmark.end("table_processing")

            num_pages = _get_page_count(conv_result)

            # Only process images for document types that can contain them
            has_images, image_chunks = False, []
            if document_type in [
                DocumentTypeEnum.PDF,
                DocumentTypeEnum.DOCX,
                DocumentTypeEnum.PPTX,
                DocumentTypeEnum.HTML,
            ]:
                benchmark.start("image_processing")
                has_images, image_chunks = process_images(
                    conv_result,
                    doc_filename,
                    document_id,
                    dataset_id,
                    ingestion_id=ingestion_id,
                    user_id=user_id,
                    split_id=split_id,
                    skip_successful_files=skip_successful_files,
                )
                benchmark.end("image_processing")

            # Only handle Document table operations for original documents
            if not split_id:
                # Generate document summary and embedding only for original documents
                benchmark.start("summary_generation")
                summary = create_document_summary(all_text, num_pages)
                summary_embedding = generate_embedding_with_retry(summary)
                benchmark.end("summary_generation")

                # Process database operations for original document
                benchmark.start("database_operations")
                with get_session() as session:
                    try:
                        # Get or create document record (only for original documents)
                        document = _get_or_create_document(
                            session,
                            document_id,
                            dataset_id,
                            file_id,
                            summary,
                            summary_embedding,
                            file_path,
                            file_name,
                            file_size,
                            ingestion_id,
                            document_type,
                        )

                        attributes.flag_modified(document, "document_metadata")
                        session.add(document)

                        # Process chunks (both original and splits create chunks)
                        _process_text_chunks(
                            session, text_chunks, document_id, split_id, results
                        )
                        _process_table_chunks(
                            session, table_chunks, document_id, split_id, results
                        )

                        session.commit()

                        # Store document reference for finalization
                        results["_document_reference"] = document

                    except Exception as e:
                        _handle_database_error(session, document_id, e, results)

                    benchmark.end("database_operations")
            else:
                # For splits, only process chunks - no Document table operations
                benchmark.start("database_operations")
                with get_session() as session:
                    try:
                        # Check if this is the first split (index 0) and handle document description
                        should_generate_description = (
                            _should_generate_document_description(split_id)
                        )

                        if should_generate_description:
                            # Generate document description from first split
                            benchmark.start("split_summary_generation")
                            summary, summary_embedding = (
                                _generate_document_description_from_split(
                                    all_text, num_pages, split_id
                                )
                            )

                            # Update document with description and embedding
                            _update_document_with_description(
                                session,
                                document_id,
                                dataset_id,
                                file_id,
                                summary,
                                summary_embedding,
                                split_id,
                            )
                            benchmark.end("split_summary_generation")

                        # Process chunks for splits
                        _process_text_chunks(
                            session, text_chunks, document_id, split_id, results
                        )
                        _process_table_chunks(
                            session, table_chunks, document_id, split_id, results
                        )

                        session.commit()

                    except Exception as e:
                        logger.error(f"Error processing split {split_id}: {str(e)}")
                        results["success"] = False
                        results["error"] = str(e)

                    benchmark.end("database_operations")

            benchmark.end("total_processing")
            results["benchmarks"] = benchmark.get_summary()

            # Finalize results based on whether this is a split or direct document
            if split_id:
                # This is a split document
                _finalize_split_results(
                    results=results,
                    document_id=document_id,
                    split_id=split_id,
                    has_images=has_images,
                    image_chunks=image_chunks,
                    table_chunks=table_chunks,
                    text_chunks=text_chunks,
                )
            else:
                # This is a direct document
                document = results.get(
                    "_document_reference"
                )  # Get stored document reference
                _finalize_direct_document_results(
                    results=results,
                    document=document,
                    document_id=document_id,
                    has_images=has_images,
                    image_chunks=image_chunks,
                    table_chunks=table_chunks,
                    text_chunks=text_chunks,
                )

        except Exception as e:
            benchmark.end("total_processing")
            logger.error(f"Error in document processing: {str(e)}")
            results["errors"].append(f"Processing error: {str(e)}")

    return results


def process_pdf_file_to_extract_data(file_path: str) -> str:
    """Process a PDF file and return the extracted data"""
    ocr_text = ""

    # Create temporary directory for processing outputs
    # Use tempfile.mkdtemp() for secure temporary directory creation
    temp_dir = tempfile.mkdtemp(prefix="pdf_processing_")

    try:
        doc_converter = get_docling_converter(DocumentTypeEnum.PDF)

        file_path_obj = Path(file_path)
        file_name = file_path_obj.name

        # Convert the document once for all extractions
        logger.info("Converting PDF document once for all extractions...")

        conv_result = doc_converter.convert(file_path_obj)

        # Setup temporary directories and process content
        _setup_temp_directories(temp_dir)

        # Extract text content from the document
        md_path = extract_text_from_document(conv_result, file_path, temp_dir)

        if md_path:
            # Process the text file
            text_content = _process_md_file(md_path)
            ocr_text += "\n" + text_content

        has_tables, table_chunks = process_tables_lite(
            conv_result,
        )
        if has_tables:
            ocr_text += "\n" + "\n".join(
                [chunk["table_markdown"] for chunk in table_chunks]
            )

        # Process each image element in the document
        figure_counter = 0
        max_images_to_process = 5  # Limit to first 5 images

        for element, _level in conv_result.document.iterate_items():
            if isinstance(element, PictureItem):
                figure_counter += 1

                # Skip if we've already processed 5 images
                if figure_counter > max_images_to_process:
                    logger.info(
                        f"Skipping image {figure_counter} - already processed {max_images_to_process} images"
                    )
                    continue

                # Generate filename and paths
                images_dir = Path(temp_dir) / "images"
                figure_filename = f"{file_name}_Image{figure_counter}.png"
                figure_path = images_dir / figure_filename

                # Save image to disk - skip if save fails
                if not _save_image_to_disk(element, conv_result, figure_path):
                    continue

                # Process the extracted image and add to OCR text
                image_ocr_text = process_image_file_to_extract_data(str(figure_path))
                if image_ocr_text:
                    ocr_text += image_ocr_text + "\n"

        logger.info(
            f"Successfully processed PDF with {min(figure_counter, max_images_to_process)} images (total images found: {figure_counter})"
        )
        logger.info(f"OCR text: {ocr_text}")
        return ocr_text

    except Exception as e:
        logger.error(f"Error processing PDF file: {str(e)}")
        return f"Error processing PDF file: {str(e)}"
    finally:
        # Clean up the temporary directory
        try:
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def _generate_image_filename(
    doc_filename: str, split_id: Optional[str], figure_counter: int
) -> str:
    """Generate appropriate filename for extracted image."""
    if split_id:
        return f"{doc_filename}_split_{split_id}_Image{figure_counter}.png"
    else:
        return f"{doc_filename}_Image{figure_counter}.png"


def _extract_page_number(element) -> Optional[int]:
    """Extract page number from element provenance."""
    if hasattr(element, "prov") and element.prov and len(element.prov) > 0:
        if hasattr(element.prov[0], "page_no"):
            return element.prov[0].page_no
    return None


def _create_image_metadata(
    figure_counter: int,
    document_id: UUID,
    dataset_id: UUID,
    page_no: Optional[int],
    split_id: Optional[str],
) -> Dict[str, Any]:
    """Create metadata dictionary for image processing."""
    return {
        "chunk_order": figure_counter,
        "file_id": str(document_id),
        "dataset_id": str(dataset_id),
        "chunked_by_engine": "docling",
        "source": "pdf",
        "page_number": page_no,
        "parent_document_id": str(document_id),
        "extracted_from_pdf": True,
        "split_id": split_id,
    }


def _create_image_chunk_info(
    image_file_id: str,
    page_no: Optional[int],
    figure_counter: int,
    figure_path: Path,
    split_id: Optional[str],
) -> Dict[str, Any]:
    """Create image chunk information dictionary."""
    return {
        "id": image_file_id,
        "page_number": page_no,
        "figure_number": figure_counter,
        "file_path": str(figure_path),
        "split_id": split_id,
    }


def _save_image_to_disk(element, conv_result, figure_path: Path) -> bool:
    """Save image element to disk. Returns True if successful."""
    try:
        with figure_path.open("wb") as fp:
            image = element.get_image(conv_result.document)
            image.save(fp, "PNG")

        if not figure_path.exists() or figure_path.stat().st_size == 0:
            logger.warning(f"Image file wasn't properly saved at {figure_path}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error saving image to {figure_path}: {str(e)}")
        return False


def _submit_image_processing_task(
    image_file_id: str,
    figure_path: Path,
    ingestion_id: Optional[str],
    dataset_id: UUID,
    user_id: Optional[UUID],
    image_metadata: Dict[str, Any],
    document_id: UUID,
) -> None:
    """Submit image processing task to Celery."""
    celery.signature(
        "tasks.image_ingestion_task_v2",
        kwargs={
            "file_id": image_file_id,
            "file_path": str(figure_path),
            "ingestion_id": ingestion_id,
            "dataset_id": str(dataset_id),
            "user_id": user_id,
            "metadata": image_metadata,
            "parent_document_id": str(document_id),
        },
    ).delay()


# def _initialize_image_summary_if_needed(parent_doc: Document) -> None:
#     """Initialize document metadata image summary structure if it doesn't exist."""
#     if not parent_doc.document_metadata:
#         parent_doc.document_metadata = {}

#     if "image_summary" not in parent_doc.document_metadata:
#         parent_doc.document_metadata["image_summary"] = {
#             "total_images": 0,
#             "images_by_split": {},
#             "images_by_page": {},
#             "all_image_info": [],
#             "processing_status": "pending",
#         }


def _clean_existing_split_data(
    image_summary: Dict[str, Any], split_id: Optional[str]
) -> None:
    """Remove old entries for a split when re-processing."""
    if not split_id:
        return

    # Remove old entries for this split from all_image_info
    image_summary["all_image_info"] = [
        img
        for img in image_summary["all_image_info"]
        if img.get("split_id") != split_id
    ]

    # Remove old page entries for this split
    for page_num in list(image_summary["images_by_page"].keys()):
        image_summary["images_by_page"][page_num] = [
            img
            for img in image_summary["images_by_page"][page_num]
            if img.get("split_id") != split_id
        ]
        # Remove empty page entries
        if not image_summary["images_by_page"][page_num]:
            del image_summary["images_by_page"][page_num]


def _update_split_image_data(
    image_summary: Dict[str, Any],
    split_key: str,
    image_chunks: List[Dict[str, Any]],
    split_id: Optional[str],
) -> None:
    """Update image summary with this split's data."""
    # Add this split's info
    image_summary["images_by_split"][split_key] = {
        "count": len(image_chunks),
        "images": image_chunks,
        "submitted_at": datetime.now().isoformat(),
    }

    # Update page-specific info
    for img in image_chunks:
        page_num = img.get("page_number", "unknown")
        if page_num not in image_summary["images_by_page"]:
            image_summary["images_by_page"][page_num] = []
        image_summary["images_by_page"][page_num].append(
            {
                "figure_number": img["figure_number"],
                "split_id": split_id,
                "image_id": img["id"],
            }
        )

    # Add to master list
    for img in image_chunks:
        img_info = img.copy()
        img_info["split_id"] = split_id
        img_info["submitted_at"] = datetime.now().isoformat()
        image_summary["all_image_info"].append(img_info)


def _finalize_document_image_summary(
    parent_doc: Document,
    existing_metadata: Dict[str, Any],
    image_summary: Dict[str, Any],
    split_id: Optional[str],
) -> None:
    """Finalize document image summary and handle special cases."""
    # Update total count
    total_images = sum(
        split_info["count"] for split_info in image_summary["images_by_split"].values()
    )
    image_summary["total_images"] = total_images
    image_summary["last_updated"] = datetime.now().isoformat()

    # Preserve original PDF summary for both direct and split ingestion
    # Only do this if not already preserved and description exists
    if parent_doc.description and not existing_metadata.get(
        "original_pdf_summary_preserved"
    ):
        existing_metadata["original_pdf_summary"] = parent_doc.description
        existing_metadata["original_pdf_summary_preserved"] = True
        logger.info(
            f"Preserved document description in metadata for document {parent_doc.id}"
        )

    # Handle different ingestion types
    if split_id:
        # Split-based ingestion
        if not existing_metadata.get("ingestion_type"):
            existing_metadata["ingestion_type"] = "split_based"
        existing_metadata["waiting_for_images"] = True
        existing_metadata["image_summary"]["processing_status"] = "processing"
    else:
        # Direct ingestion
        existing_metadata["ingestion_type"] = "direct"
        existing_metadata["waiting_for_images"] = True
        existing_metadata["image_summary"]["processing_status"] = "processing"


def _update_split_metadata(
    session, split_id: str, image_chunks: List[Dict[str, Any]]
) -> None:
    """Update split record metadata with image information."""
    split_record = session.query(FileSplit).filter(FileSplit.id == split_id).first()
    if not split_record:
        return

    # Initialize split metadata if needed
    if not split_record.split_metadata:
        split_record.split_metadata = {}

    # Store ONLY this split's image count
    split_record.split_metadata["has_images"] = True
    split_record.split_metadata["images_total"] = len(image_chunks)
    split_record.split_metadata["images_submitted_at"] = datetime.now().isoformat()
    split_record.split_metadata["image_info"] = image_chunks
    split_record.split_metadata["waiting_for_images"] = True

    session.add(split_record)
    attributes.flag_modified(split_record, "split_metadata")
    session.commit()

    logger.info(
        f"Updated split {split_id} with {len(image_chunks)} images for processing tracking"
    )


def _update_document_metadata_with_images(
    document_id: UUID, image_chunks: List[Dict[str, Any]], split_id: Optional[str]
) -> None:
    """Update document and split metadata with extracted image information."""
    try:
        with get_session() as session:
            # Handle document-level metadata first
            parent_doc = (
                session.query(Document).filter(Document.id == document_id).first()
            )
            if not parent_doc:
                logger.warning(f"Document {document_id} not found for metadata update")
                return

            existing_metadata = {}
            if parent_doc.document_metadata:
                existing_metadata = parent_doc.document_metadata.copy()

            # Initialize image summary
            if "image_summary" not in existing_metadata:
                existing_metadata["image_summary"] = {
                    "total_images": 0,
                    "images_by_split": {},
                    "images_by_page": {},
                    "all_image_info": [],
                    "processing_status": "pending",
                }

            # Update image data
            image_summary = existing_metadata["image_summary"]
            split_key = split_id if split_id else "direct"

            # Check if this split already exists (for re-processing scenarios)
            if split_key in image_summary["images_by_split"]:
                logger.info(
                    f"Split {split_key} already exists in image summary - updating"
                )
                _clean_existing_split_data(image_summary, split_id)

            # Update split data
            _update_split_image_data(image_summary, split_key, image_chunks, split_id)

            # Finalize document summary - PASS existing_metadata instead of parent_doc
            _finalize_document_image_summary(
                parent_doc, existing_metadata, image_summary, split_id
            )

            # Now assign the fully built existing_metadata back to the document
            parent_doc.document_metadata = existing_metadata

            session.add(parent_doc)
            attributes.flag_modified(parent_doc, "document_metadata")
            session.commit()

            logger.info(
                f"Updated document {document_id} image summary: "
                f"total={image_summary['total_images']}, current_split={split_key} (+{len(image_chunks)}), "
                f"all_splits={list(image_summary['images_by_split'].keys())}"
            )

            # Update split metadata if needed
            if split_id:
                _update_split_metadata(session, split_id, image_chunks)

    except Exception as e:
        logger.error(f"Error updating metadata with image info: {str(e)}")


def process_images(
    conv_result,
    doc_filename,
    document_id,
    dataset_id,
    ingestion_id=None,
    user_id=None,
    split_id=None,
    skip_successful_files=True,  # Add this parameter
):
    """
    Process and extract images from the document.
    Refactored to reduce cyclomatic complexity by using helper functions.
    """
    logger.info(f"Extracting figures... (split_id: {split_id})")
    figure_counter = 0
    has_images = False
    image_chunks = []

    # Setup permanent images directory
    try:
        permanent_images_dir = Path(settings.TEMP_PDF_IMG_DIR)
        permanent_images_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error(f"Failed to create permanent images directory: {str(e)}")
        raise

    # Process each image element in the document
    for element, _level in conv_result.document.iterate_items():
        if isinstance(element, PictureItem):
            has_images = True
            figure_counter += 1

            # Generate filename and paths
            figure_filename = _generate_image_filename(
                doc_filename, split_id, figure_counter
            )
            image_file_id = str(uuid.uuid4())
            figure_path = permanent_images_dir / figure_filename

            # Save image to disk - skip if save fails
            if not _save_image_to_disk(element, conv_result, figure_path):
                continue

            # Extract page number from element
            page_no = _extract_page_number(element)

            # Create metadata for image processing
            image_metadata = _create_image_metadata(
                figure_counter, document_id, dataset_id, page_no, split_id
            )

            # Submit image processing task
            _submit_image_processing_task(
                image_file_id,
                figure_path,
                ingestion_id,
                dataset_id,
                user_id,
                image_metadata,
                document_id,
            )

            # Add to image chunks list
            image_chunks.append(
                _create_image_chunk_info(
                    image_file_id, page_no, figure_counter, figure_path, split_id
                )
            )

            logger.info(
                f"Submitted image {figure_counter} from page {page_no if page_no else 'unknown'} "
                f"for processing (split_id: {split_id})"
            )

    # Update document and split metadata if images were found
    if has_images and len(image_chunks) > 0:
        _update_document_metadata_with_images(document_id, image_chunks, split_id)

    return has_images, image_chunks


def get_model_names() -> Dict[str, str]:
    """Get model names from environment variables or use defaults."""
    return {
        "chat_model": os.getenv("CHAT_COMPLETION_NAME", "gpt-4o"),
        "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large"),
    }


def get_table_html(table_df) -> str:
    """
    Convert pandas DataFrame to HTML table representation.

    Args:
        table_df: Pandas DataFrame representing a table

    Returns:
        HTML string of the table
    """
    return table_df.to_html(index=False)


def get_table_markdown(table_df) -> str:
    """
    Convert pandas DataFrame to Markdown table representation.

    Args:
        table_df: Pandas DataFrame representing a table

    Returns:
        Markdown string of the table
    """
    return table_df.to_markdown(index=False)


def analyze_table(table_df: pd.DataFrame, client=None) -> str:
    """
    Analyze a table and generate a description using Azure OpenAI.

    Args:
        table_df: Pandas DataFrame representing a table
        client: OpenAI client instance

    Returns:
        Table description
    """
    models = get_model_names()

    try:
        # Convert DataFrame to string representation
        table_string = table_df.to_string(index=False)

        # Limit input size
        if len(table_string) > 4000:
            table_string = table_string[:4000] + "..."

        prompt = f"Please analyze this table data and provide a concise summary of its content:\n\n{table_string}"

        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing tabular data and providing insightful summaries.",
            },
            {"role": "user", "content": prompt},
        ]

        # Use the retry-enabled chat completion function
        response_content = chat_completion_with_retry(
            messages=messages, model=models["chat_model"], max_tokens=500, client=client
        )

        return response_content
    except Exception as e:
        logger.error(f"Error analyzing table: {str(e)}")
        # Return basic info as fallback
        return f"Table with {len(table_df)} rows and {len(table_df.columns)} columns."


def extract_text_from_document(
    conv_result, file_path: str, output_dir: Optional[str] = None
) -> Path:
    """
    Extract text content from a PDF file using docling.

    Args:
        conv_result: Result from document converter
        file_path: Path to the PDF file
        output_dir: Optional directory for temporary files

    Returns:
        Tuple of (extracted_text, md_file_path)
    """
    logger.info(f"Extracting text from PDF: {file_path}")

    # Create temp directory if none specified
    if not output_dir:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir

    try:
        # Set up directory for text extraction
        text_dir = Path(output_dir) / "text"
        text_dir.mkdir(exist_ok=True)

        # Convert file_path to Path object
        file_path_obj = Path(file_path)

        # Extract MD file for text processing
        md_path = text_dir / f"{file_path_obj.stem}_full.md"

        # Use the provided conversion result instead of creating a new one
        conv_result.document.save_as_markdown(
            md_path,
            image_mode=ImageRefMode.EMBEDDED,
            page_break_placeholder="<!-- page -->",
        )

        logger.info(f"Successfully extracted text from {file_path}")
        return md_path

    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_tables_from_document(
    conv_result, file_path: str, output_dir: str
) -> List[Dict[str, Any]]:
    """
    Extract tables from a document file.
    Supports PDF, DOCX, XLSX, PPTX, HTML, and CSV files.

    Args:
        conv_result: Result from document converter
        file_path: Path to the document file
        output_dir: Directory to store extracted table data

    Returns:
        List of dictionaries with table data and metadata
    """
    logger.info(f"Extracting tables from document: {file_path}")

    try:
        # Create tables directory
        tables_dir = Path(output_dir) / "tables"
        tables_dir.mkdir(exist_ok=True)

        table_items = []

        # For CSV files, the entire file is essentially one table
        file_path_obj = Path(file_path)
        if file_path_obj.suffix.lower() == ".csv":
            # Handle CSV as a single table
            table_markdown = conv_result.document.export_to_markdown()

            # Save table to file
            table_path = tables_dir / "csv_table.md"
            with open(table_path, "w", encoding="utf-8") as f:
                f.write(table_markdown)

            table_items.append(
                {
                    "path": str(table_path),
                    "page_number": 1,
                    "table_index": 0,
                    "row_count": (
                        len(conv_result.document.texts)
                        if conv_result.document.texts
                        else 0
                    ),
                    "column_count": 1,  # CSV structure will be in the markdown
                    "markdown_content": table_markdown,
                    "caption": f"CSV data from {file_path_obj.name}",
                }
            )
        else:
            # Process tables from each page for other document types
            for page_idx, page in enumerate(conv_result.document.pages):
                for idx, table in enumerate(page.tables):
                    # Convert table to markdown
                    table_markdown = table.to_markdown()

                    # Save table to file
                    table_path = tables_dir / f"page_{page_idx+1}_table_{idx+1}.md"
                    with open(table_path, "w", encoding="utf-8") as f:
                        f.write(table_markdown)

                    # Create table metadata
                    table_items.append(
                        {
                            "path": str(table_path),
                            "page_number": page_idx + 1,
                            "table_index": idx,
                            "row_count": len(table.rows),
                            "column_count": len(table.columns) if table.columns else 0,
                            "markdown_content": table_markdown,
                            "caption": table.caption
                            or f"Table {idx+1} on page {page_idx+1}",
                        }
                    )

        logger.info(f"Extracted {len(table_items)} tables from {file_path}")
        return table_items

    except Exception as e:
        logger.error(f"Error extracting tables from document {file_path}: {str(e)}")
        return []


def generate_text_summary(text: str, max_length: int = 1000) -> str:
    """
    Generate a summary of the document text using Azure OpenAI.

    Args:
        text: Text to summarize
        max_length: Maximum length of text to consider for summarization

    Returns:
        Summarized text
    """
    # Truncate text if it's too long
    if len(text) > max_length:
        input_text = text[:max_length] + "..."
    else:
        input_text = text

    try:
        # Get model name
        models = get_model_names()
        chat_model = models["chat_model"]

        # Create summary prompt
        prompt = f"Please provide a concise summary of the following document in 2-3 sentences:\n\n{input_text}"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides concise document summaries.",
            },
            {"role": "user", "content": prompt},
        ]

        # Use the retry-enabled chat completion function
        summary = chat_completion_with_retry(
            messages=messages, model=chat_model, max_tokens=150, temperature=0.3
        )

        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        # Fallback to first 300 characters as summary
        if len(text) > 300:
            return text[:300] + "..."
        return text


# Database session helper function
def get_session():
    """Get a database session context manager."""
    from app.db.session import SyncSessionLocal

    return SyncSessionLocal()


def process_tables(doc_converter_result, document_id, dataset_id, split_id=None):
    """
    Process and extract tables from the document.

    Args:
        doc_converter_result: Result from document converter
        document_id: ID of the document
        dataset_id: ID of the dataset
        split_id: Optional ID of the split if this is a split being processed

    Returns:
        Tuple of (has_tables, table_chunks)
    """
    logger.info(f"Extracting tables... (split_id: {split_id})")
    has_tables = len(doc_converter_result.document.tables) > 0
    table_chunks = []
    client = get_openai_client()

    text_lookup = {
        text.self_ref: text.text for text in doc_converter_result.document.texts
    }

    for table_ix, table in enumerate(doc_converter_result.document.tables):
        try:
            table_df = table.export_to_dataframe()
            table_number = table_ix + 1

            # Get caption if available
            caption_text = ""
            if table.captions:
                caption_texts = [
                    text_lookup.get(ref.cref, "") for ref in table.captions
                ]
                caption_text = " ".join(caption_texts).strip()

            # Get page number
            page_no = table.prov[0].page_no if table.prov else None

            # Convert table to HTML
            table_html = get_table_html(table_df)

            # Generate AI analysis of the table
            table_analysis = analyze_table(table_df, client)

            # Create a summary for the table
            table_summary = f"Table {table_number}"
            if caption_text:
                table_summary += f": {caption_text}"
            table_summary += f". {table_analysis}"

            # Generate embedding for the table
            table_embedding = generate_embedding_with_retry(table_summary)

            # Create chunk record for the table
            current_timestamp = datetime.now().isoformat()
            table_chunk_id = str(uuid.uuid4())
            table_chunk = {
                "id": table_chunk_id,
                "document_id": document_id,
                "chunk_type": ChunkTypeEnum.PDFTable,
                "chunk_text": table_summary,
                "chunk_embedding": table_embedding,
                "created_at": current_timestamp,
                "deleted_at": None,
                "metadata": {
                    "version": "1.0",
                    "chunk_order": table_number,
                    "file_id": str(document_id),
                    "dataset_id": str(dataset_id),
                    "chunked_by_engine": "docling",
                    "source": "pdf",
                    "page_number": page_no,
                    "table_html": table_html,
                    "split_id": split_id,  # Include split_id in metadata
                },
            }

            table_chunks.append(table_chunk)
            logger.info(
                f"Processed table {table_number} from page {page_no if page_no else 'unknown'} "
                f"(split_id: {split_id})"
            )
        except Exception as e:
            logger.error(f"Error processing table {table_ix + 1}: {str(e)}")

    return has_tables, table_chunks


def process_tables_lite(doc_converter_result):
    """
    Process and extract tables from the document.

    Args:
        doc_converter_result: Result from document converter=

    Returns:
        Tuple of (has_tables, table_chunks)
    """
    has_tables = len(doc_converter_result.document.tables) > 0
    table_chunks = []

    text_lookup = {
        text.self_ref: text.text for text in doc_converter_result.document.texts
    }

    for table_ix, table in enumerate(doc_converter_result.document.tables):
        try:
            table_df = table.export_to_dataframe()
            table_number = table_ix + 1

            # Get caption if available
            caption_text = ""
            if table.captions:
                caption_texts = [
                    text_lookup.get(ref.cref, "") for ref in table.captions
                ]
                caption_text = " ".join(caption_texts).strip()

            # Get page number
            page_no = table.prov[0].page_no if table.prov else None

            # Convert table to HTML
            table_df.to_csv(f"table_{table_number}.csv", index=False)
            table_markdown = get_table_markdown(table_df)

            # Create chunk record for the table
            current_timestamp = datetime.now().isoformat()
            table_chunk_id = str(uuid.uuid4())
            table_chunk = {
                "id": table_chunk_id,
                "chunk_type": ChunkTypeEnum.PDFTable,
                "created_at": current_timestamp,
                "deleted_at": None,
                "table_markdown": table_markdown,
                "table_caption": caption_text,
                "metadata": {
                    "version": "1.0",
                    "chunk_order": table_number,
                    "chunked_by_engine": "docling",
                    "source": "pdf",
                    "page_number": page_no,
                },
            }

            table_chunks.append(table_chunk)
        except Exception as e:
            logger.error(f"Error processing table {table_ix + 1}: {str(e)}")

    return has_tables, table_chunks


def process_text(
    md_path,
    document_id,
    dataset_id,
    chunk_size=settings.DEFAULT_CHUNK_SIZE,
    chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP,
    chunking_config=None,
    split_id=None,
):
    """
    Process text content from markdown file.
    Refactored to reduce cyclomatic complexity by using helper functions.

    Args:
        md_path: Path to the markdown file
        document_id: ID of the document
        dataset_id: ID of the dataset
        chunk_size: Maximum size of text chunks (default: 10000)
        chunk_overlap: Overlap size between chunks (default: 2000)
        chunking_config: Additional configuration for text chunking
        split_id: Optional ID of the split if this is a split being processed

    Returns:
        Tuple of (text_chunks, all_text)
    """
    logger.info(f"Processing text with chunking... (split_id: {split_id})")

    # Setup configuration and initialize processing
    config = _prepare_text_chunking_config(chunk_size, chunk_overlap, chunking_config)
    text_chunks = []
    all_text = []
    client = get_openai_client()

    try:
        # Partition markdown into chunks
        chunks = _partition_markdown_file(md_path, config)

        # Process each chunk with helper function
        processing_state = _create_text_processing_state()

        for _chunk_idx, chunk in enumerate(chunks):
            _process_single_chunk(
                chunk,
                processing_state,
                text_chunks,
                all_text,
                document_id,
                dataset_id,
                split_id,
                client,
            )

        logger.info(
            f"Processed {processing_state['chunk_counter']} text chunks (split_id: {split_id})"
        )

    except Exception as e:
        logger.error(f"Error in text chunking: {str(e)}")

    return text_chunks, all_text


def _process_md_file(md_path: str) -> str:
    """Process a text file and return the extracted data"""
    chunk_size = 1000
    chunk_overlap = 200
    chunking_config = None
    config = _prepare_text_chunking_config(chunk_size, chunk_overlap, chunking_config)

    all_text = []

    try:
        # Partition markdown into chunks
        chunks = _partition_markdown_file(md_path, config)

        # Process each chunk with helper function
        processing_state = _create_text_processing_state()

        for _chunk_idx, chunk in enumerate(chunks):
            chunk_text = _process_single_chunk_lite(
                chunk,
                processing_state,
            )
            if chunk_text:
                all_text.append(chunk_text)

        return "\n".join(all_text)
    except Exception as e:
        logger.error(f"Error processing text file: {str(e)}")
        return f"Error processing text file: {str(e)}"


def _prepare_text_chunking_config(
    chunk_size: int, chunk_overlap: int, chunking_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Prepare the chunking configuration with defaults and user overrides."""
    default_config = {
        "chunking_strategy": "by_title",
        "max_characters": chunk_size,
        "combine_text_under_n_chars": chunk_overlap,
        "new_after_n_chars": chunk_size - chunk_overlap,
    }

    # Update with user-provided config if available
    if chunking_config:
        default_config.update(chunking_config)

    # Ensure include_page_breaks is set to avoid duplicate parameter error
    if "include_page_breaks" not in default_config:
        default_config["include_page_breaks"] = True

    return default_config


def _partition_markdown_file(md_path, config: Dict[str, Any]):
    """Partition the markdown file into chunks using unstructured."""
    return partition_md(
        filename=str(md_path),
        **config,
        extract_image_block_to_payload=False,
    )


def _create_text_processing_state() -> Dict[str, Any]:
    """Create initial state for text processing."""
    return {
        "chunk_counter": 0,
        "current_page": 1,
    }


def _should_skip_chunk(chunk) -> bool:
    """Determine if a chunk should be skipped during processing."""
    # Skip page breaks (handled separately)
    if isinstance(chunk, PageBreak):
        return True

    # Skip images and tables since we've already handled them
    element_type = type(chunk).__name__
    if element_type in ["Image", "Table"]:
        return True

    # Skip chunks without valid text content
    if not (hasattr(chunk, "text") and chunk.text and str(chunk.text).strip()):
        return True

    return False


def _extract_chunk_page_number(chunk, current_page: int) -> int:
    """Extract page number from chunk metadata or use current page."""
    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "page_number"):
        return chunk.metadata.page_number
    return current_page


def _create_text_chunk_record(
    chunk_text: str,
    chunk_counter: int,
    page_number: int,
    document_id,
    dataset_id,
    split_id: Optional[str],
    chunk_embedding: List[float],
) -> Dict[str, Any]:
    """Create a complete text chunk record with metadata."""
    current_timestamp = datetime.now().isoformat()
    text_chunk_id = str(uuid.uuid4())

    return {
        "id": text_chunk_id,
        "document_id": document_id,
        "chunk_type": ChunkTypeEnum.PDFText,
        "chunk_text": chunk_text,
        "chunk_embedding": chunk_embedding,
        "created_at": current_timestamp,
        "deleted_at": None,
        "metadata": {
            "version": "1.0",
            "chunk_order": chunk_counter,
            "file_id": str(document_id),
            "dataset_id": str(dataset_id),
            "chunked_by_engine": "unstructured",
            "source": "pdf",
            "page_number": page_number,
            "split_id": split_id,
        },
    }


def _process_single_chunk(
    chunk,
    processing_state: Dict[str, Any],
    text_chunks: List[Dict[str, Any]],
    all_text: List[str],
    document_id,
    dataset_id,
    split_id: Optional[str],
    client,
) -> None:
    """Process a single chunk and update the collections accordingly."""
    # Handle page breaks
    if isinstance(chunk, PageBreak):
        processing_state["current_page"] += 1
        return

    # Skip unwanted chunks
    if _should_skip_chunk(chunk):
        return

    # Process valid text chunk
    processing_state["chunk_counter"] += 1
    chunk_text = str(chunk.text).strip()

    # Collect text for document summary
    if processing_state["current_page"] <= settings.MAX_SUMMARY_PAGES:
        all_text.append(chunk_text)

    # Extract page number
    page_number = _extract_chunk_page_number(chunk, processing_state["current_page"])

    # Generate embedding and create chunk record
    chunk_embedding = generate_embedding_with_retry(chunk_text)
    text_chunk = _create_text_chunk_record(
        chunk_text,
        processing_state["chunk_counter"],
        page_number,
        document_id,
        dataset_id,
        split_id,
        chunk_embedding,
    )

    text_chunks.append(text_chunk)


def _process_single_chunk_lite(chunk, processing_state: Dict[str, Any]) -> None:
    """Process a single chunk"""
    # Handle page breaks
    if isinstance(chunk, PageBreak):
        processing_state["current_page"] += 1
        return

    # Skip unwanted chunks
    if _should_skip_chunk(chunk):
        return

    # Process valid text chunk
    processing_state["chunk_counter"] += 1
    chunk_text = str(chunk.text).strip()

    return chunk_text


def create_document_summary(all_text, num_pages, client=None):
    """
    Create a summary of the document.

    Args:
        all_text: List of text chunks
        num_pages: Total number of pages in document
        client: OpenAI client instance

    Returns:
        Document summary text
    """
    if not all_text:
        return "No text content available."

    # Combine first chunks of text, limiting to reasonable length
    combined_text = " ".join(all_text)
    combined_text = combined_text[:8000] if len(combined_text) > 8000 else combined_text

    if num_pages > settings.MAX_SUMMARY_PAGES:
        summary_prefix = f"This is a summary based on the first {settings.MAX_SUMMARY_PAGES} pages of a {num_pages}-page document: "
    else:
        summary_prefix = ""

    # Use AI to generate a better summary
    if client is None:
        client = get_openai_client()

    try:
        ai_summary = generate_text_summary(combined_text)
        return summary_prefix + ai_summary
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        # Fallback to simple summary
        simple_summary = combined_text[:500] + "..."
        return summary_prefix + simple_summary


def _initialize_processing_results() -> Dict[str, Any]:
    """Initialize the results dictionary for PDF processing."""
    return {
        "document": None,
        "chunks": {"text": [], "images": [], "tables": []},
        "success": False,
        "errors": [],
    }


def _setup_chunking_config(chunking_config: Dict[str, Any]) -> Tuple[int, int]:
    """Setup chunking configuration with defaults."""
    if not chunking_config:
        chunking_config = {}

    chunk_size = chunking_config.get("max_characters", settings.DEFAULT_CHUNK_SIZE)
    chunk_overlap = chunking_config.get("overlap", settings.DEFAULT_CHUNK_OVERLAP)
    return chunk_size, chunk_overlap


def _get_file_metadata(file_path: str) -> Tuple[Path, int, str]:
    """Extract file metadata including size and name."""
    file_path_obj = Path(file_path)
    file_size = file_path_obj.stat().st_size
    file_name = file_path_obj.name
    return file_path_obj, file_size, file_name


def _setup_temp_directories(temp_dir: str) -> None:
    """Create subdirectories for different content types."""
    text_dir = Path(temp_dir) / "text"
    tables_dir = Path(temp_dir) / "tables"
    images_dir = Path(temp_dir) / "images"
    for dir_path in [text_dir, tables_dir, images_dir]:
        dir_path.mkdir(exist_ok=True)


def _get_page_count(conv_result) -> int:
    """Extract page count from conversion result."""
    return (
        len(conv_result.document.pages) if hasattr(conv_result.document, "pages") else 0
    )


def _create_document_metadata(
    file_path: str,
    file_name: str,
    file_size: int,
    dataset_id: UUID,
    document_type: DocumentTypeEnum = DocumentTypeEnum.PDF,
) -> Dict[str, Any]:
    """Create document metadata dictionary with support for different document types."""
    # Determine MIME type based on document type
    mime_type_mapping = {
        DocumentTypeEnum.PDF: "application/pdf",
        DocumentTypeEnum.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        DocumentTypeEnum.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        DocumentTypeEnum.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        DocumentTypeEnum.Markdown: "text/markdown",
        DocumentTypeEnum.HTML: "text/html",
        DocumentTypeEnum.CSV: "text/csv",
    }

    return {
        "file_path": str(file_path),
        "file_name": file_name,
        "file_size": file_size,
        "dataset_id": str(dataset_id),
        "document_type": document_type.value,
        "mime_type": mime_type_mapping.get(document_type, "application/octet-stream"),
        "processing_time": datetime.now().isoformat(),
    }


def _update_existing_document(
    document: Document,
    file_id: UUID,
    dataset_id: UUID,
    summary: str,
    summary_embedding: Any,
    file_path: str,
    file_name: str,
    file_size: int,
    ingestion_id: Optional[UUID],
    document_type: DocumentTypeEnum = DocumentTypeEnum.PDF,
) -> None:
    """Update an existing document with new processing data."""
    document.file_id = file_id
    document.dataset_id = dataset_id
    document.document_type = document_type
    document.description = summary
    document.description_embedding = summary_embedding
    document.updated_at = datetime.now()

    # Update with new metadata while preserving existing values
    if not document.document_metadata:
        document.document_metadata = {}

    # Determine MIME type based on document type
    mime_type_mapping = {
        DocumentTypeEnum.PDF: "application/pdf",
        DocumentTypeEnum.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        DocumentTypeEnum.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        DocumentTypeEnum.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        DocumentTypeEnum.Markdown: "text/markdown",
        DocumentTypeEnum.HTML: "text/html",
        DocumentTypeEnum.CSV: "text/csv",
    }

    document.document_metadata.update(
        _create_document_metadata(
            file_path, file_name, file_size, dataset_id, document_type
        )
    )

    document.processing_status = DocumentProcessingStatusEnum.Processing
    document.file_path = str(file_path)
    document.file_size = file_size
    document.mime_type = mime_type_mapping.get(
        document_type, "application/octet-stream"
    )
    document.ingestion_id = ingestion_id


def _create_new_document(
    document_id: UUID,
    file_id: UUID,
    dataset_id: UUID,
    summary: str,
    summary_embedding: Any,
    file_path: str,
    file_name: str,
    file_size: int,
    ingestion_id: Optional[UUID],
    document_type: DocumentTypeEnum = DocumentTypeEnum.PDF,
) -> Document:
    """Create a new document record with support for different document types."""
    # Determine MIME type based on document type
    mime_type_mapping = {
        DocumentTypeEnum.PDF: "application/pdf",
        DocumentTypeEnum.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        DocumentTypeEnum.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        DocumentTypeEnum.PPTX: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        DocumentTypeEnum.Markdown: "text/markdown",
        DocumentTypeEnum.HTML: "text/html",
        DocumentTypeEnum.CSV: "text/csv",
    }

    return Document(
        id=document_id,
        file_id=file_id,
        dataset_id=dataset_id,
        document_type=document_type,
        description=summary,
        description_embedding=summary_embedding,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        document_metadata=_create_document_metadata(
            file_path, file_name, file_size, dataset_id, document_type
        ),
        processing_status=DocumentProcessingStatusEnum.Processing,
        file_path=str(file_path),
        file_size=file_size,
        mime_type=mime_type_mapping.get(document_type, "application/octet-stream"),
        ingestion_id=ingestion_id,
    )


def _get_or_create_document(
    session,
    document_id: UUID,
    dataset_id: UUID,
    file_id: UUID,
    summary: str,
    summary_embedding: Any,
    file_path: str,
    file_name: str,
    file_size: int,
    ingestion_id: Optional[UUID],
    document_type: DocumentTypeEnum = DocumentTypeEnum.PDF,
) -> Document:
    """Get existing document or create new one with support for different document types."""
    document = (
        session.query(Document)
        .filter(
            Document.id == document_id,
            Document.dataset_id == dataset_id,
            Document.file_id == file_id,
        )
        .first()
    )

    if document:
        _update_existing_document(
            document,
            file_id,
            dataset_id,
            summary,
            summary_embedding,
            file_path,
            file_name,
            file_size,
            ingestion_id,
            document_type,
        )
    else:
        document = _create_new_document(
            document_id,
            file_id,
            dataset_id,
            summary,
            summary_embedding,
            file_path,
            file_name,
            file_size,
            ingestion_id,
            document_type,
        )

    return document


def _process_text_chunks(
    session,
    text_chunks: List[Dict[str, Any]],
    document_id: UUID,
    split_id: Optional[str],
    results: Dict[str, Any],
) -> None:
    """Process and save text chunks to the database."""
    for chunk in text_chunks:
        try:
            chunk_obj = DocumentChunk(
                id=chunk["id"],
                document_id=document_id,
                chunk_type=ChunkTypeEnum.PDFText,
                chunk_text=chunk["chunk_text"],
                chunk_embedding=chunk["chunk_embedding"],
                chunk_metadata=chunk["metadata"],
                split_id=split_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            attributes.flag_modified(chunk_obj, "chunk_metadata")
            session.add(chunk_obj)

            results["chunks"]["text"].append(
                {
                    "id": chunk["id"],
                    "document_id": document_id,
                    "success": True,
                }
            )

        except Exception as chunk_e:
            logger.error(f"Error processing text chunk: {str(chunk_e)}")
            results["errors"].append(f"Text chunk error: {str(chunk_e)}")


def _process_table_chunks(
    session,
    table_chunks: List[Dict[str, Any]],
    document_id: UUID,
    split_id: Optional[str],
    results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Process and save table chunks to the database."""
    table_results = []

    for table_chunk in table_chunks:
        try:
            chunk_id = (
                uuid.UUID(table_chunk["id"])
                if isinstance(table_chunk["id"], str)
                else table_chunk["id"]
            )

            table_chunk_obj = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                chunk_type=ChunkTypeEnum.PDFTable,
                chunk_text=table_chunk["chunk_text"],
                chunk_embedding=table_chunk["chunk_embedding"],
                chunk_metadata=table_chunk["metadata"],
                split_id=split_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            attributes.flag_modified(table_chunk_obj, "chunk_metadata")
            session.add(table_chunk_obj)

            table_results.append(
                {
                    "id": str(chunk_id),
                    "document_id": document_id,
                    "page_number": table_chunk["metadata"].get("page_number"),
                    "success": True,
                }
            )

        except Exception as table_e:
            logger.error(f"Error processing table chunk: {str(table_e)}")
            results["errors"].append(f"Table chunk error: {str(table_e)}")

    return table_results


def _handle_database_error(
    session, document_id: UUID, error: Exception, results: Dict[str, Any]
) -> None:
    """Handle database errors during document processing."""
    logger.error(f"Database error during document processing: {str(error)}")
    session.rollback()
    results["errors"].append(f"Database error: {str(error)}")

    # Try to update document status to failed if it exists
    try:
        existing_doc = session.get(Document, document_id)
        if existing_doc:
            existing_doc.processing_status = DocumentProcessingStatusEnum.Failed
            if not existing_doc.document_metadata:
                existing_doc.document_metadata = {}
            existing_doc.document_metadata["error"] = str(error)
            existing_doc.updated_at = datetime.now()
            session.commit()
    except Exception as e:
        logger.error(f"Failed to update document status: {str(e)}")


def _finalize_direct_document_results(
    results: Dict[str, Any],
    document: Document,
    document_id: UUID,
    has_images: bool,
    image_chunks: List[Dict[str, Any]],
    table_chunks: List[Dict[str, Any]],
    text_chunks: List[Dict[str, Any]],
) -> None:
    """Finalize results for direct document ingestion."""
    if not results["errors"]:
        results["success"] = True
        results["has_images"] = has_images
        results["image_count"] = len(image_chunks)
        results["table_count"] = len(table_chunks)
        results["text_chunk_count"] = len(text_chunks)

        # Include document details for direct ingestion
        results["document"] = {
            "id": str(document_id),
            "type": "PDF",
            "description": document.description if document else None,
            "created_at": (
                document.created_at.isoformat()
                if document and document.created_at
                else None
            ),
            "updated_at": (
                document.updated_at.isoformat()
                if document and document.updated_at
                else None
            ),
            "metadata": document.document_metadata if document else {},
            "status": "success",
            "processing_type": "direct",
        }

        logger.info(
            f"Direct document processing completed successfully for document {document_id}"
        )
    else:
        results["success"] = False
        results["has_images"] = False
        results["document"] = {
            "id": str(document_id),
            "status": "failed",
            "processing_type": "direct",
            "errors": results["errors"],
        }
        logger.error(
            f"Direct document processing failed for document {document_id}: {results['errors']}"
        )


def _finalize_split_results(
    results: Dict[str, Any],
    document_id: UUID,
    split_id: str,
    has_images: bool,
    image_chunks: List[Dict[str, Any]],
    table_chunks: List[Dict[str, Any]],
    text_chunks: List[Dict[str, Any]],
) -> None:
    """Finalize results for split document ingestion."""
    if not results["errors"]:
        results["success"] = True
        results["has_images"] = has_images
        results["image_count"] = len(image_chunks)
        results["table_count"] = len(table_chunks)
        results["text_chunk_count"] = len(text_chunks)

        # Include split-specific details
        results["split"] = {
            "id": split_id,
            "document_id": str(document_id),
            "status": "success",
            "processing_type": "split",
            "chunks_created": {
                "text": len(text_chunks),
                "tables": len(table_chunks),
                "images": len(image_chunks),
            },
        }

        logger.info(
            f"Split processing completed successfully for split {split_id} of document {document_id}"
        )
    else:
        results["success"] = False
        results["has_images"] = False
        results["split"] = {
            "id": split_id,
            "document_id": str(document_id),
            "status": "failed",
            "processing_type": "split",
            "errors": results["errors"],
        }
        logger.error(
            f"Split processing failed for split {split_id}: {results['errors']}"
        )


def _should_generate_document_description(split_id: Optional[str]) -> bool:
    """
    Determine if we should generate document description for this split.
    Only generate for the first split (index 0) in split-based ingestion.
    """
    if not split_id:
        return False

    try:
        # Get split information to check index
        with get_session() as session:
            split_record = (
                session.query(FileSplit).filter(FileSplit.id == split_id).first()
            )
            if split_record and split_record.split_index is not None:
                is_first_split = split_record.split_index == 0
                logger.info(
                    f"Split {split_id} has index {split_record.split_index}, is_first_split: {is_first_split}"
                )
                return is_first_split

        logger.warning(f"Could not determine split index for {split_id}")
        return False

    except Exception as e:
        logger.error(f"Error checking split index for {split_id}: {str(e)}")
        return False


def _generate_document_description_from_split(
    all_text: List[str], num_pages: int, split_id: str, client=None
) -> Tuple[str, List[float]]:
    """
    Generate document description and embedding from first split content.
    Uses first 10 pages if split has more than 10 pages, otherwise uses whole split.

    Args:
        all_text: List of text chunks from the split
        num_pages: Number of pages in the split
        split_id: ID of the split
        client: OpenAI client instance

    Returns:
        Tuple of (description, description_embedding)
    """
    logger.info(
        f"Generating document description from first split {split_id} ({num_pages} pages)"
    )

    if not all_text:
        fallback_description = (
            f"Document processed via split ingestion (first split {split_id})"
        )
        fallback_embedding = generate_embedding(fallback_description)
        return fallback_description, fallback_embedding

    try:
        # Determine how many pages to use for summary (max 10 pages from first split)
        max_pages_for_summary = min(
            num_pages, 10
        )  # Use settings.MAX_SUMMARY_PAGES or hardcode 10

        # Collect text from first N pages only
        summary_text = []
        pages_used = 0

        # Since all_text contains chunks that were already limited by MAX_SUMMARY_PAGES
        # in the process_text function, we can use all available text
        for text_chunk in all_text:
            summary_text.append(text_chunk)
            pages_used += 1  # Approximate - each chunk might not be exactly one page

            # Stop if we have enough content (approximate page limit)
            if pages_used >= max_pages_for_summary:
                break

        # Combine text for summary generation
        combined_text = " ".join(summary_text)

        # Limit total text length for API call efficiency
        if len(combined_text) > 8000:
            combined_text = combined_text[:8000] + "..."

        # Generate description using AI
        if client is None:
            client = get_openai_client()

        # Create summary with context about split processing
        if num_pages > 10:
            summary_prefix = f"This document summary is based on the first {max_pages_for_summary} pages of the first section: "
        else:
            summary_prefix = f"This document summary is based on the first section ({num_pages} pages): "

        ai_summary = generate_text_summary(combined_text)
        full_description = summary_prefix + ai_summary

        # Generate embedding for the description
        description_embedding = generate_embedding_with_retry(full_description)

        logger.info(
            f"Generated description from split {split_id}: {len(full_description)} characters, using ~{pages_used} pages"
        )
        return full_description, description_embedding

    except Exception as e:
        logger.error(f"Error generating description from split {split_id}: {str(e)}")
        # Fallback description
        fallback_description = f"Document content from first split with {num_pages} pages (split {split_id})"
        fallback_embedding = generate_embedding(fallback_description)
        return fallback_description, fallback_embedding


def _update_document_with_description(
    session,
    document_id: UUID,
    dataset_id: UUID,
    file_id: UUID,
    description: str,
    description_embedding: List[float],
    split_id: str,
) -> None:
    """
    Update the Document record with description and embedding from first split.
    """
    try:
        # Get the document record
        document = (
            session.query(Document)
            .filter(
                Document.id == document_id,
                Document.dataset_id == dataset_id,
                Document.file_id == file_id,
            )
            .first()
        )

        if not document:
            logger.warning(f"Document {document_id} not found for description update")
            return

        # Update description and embedding
        document.description = description
        document.description_embedding = description_embedding
        document.updated_at = datetime.now()

        # Preserve existing metadata including image_summary
        existing_metadata = {}
        if document.document_metadata:
            existing_metadata = document.document_metadata.copy()

        # Update with new description metadata while preserving existing data
        existing_metadata.update(
            {
                "original_pdf_summary": description,
                "original_pdf_summary_preserved": True,
                "description_generated_from_split": True,
                "description_source_split_id": split_id,
                "description_generated_at": datetime.now().isoformat(),
                "ingestion_type": "split_based",
            }
        )

        # Assign back to document
        document.document_metadata = existing_metadata

        attributes.flag_modified(document, "document_metadata")
        session.add(document)

        logger.info(
            f"Updated document {document_id} with description from split {split_id}, "
            f"preserved existing metadata keys: {list(existing_metadata.keys())}"
        )

    except Exception as e:
        logger.error(
            f"Error updating document {document_id} with description: {str(e)}"
        )
        session.rollback()
        raise
