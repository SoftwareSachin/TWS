import base64
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.document_chunk_model import ChunkTypeEnum, DocumentChunk
from app.models.document_model import (
    Document,
    DocumentProcessingStatusEnum,
    DocumentTypeEnum,
)
from app.models.file_ingestion_model import FileIngestion, FileIngestionStatusType
from app.models.file_model import File
from app.utils.ingestion_utils import publish_ingestion_status
from app.utils.openai_utils import generate_embedding, get_openai_client


@celery.task(name="tasks.image_ingestion_task", bind=True, acks_late=True)
def image_ingestion_task(
    self,
    file_id: str,
    file_path: str,
    ingestion_id: str,
    dataset_id: str,
    user_id: uuid,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Process an image file and extract text using OCR, analyze content using AI.

    Args:
        file_id: ID of the file being processed
        file_path: Path to the image file
        ingestion_id: ID of the current ingestion batch
        dataset_id: ID of the dataset
        metadata: Additional metadata to store with the document
        user_id: ID of the user who initiated the ingestion (for WebSocket notifications)

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Processing image document: {file_path}")
    logger.info(metadata)

    with SyncSessionLocal() as db:
        try:
            # Update initial ingestion status
            _update_file_ingestion_status(
                db=db,
                file_id=file_id,
                ingestion_id=ingestion_id,
                status=FileIngestionStatusType.Processing,
                task_id=self.request.id,
            )

            # Create and save initial document
            document = _create_document(
                file_id=file_id,
                dataset_id=dataset_id,
                file_path=file_path,
                task_id=self.request.id,
                ingestion_id=ingestion_id,
                metadata=metadata,
            )
            db.add(document)
            db.commit()
            db.refresh(document)

            # Update document metadata
            image_metadata = _extract_image_metadata(file_path)
            merged_metadata = {**(document.document_metadata or {}), **image_metadata}
            document.document_metadata = merged_metadata
            document = db.merge(document)
            db.commit()
            db.refresh(document)

            # Update to extraction status
            document.processing_status = DocumentProcessingStatusEnum.Extracting
            db.commit()

            # Process image content
            ocr_text, ocr_chunks = _extract_ocr_text(file_path)
            description, detected_objects = _analyze_image(
                file_path, Path(file_path).name, ocr_text
            )
            description_embedding = generate_embedding(description)

            # Update document with extracted content
            document.description = description
            document.description_embedding = description_embedding
            document.processing_status = (
                DocumentProcessingStatusEnum.ExtractionCompleted
            )
            db.commit()

            # Create and save chunks
            chunk_ids = []

            # Add description chunk
            desc_chunk = _create_description_chunk(
                document.id, description, description_embedding, file_path
            )
            db.add(desc_chunk)
            chunk_ids.append(desc_chunk.id)

            # Add OCR chunks
            ocr_db_chunks = _create_ocr_chunks(document.id, ocr_chunks)
            for chunk in ocr_db_chunks:
                db.add(chunk)
                chunk_ids.append(chunk.id)

            # Add object detection chunks
            obj_db_chunks = _create_object_chunks(document.id, detected_objects)
            for chunk in obj_db_chunks:
                db.add(chunk)
                chunk_ids.append(chunk.id)

            # Commit all chunks
            db.commit()

            # Update final statuses
            document.processing_status = DocumentProcessingStatusEnum.Success
            document.processed_at = datetime.utcnow()
            db.commit()

            _update_file_ingestion_status(
                db=db,
                file_id=file_id,
                ingestion_id=ingestion_id,
                status=FileIngestionStatusType.Success,
            )

            # Send success notification
            file_info = db.query(File).filter(File.id == uuid.UUID(file_id)).first()
            file_name = file_info.filename if file_info else Path(file_path).name

            if user_id:
                _send_success_notification(
                    user_id=user_id,
                    file_id=file_id,
                    file_name=file_name,
                    document_id=document.id,
                    chunk_count=len(chunk_ids),
                    ingestion_id=ingestion_id,
                    task_id=self.request.id,
                )

            return {
                "success": True,
                "document_id": document.id,
                "file_id": file_id,
                "dataset_id": dataset_id,
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
            }

        except Exception as e:
            logger.error(
                f"Error processing image file {file_path}: {str(e)}", exc_info=True
            )

            # Update document status if it exists
            if "document" in locals():
                document.processing_status = DocumentProcessingStatusEnum.Failed
                document.error_message = str(e)
                document.processed_at = datetime.utcnow()
                db.commit()

            # Update ingestion status
            _update_file_ingestion_status(
                db=db,
                file_id=file_id,
                ingestion_id=ingestion_id,
                status=FileIngestionStatusType.Failed,
                error_message=str(e),
            )

            # Send error notification
            file_info = db.query(File).filter(File.id == uuid.UUID(file_id)).first()
            file_name = file_info.filename if file_info else Path(file_path).name

            if user_id:
                _send_error_notification(
                    user_id=uuid.UUID(user_id),
                    file_id=file_id,
                    file_name=file_name,
                    error=str(e),
                    ingestion_id=ingestion_id,
                    task_id=self.request.id,
                )

            return {"success": False, "file_id": file_id, "error": str(e)}


def _create_document(
    file_id: str,
    dataset_id: str,
    file_path: str,
    task_id: str,
    ingestion_id: str,
    metadata: Dict[str, Any] = None,
) -> Document:
    """Create initial document record"""
    document_id = str(uuid.uuid4())
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
    mime_type = _get_mime_type(file_path)

    document = Document(
        id=document_id,
        file_id=file_id,
        dataset_id=dataset_id,
        document_type=DocumentTypeEnum.Image,
        processing_status=DocumentProcessingStatusEnum.Processing,
        file_path=file_path,
        file_size=file_size,
        mime_type=mime_type,
        document_metadata=metadata or {},
        task_id=task_id,
        ingestion_id=ingestion_id,
    )
    return document


def _create_description_chunk(
    document_id: str,
    description: str,
    description_embedding: List[float],
    file_path: str,
) -> DocumentChunk:
    """Create chunk for image description"""
    return DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=document_id,
        chunk_type=ChunkTypeEnum.ImageDescription,
        chunk_text=description,
        chunk_embedding=description_embedding,
        chunk_metadata={
            "chunk_order": 0,
            "chunked_by_engine": "gpt-4o",
            "file_path": file_path,
        },
    )


def _create_ocr_chunks(
    document_id: str, ocr_chunks: List[Dict[str, Any]]
) -> List[DocumentChunk]:
    """Create chunks for OCR text"""
    chunks = []
    for i, chunk_info in enumerate(ocr_chunks):
        chunk = DocumentChunk(
            id=str(uuid.uuid4()),
            document_id=document_id,
            chunk_type=ChunkTypeEnum.ImageText,
            chunk_text=chunk_info["text"],
            chunk_embedding=generate_embedding(chunk_info["text"]),
            chunk_metadata={
                "coordinates": chunk_info.get("coordinates"),
                "confidence": chunk_info.get("confidence"),
                "chunk_order": i,
                "chunked_by_engine": "pytesseract",
            },
        )
        chunks.append(chunk)
    return chunks


def _create_object_chunks(
    document_id: str, detected_objects: List[Dict[str, Any]]
) -> List[DocumentChunk]:
    """Create a single chunk for all detected objects"""
    if not detected_objects:
        return []

    # Join all object names with semicolons
    object_names = "; ".join(
        obj.get("object_name") if isinstance(obj, dict) else str(obj)
        for obj in detected_objects
    )

    chunk = DocumentChunk(
        id=str(uuid.uuid4()),
        document_id=document_id,
        chunk_type=ChunkTypeEnum.ImageObject,
        chunk_text=object_names,
        chunk_embedding=generate_embedding(object_names),
        chunk_metadata={
            "chunk_order": 0,
            "chunked_by_engine": "gpt-4o",
        },
    )
    return [chunk]


def _send_success_notification(
    user_id: uuid.UUID,
    file_id: str,
    file_name: str,
    document_id: str,
    chunk_count: int,
    ingestion_id: str,
    task_id: str,
):
    """Send websocket notification for successful processing"""
    result_data = {
        "file_id": file_id,
        "file_name": file_name,
        "document_id": document_id,
        "status": FileIngestionStatusType.Success.value,
        "success": True,
        "chunk_count": chunk_count,
        "document_type": DocumentTypeEnum.Image.value,
        "ingestion_id": ingestion_id,
        "finished_at": datetime.utcnow().isoformat(),
        "task_id": task_id,
    }
    publish_ingestion_status(
        user_id=user_id,
        ingestion_id=ingestion_id,
        task_id=task_id,
        ingestion_result=result_data,
    )


def _send_error_notification(
    user_id: uuid.UUID,
    file_id: str,
    file_name: str,
    error: str,
    ingestion_id: str,
    task_id: str,
):
    """Send websocket notification for failed processing"""
    error_data = {
        "file_id": file_id,
        "file_name": file_name,
        "status": FileIngestionStatusType.Failed.value,
        "success": False,
        "error": error,
        "ingestion_id": ingestion_id,
        "finished_at": datetime.utcnow().isoformat(),
        "task_id": task_id,
    }
    publish_ingestion_status(
        user_id=user_id,
        ingestion_id=ingestion_id,
        task_id=task_id,
        ingestion_result=error_data,
    )


def _get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension"""
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".jpg" or extension == ".jpeg":
        return "image/jpeg"
    elif extension == ".png":
        return "image/png"
    else:
        return "application/octet-stream"


def _update_file_ingestion_status(
    db: Session,
    file_id: str,
    ingestion_id: str,
    status: FileIngestionStatusType,
    task_id: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """Update file ingestion status in database"""
    ingestion = (
        db.query(FileIngestion)
        .filter(
            FileIngestion.file_id == file_id, FileIngestion.ingestion_id == ingestion_id
        )
        .first()
    )

    if ingestion:
        ingestion.status = status
        if task_id:
            ingestion.task_id = task_id
        if error_message:
            # Safely handle error message - log it but don't try to set it if field doesn't exist
            logger.error(f"Ingestion error for file {file_id}: {error_message}")
            try:
                # Try to set error_message field if it exists
                ingestion.error_message = error_message
            except (ValueError, AttributeError):
                # If field doesn't exist, store it in a metadata field if available
                try:
                    if hasattr(ingestion, "metadata") and isinstance(
                        ingestion.metadata, dict
                    ):
                        ingestion.metadata["error_message"] = error_message
                    else:
                        # If no metadata field, just log the error
                        logger.warning(
                            "Could not store error message in FileIngestion model"
                        )
                except Exception as e:
                    logger.warning(f"Failed to store error message: {str(e)}")

        ingestion.updated_at = datetime.utcnow()
        if status in [
            FileIngestionStatusType.Success,
            FileIngestionStatusType.Failed,
            FileIngestionStatusType.Exception,
        ]:
            ingestion.finished_at = datetime.utcnow()
        db.commit()


def _extract_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract detailed metadata and features from image file using PIL and OpenCV"""
    try:
        # Basic metadata with PIL
        img = Image.open(image_path)
        width, height = img.size
        format = img.format
        mode = img.mode

        # Get additional metadata if available
        info = {}
        for key, value in img.info.items():
            if isinstance(value, (str, int, float, bool)):
                info[key] = value

        # Enhanced feature extraction with OpenCV
        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            # Calculate average color
            avg_color = np.average(np.average(cv_img, axis=0), axis=0).tolist()

            # Detect edges and lines
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
            )
            line_count = 0 if lines is None else len(lines)

            # Prepare feature summary
            feature_summary = f"""
            Image dimensions: {width}x{height}
            Average RGB color: {avg_color}
            Number of straight lines detected: {line_count}
            """

            # Add interpretation hints based on image features
            if line_count > 10:
                feature_summary += "This image likely contains a chart, diagram or structured content.\n"

            # Add file size information
            file_size = os.path.getsize(image_path)
            feature_summary += f"File size: {file_size} bytes\n"

            return {
                "width": width,
                "height": height,
                "format": format,
                "mode": mode,
                "info": info,
                "avg_color": avg_color,
                "line_count": line_count,
                "feature_summary": feature_summary.strip(),
                "file_size": file_size,
            }
        else:
            # Return basic metadata if OpenCV processing fails
            return {
                "width": width,
                "height": height,
                "format": format,
                "mode": mode,
                "info": info,
            }
    except Exception as e:
        logger.error(f"Error extracting image metadata: {str(e)}")
        return {}


def _extract_ocr_text(image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text from image using OCR with enhanced handling"""
    try:
        # Try with PIL first
        img = Image.open(image_path)
        full_text = pytesseract.image_to_string(img).strip()
        logger.debug("OCR text extracted")

        # Also try with OpenCV for potentially better results in some cases
        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            cv_text = pytesseract.image_to_string(gray).strip()

            # Use the longer text (often has better results)
            if len(cv_text) > len(full_text):
                full_text = cv_text

        # Create chunks if we have meaningful text
        chunks = []
        if len(full_text) > 5:  # Minimal text requirement
            chunks.append({"text": full_text, "coordinates": None, "confidence": 1.0})

        # Return both the full text and the chunks
        return full_text, chunks
    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        return "", []


def _analyze_image_description(
    image_path: str, filename: str, feature_text: str
) -> str:
    """Analyze image using AI to get a comprehensive description"""
    try:
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Get OpenAI client
        client = get_openai_client()

        # Create a prompt focused on image analysis
        analysis_prompt = f"""
        Filename: {filename}
        {feature_text}

        Based on the extracted features, please provide a comprehensive analysis of what this image likely contains.

        If it appears to be a chart, graph, or other data visualization:
        - Identify the probable chart type (bar chart, line graph, pie chart, scatter plot, etc.).
        - Describe the likely subject of the chart: what is being measured or compared.
        - Specify what the X-axis and Y-axis represent (including units if available).
        - Extract **all data points** shown in the chart:
        - **If there are multiple values in a single bar/point/segment (e.g., stacked bars, grouped bars, or subcategories), extract them separately.**
        - **If exact values are not labeled, estimate based on axis markers.**
        - **Provide the extracted data in a structured table format.**
        - Table columns should dynamically match the information available (e.g., Category, Value 1, Value 2, etc.).
        - **Avoid using qualitative terms** like "high", "low", "increase", "spike", etc. Always provide numeric or factual information.

        If it appears to be a photograph or non-chart image:
        - Identify the probable subject matter or scene.
        - List key objects or elements present.
        - Mention any relevant contextual information inferred from the image features.

        Additionally, for all image types:
        - **Count and report** the number of each distinct object detected (e.g., 3 cars, 2 trees, 5 people).
        - **Accuracy in counting is important.**

        Finally, if any values are estimated rather than explicitly shown, note that clearly at the end.
        """

        # Call GPT-4o with vision capabilities
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing images and providing detailed descriptions.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=1000,
        )

        analysis = response.choices[0].message.content.strip()
        logger.info(f"Successfully generated image analysis ({len(analysis)} chars)")
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing image description: {str(e)}")
        return f"Error analyzing image: {str(e)}"


def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise


def _create_object_detection_prompt(filename: str, feature_text: str) -> str:
    """Create the prompt for object detection"""
    return f"""
    Filename: {filename}
    {feature_text}

        Based on the visual features extracted from this image, perform an object detection analysis.

        Try to identify and list prominent objects in the image (such as axes, legends, bars, labels, people, tools, charts, etc.).

        Please return a JSON array where each object follows this structure:
    {{
        "object_name": "string",
        "confidence": float (between 0 and 1),
        "coordinates": [x_min, y_min, x_max, y_max]
    }}

    Example:
    [
        {{
        "object_name": "Person",
        "confidence": 0.94,
        "coordinates": [32, 45, 180, 300]
        }},
        ...
    ]

    If no object is detected, return an empty array: []"""


def _parse_object_detection_response(content: str) -> List[Dict[str, Any]]:
    """Parse the GPT response for object detection"""
    # Clean up markdown code blocks if present
    if content.startswith("```") and content.endswith("```"):
        content = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content).group(1)

    try:
        # Try parsing the JSON response
        detected_objects = json.loads(content)

        # Handle case where response might be wrapped in an object
        if isinstance(detected_objects, dict) and "objects" in detected_objects:
            detected_objects = detected_objects["objects"]
        elif not isinstance(detected_objects, list):
            detected_objects = []

        return detected_objects

    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Failed to parse object detection response as JSON: {str(e)}")

        # Fallback: Try to extract array using regex
        array_match = re.search(r"\[(.*?)\]", content, re.DOTALL)
        if array_match:
            array_text = array_match.group(1)
            # Extract strings from the array format
            detected_objects = [
                item.strip().strip("\"'")
                for item in re.findall(r'"([^"]*?)"|\'([^\']*?)\'', array_text)
                if item
            ]
            return detected_objects

        return []


def _detect_objects_in_image(
    image_path: str, filename: str, feature_text: str
) -> List[Dict[str, Any]]:
    """Detect objects in an image using AI"""
    try:
        # Get base64 encoded image
        encoded_image = _encode_image_to_base64(image_path)

        # Get OpenAI client
        client = get_openai_client()

        # Create detection prompt
        detection_prompt = _create_object_detection_prompt(filename, feature_text)

        # Call GPT-4o with vision capabilities
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at detecting and identifying objects in images. Return a JSON array of object names.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=800,
        )

        content = response.choices[0].message.content.strip()

        # Parse and return the results
        detected_objects = _parse_object_detection_response(content)
        logger.info(f"Successfully detected {len(detected_objects)} objects")
        return detected_objects

    except Exception as e:
        logger.error(f"Error detecting objects in image: {str(e)}")
        return []


def _analyze_image(
    image_path: str, filename: str, ocr_text: str
) -> Tuple[str, List[str]]:
    """Analyze image using AI to get description and detect objects"""
    try:
        # Extract image features to enhance analysis
        feature_text = _extract_image_features(image_path)
        feature_text = f"OCR Text extracted: {ocr_text}\n{feature_text}"

        # Get image description using separated function
        description = _analyze_image_description(image_path, filename, feature_text)

        # Detect objects using separated function
        detected_objects = _detect_objects_in_image(image_path, filename, feature_text)

        return description, detected_objects

    except Exception as e:
        logger.error(f"Error in image analysis process: {str(e)}")
        return f"Error analyzing image: {str(e)}", []


def _extract_image_features(image_path: str) -> str:
    """Extract detailed image features and return them as formatted text"""
    try:
        img = Image.open(image_path)
        width, height = img.size

        # Extract features using OpenCV for more advanced analysis
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return f"Image dimensions: {width}x{height}"

        # Calculate average color
        avg_color = np.average(np.average(cv_img, axis=0), axis=0)

        # Detect edges and lines (useful for charts, diagrams)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )
        line_count = 0 if lines is None else len(lines)

        feature_text = f"""
        Text extracted from image using OCR appears above.

        Image features:
        - Image dimensions: {width}x{height}
        - Average RGB color: {avg_color}
        - Number of straight lines detected: {line_count}
        """

        if line_count > 10:
            feature_text += "This image likely contains a chart or diagram.\n"

        # Add file size
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            feature_text += f"- File size: {file_size} bytes\n"

        return feature_text.strip()
    except Exception as e:
        logger.error(f"Error extracting image features: {str(e)}")
        return "Error extracting image features"
