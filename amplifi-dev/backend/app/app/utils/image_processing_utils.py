import base64
import gc
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from app.be_core.config import settings
from app.be_core.logger import logger


async def encode_image(image_path):
    """Convert image to base64 for sending to GPT-4o."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _is_valid_ocr_text(text: str, confidence: int) -> bool:
    """
    Check if OCR text is valid based on content and confidence.

    Args:
        text: OCR extracted text
        confidence: OCR confidence score

    Returns:
        True if text is valid, False otherwise
    """
    return text.strip() and confidence >= 5  # OCR confidence threshold


def _calculate_scaled_coordinates(
    ocr_data: Dict[str, Any], index: int, scale_x: float, scale_y: float
) -> Tuple[int, int, int, int]:
    """
    Calculate scaled coordinates for OCR text.

    Args:
        ocr_data: OCR data dictionary
        index: Index of the current text element
        scale_x: X-axis scaling factor
        scale_y: Y-axis scaling factor

    Returns:
        Tuple of (x, y, w, h) coordinates
    """
    x = int(ocr_data["left"][index] * scale_x)
    y = int(ocr_data["top"][index] * scale_y)
    w = int(ocr_data["width"][index] * scale_x)
    h = int(ocr_data["height"][index] * scale_y)
    return x, y, w, h


def _update_bounding_box(
    current_coords: Optional[List[int]], x: int, y: int, w: int, h: int
) -> List[int]:
    """
    Update bounding box coordinates to include new text element.

    Args:
        current_coords: Current bounding box coordinates [x1, y1, x2, y2] or None
        x, y, w, h: New text element coordinates

    Returns:
        Updated bounding box coordinates
    """
    if not current_coords:
        return [x, y, x + w, y + h]
    else:
        # Expand bounding box
        return [
            min(current_coords[0], x),
            min(current_coords[1], y),
            max(current_coords[2], x + w),
            max(current_coords[3], y + h),
        ]


def is_end_of_line(ocr_data: Dict[str, Any], current_index: int) -> bool:
    """
    Check if current OCR element is at the end of a line.

    Args:
        ocr_data: OCR data dictionary
        current_index: Current index in OCR data

    Returns:
        True if at end of line, False otherwise
    """
    return (
        current_index + 1 >= len(ocr_data["text"])
        or ocr_data["line_num"][current_index + 1]
        != ocr_data["line_num"][current_index]
    )


def _create_ocr_chunk(
    current_text: List[str], current_conf: List[int], current_coords: List[int]
) -> Optional[Dict[str, Any]]:
    """
    Create an OCR chunk from accumulated text and metadata.

    Args:
        current_text: List of text elements
        current_conf: List of confidence scores
        current_coords: Bounding box coordinates

    Returns:
        OCR chunk dictionary or None if no valid text
    """
    chunk_text = " ".join(current_text).strip()
    if not chunk_text:
        return None

    avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0
    return {
        "text": chunk_text,
        "coordinates": current_coords,
        "confidence": avg_conf / 100.0,  # Normalize to 0-1 range
    }


def _process_ocr_data_to_chunks(
    ocr_data: Dict[str, Any], scale_x: float, scale_y: float
) -> List[Dict[str, Any]]:
    """
    Process OCR data to create meaningful text chunks.

    Args:
        ocr_data: OCR data dictionary
        scale_x: X-axis scaling factor
        scale_y: Y-axis scaling factor

    Returns:
        List of OCR chunk dictionaries
    """
    chunks = []
    current_text = []
    current_conf = []
    current_coords = None

    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i]
        conf = ocr_data["conf"][i]

        # Skip empty text or very low confidence
        if not _is_valid_ocr_text(text, conf):
            continue

        # Get coordinates, adjusted for original image size
        x, y, w, h = _calculate_scaled_coordinates(ocr_data, i, scale_x, scale_y)
        current_coords = _update_bounding_box(current_coords, x, y, w, h)

        current_text.append(text)
        current_conf.append(conf)

        # If we reach end of line or paragraph, create a chunk
        if is_end_of_line(ocr_data, i):
            chunk = _create_ocr_chunk(current_text, current_conf, current_coords)
            if chunk:
                chunks.append(chunk)

            # Reset for next chunk
            current_text = []
            current_conf = []
            current_coords = None

    return chunks


def get_ocr_data_with_scaling(img: Image.Image) -> Tuple[Dict[str, Any], float, float]:
    """
    Get OCR data with appropriate scaling for large images.

    Args:
        img: PIL Image object

    Returns:
        Tuple of (OCR data dict, scale_x, scale_y)
    """
    if max(img.size) > 2000:
        # Scale down for OCR data extraction to improve memory usage
        img_small = img.copy()
        img_small.thumbnail((2000, 2000), Image.LANCZOS)
        ocr_data = pytesseract.image_to_data(
            img_small, output_type=pytesseract.Output.DICT
        )
        # Adjust coordinates to original size
        scale_x = img.width / img_small.width
        scale_y = img.height / img_small.height
        del img_small
        return ocr_data, scale_x, scale_y
    else:
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        return ocr_data, 1.0, 1.0


def extract_basic_ocr_text(img: Image.Image) -> str:
    """
    Extract basic OCR text from image.

    Args:
        img: PIL Image object

    Returns:
        Extracted text string
    """
    full_text = pytesseract.image_to_string(img).strip()
    logger.debug("OCR text extracted")
    return full_text


def process_image_in_tiles(
    image_path: str, tile_size: int = 1024
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process a large image in tiles to limit memory usage.

    Args:
        image_path: Path to the image file
        tile_size: Size of each processing tile

    Returns:
        Tuple of (full OCR text, list of text chunks with coordinates)
    """
    try:
        full_text = []
        text_chunks = []

        # Process image in tiles for OCR
        with Image.open(image_path) as img:
            width, height = img.size

            # Create a small overlap between tiles to prevent text splitting
            overlap = 50
            effective_tile_size = tile_size - overlap

            for y in range(0, height, effective_tile_size):
                for x in range(0, width, effective_tile_size):
                    # Extract and process tile
                    tile_height = min(tile_size, height - y)
                    tile_width = min(tile_size, width - x)

                    # Skip very small tiles
                    if tile_height < 20 or tile_width < 20:
                        continue

                    tile = img.crop((x, y, x + tile_width, y + tile_height))

                    # Get OCR data for tile
                    try:
                        tile_text = pytesseract.image_to_string(tile).strip()

                        # Only add non-empty text
                        if tile_text:
                            full_text.append(tile_text)
                            text_chunks.append(
                                {
                                    "text": tile_text,
                                    "coordinates": [
                                        x,
                                        y,
                                        x + tile_width,
                                        y + tile_height,
                                    ],
                                    "confidence": 0.8,  # Estimated confidence
                                }
                            )
                    except Exception as e:
                        logger.warning(f"OCR error on tile at ({x},{y}): {str(e)}")

                    # Force release of tile memory
                    del tile

                # Force garbage collection after each row
                gc.collect()

        return " ".join(full_text), text_chunks

    except Exception as e:
        logger.error(f"Error in tile processing: {str(e)}")
        return "", []


def process_image_file_to_extract_data(file_path: str) -> str:
    # Use tiled processing for OCR to improve memory efficiency
    if os.path.getsize(file_path) > 5 * 1024 * 1024:  # 5MB threshold
        ocr_text, _ = process_image_in_tiles(file_path)
    else:
        ocr_text = extract_ocr_text(file_path)

    return ocr_text


def _should_use_tiled_processing(image_path: str) -> bool:
    """
    Check if image is too large for direct OCR processing.

    Args:
        image_path: Path to the image file

    Returns:
        True if should use tiled processing, False otherwise
    """
    file_size = os.path.getsize(image_path)
    return file_size > 10 * 1024 * 1024  # 10MB threshold


def _create_fallback_chunk(full_text: str) -> List[Dict[str, Any]]:
    """
    Create fallback chunk when no chunks were created from OCR data.

    Args:
        full_text: Full text extracted from image

    Returns:
        List containing single fallback chunk
    """
    return [
        {
            "text": full_text,
            "coordinates": [0, 0, 0, 0],  # No coordinates available
            "confidence": 0.5,  # Default confidence when unknown
        }
    ]


def extract_ocr_text(image_path: str) -> str:
    """Extract text from image using OCR with enhanced handling"""
    try:
        # Step 1: Check if image is too large for direct processing
        if _should_use_tiled_processing(image_path):
            return process_image_in_tiles(image_path)

        # Step 2: Standard approach for smaller images
        with Image.open(image_path) as img:
            # Step 3: Extract basic OCR text
            full_text = extract_basic_ocr_text(img)

        return full_text

    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        return ""


def extract_ocr_text_with_chunks(image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text from image using OCR with enhanced handling"""
    try:
        # Step 1: Check if image is too large for direct processing
        if _should_use_tiled_processing(image_path):
            return process_image_in_tiles(image_path)

        # Step 2: Standard approach for smaller images
        with Image.open(image_path) as img:
            # Step 3: Extract basic OCR text
            full_text = extract_basic_ocr_text(img)

            # Step 4: Get OCR data with appropriate scaling
            ocr_data, scale_x, scale_y = get_ocr_data_with_scaling(img)

        # Step 5: Create chunks with proper coordinates
        chunks = []
        if full_text:
            chunks = _process_ocr_data_to_chunks(ocr_data, scale_x, scale_y)

        # Step 6: If no chunks were created, fall back to full text
        if not chunks and full_text:
            chunks = _create_fallback_chunk(full_text)

        # Step 7: Clean up memory
        gc.collect()

        return full_text, chunks

    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        return "", []


def _get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Extract image dimensions using PIL."""
    with Image.open(image_path) as img:
        return img.size


def _load_and_optimize_cv_image(image_path: str) -> Optional[np.ndarray]:
    """Load OpenCV image and optimize for analysis by resizing if too large."""
    cv_img = cv2.imread(image_path)

    if cv_img is None:
        return None

    # For large images, resize for analysis to save memory
    if cv_img.shape[0] * cv_img.shape[1] > 1_000_000:  # 1 million pixels
        scale = min(1.0, 1000 / max(cv_img.shape[0], cv_img.shape[1]))
        cv_img = cv2.resize(cv_img, None, fx=scale, fy=scale)

    return cv_img


def _calculate_average_color(cv_img: np.ndarray) -> np.ndarray:
    """Calculate the average RGB color of the image."""
    return np.average(np.average(cv_img, axis=0), axis=0)


def _detect_edges(cv_img: np.ndarray) -> np.ndarray:
    """Detect edges in the image using Canny edge detection."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    threshold1, threshold2, aperture_size = _get_canny_parameters()

    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)

    # Free memory
    del gray
    return edges


def _get_canny_parameters() -> Tuple[int, int, int]:
    """Get Canny edge detection parameters from settings with defaults."""
    threshold1 = (
        settings.CANNY_THRESHOLD1 if hasattr(settings, "CANNY_THRESHOLD1") else 50
    )
    threshold2 = (
        settings.CANNY_THRESHOLD2 if hasattr(settings, "CANNY_THRESHOLD2") else 150
    )
    aperture_size = (
        settings.CANNY_APERTURE_SIZE if hasattr(settings, "CANNY_APERTURE_SIZE") else 3
    )
    return threshold1, threshold2, aperture_size


def _get_hough_parameters() -> Tuple[int, int, int]:
    """Get Hough line detection parameters from settings with defaults."""
    threshold = (
        settings.HOUGH_THRESHOLD if hasattr(settings, "HOUGH_THRESHOLD") else 100
    )
    min_line_length = (
        settings.MIN_LINE_LENGTH if hasattr(settings, "MIN_LINE_LENGTH") else 100
    )
    max_line_gap = settings.MAX_LINE_GAP if hasattr(settings, "MAX_LINE_GAP") else 10
    return threshold, min_line_length, max_line_gap


def _detect_lines(edges: np.ndarray) -> int:
    """Detect lines in the edge-detected image and return count."""
    threshold, min_line_length, max_line_gap = _get_hough_parameters()

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    line_count = 0 if lines is None else len(lines)

    # Free memory
    if lines is not None:
        del lines

    return line_count


def _get_file_size(image_path: str) -> Optional[int]:
    """Get file size in bytes if file exists."""
    if os.path.exists(image_path):
        return os.path.getsize(image_path)
    return None


def _generate_feature_text(
    width: int,
    height: int,
    avg_color: np.ndarray,
    line_count: int,
    file_size: Optional[int],
) -> str:
    """Generate formatted feature text from extracted image features."""
    feature_text = f"""
        Text extracted from image using OCR appears above.

        Image features:
        - Image dimensions: {width}x{height}
        - Average RGB color: {avg_color}
        - Number of straight lines detected: {line_count}
        """

    if line_count > 10:
        feature_text += "This image likely contains a chart or diagram.\n"

    if file_size is not None:
        feature_text += f"- File size: {file_size} bytes\n"

    return feature_text.strip()


def _get_original_image_base64(image_path: str, original_size: int) -> Optional[str]:
    """
    Return base64 encoding of original image if it's small enough (under 1MB).

    Args:
        image_path: Path to the image file
        original_size: Size of the original image in bytes

    Returns:
        Base64 encoded string if successful, None if should proceed with processing
    """
    if original_size <= 1 * 1024 * 1024:
        try:
            with open(image_path, "rb") as image_file:
                logger.debug(
                    f"Using original image without re-encoding: {original_size} bytes"
                )
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.warning(
                f"Failed to read original image, falling back to processing: {str(e)}"
            )
    return None


def _determine_image_format(img: Image.Image, extension: str) -> str:
    """
    Determine the image format to use for processing.

    Args:
        img: PIL Image object
        extension: File extension from the path

    Returns:
        Format string ("PNG" or "JPEG")
    """
    return img.format or ("PNG" if extension in [".png"] else "JPEG")


def _handle_jpeg_transparency(img: Image.Image) -> Image.Image:
    """
    Handle transparency for JPEG conversions by creating a white background.

    Args:
        img: PIL Image object

    Returns:
        Image object with transparency handled
    """
    if img.mode == "RGBA":
        # Create white background for JPEG
        background = Image.new("RGB", img.size, (255, 255, 255))
        # Paste using alpha channel as mask
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        return background
    return img


def _optimize_png_image(
    img: Image.Image, buffer: BytesIO, original_size: int, max_size_bytes: int
) -> None:
    """
    Optimize PNG image with compression and optional resizing.

    Args:
        img: PIL Image object
        buffer: BytesIO buffer to write to
        original_size: Original image size in bytes
        max_size_bytes: Maximum allowed size in bytes
    """
    # For PNG, use compression level
    compress_level = 6  # Default compression level

    # If the image is large, try higher compression
    if original_size > 2 * 1024 * 1024:  # 2MB
        compress_level = 9  # Max compression

    img.save(buffer, format="PNG", compress_level=compress_level)

    # Check if we need to resize for very large PNGs
    if buffer.tell() > max_size_bytes and max(img.size) > 2000:
        # Reset buffer and resize image
        buffer.seek(0)
        buffer.truncate(0)

        # Calculate new size while maintaining aspect ratio
        scale_factor = min(1.0, 2000 / max(img.size))
        new_size = (
            int(img.width * scale_factor),
            int(img.height * scale_factor),
        )
        resized_img = img.resize(new_size, Image.LANCZOS)

        # Save resized image
        resized_img.save(buffer, format="PNG", compress_level=compress_level)

        logger.debug(f"Resized PNG image to {new_size}")


def _optimize_jpeg_image(
    img: Image.Image, buffer: BytesIO, original_size: int, max_size_bytes: int
) -> None:
    """
    Optimize JPEG image with quality reduction.

    Args:
        img: PIL Image object
        buffer: BytesIO buffer to write to
        original_size: Original image size in bytes
        max_size_bytes: Maximum allowed size in bytes
    """
    # For JPEG, use quality setting
    quality = 85 if original_size < 1 * 1024 * 1024 else 75

    # Try encoding with quality
    img.save(buffer, format="JPEG", quality=quality, optimize=True)

    # Progressive quality reduction if needed
    while buffer.tell() > max_size_bytes and quality > 30:
        quality -= 10
        buffer.seek(0)
        buffer.truncate(0)
        img.save(buffer, format="JPEG", quality=quality, optimize=True)


def _should_use_original_file(buffer: BytesIO, original_size: int) -> bool:
    """
    Check if processing made the file larger and we should use the original.

    Args:
        buffer: BytesIO buffer with processed image
        original_size: Original image size in bytes

    Returns:
        True if should use original file, False otherwise
    """
    return buffer.tell() > original_size * 1.1  # 10% threshold


def _get_original_file_base64(image_path: str, original_size: int) -> str:
    """
    Get base64 encoding of the original file.

    Args:
        image_path: Path to the image file
        original_size: Original image size in bytes

    Returns:
        Base64 encoded string of original file
    """
    with open(image_path, "rb") as image_file:
        logger.debug(
            f"Processing increased size by >10%, using original: {original_size} bytes"
        )
        return base64.b64encode(image_file.read()).decode("utf-8")


def _encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string (fallback method)"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise


def optimize_image_for_api(image_path: str, max_size_bytes: int = 4 * 1024 * 1024):
    """
    Optimizes an image for API transmission while preserving its original format.

    Args:
        image_path: Path to the image file
        max_size_bytes: Maximum size in bytes for the optimized image

    Returns:
        Base64 encoded string of the optimized image in its original format
    """
    original_size = os.path.getsize(image_path)
    extension = os.path.splitext(image_path)[1].lower()

    # Step 1: Check if we can use the original image without processing
    original_base64 = _get_original_image_base64(image_path, original_size)
    if original_base64:
        return original_base64

    # Step 2: Process the image for optimization
    try:
        with Image.open(image_path) as img:
            # Step 3: Determine format and handle transparency
            original_format = _determine_image_format(img, extension)

            if original_format == "JPEG":
                img = _handle_jpeg_transparency(img)

            # Step 4: Create buffer and optimize based on format
            buffer = BytesIO()

            if original_format == "PNG":
                _optimize_png_image(img, buffer, original_size, max_size_bytes)
            else:
                _optimize_jpeg_image(img, buffer, original_size, max_size_bytes)

            # Step 5: Check if we should use the original file instead
            if _should_use_original_file(buffer, original_size):
                return _get_original_file_base64(image_path, original_size)

            # Step 6: Log results and return processed image
            logger.debug(
                f"Image processed: original size {original_size}, "
                f"new size {buffer.tell()}, format {original_format}"
            )

            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")

    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)}")
        # Fall back to standard encoding if optimization fails
        return _encode_image_to_base64(image_path)


def extract_image_features(image_path: str) -> str:
    """
    Extract detailed image features with memory optimization.

    This function orchestrates the extraction of various image features including:
    1. Image dimensions
    2. Average color analysis
    3. Edge and line detection for chart/diagram identification
    4. File size information

    Args:
        image_path: Path to the image file

    Returns:
        Formatted string containing all extracted image features
    """
    try:
        # Step 1: Extract basic image dimensions
        width, height = _get_image_dimensions(image_path)

        # Step 2: Load and optimize image for OpenCV analysis
        cv_img = _load_and_optimize_cv_image(image_path)

        if cv_img is None:
            return f"Image dimensions: {width}x{height}"

        # Step 3: Calculate average color
        avg_color = _calculate_average_color(cv_img)

        # Step 4: Detect edges for line analysis
        edges = _detect_edges(cv_img)

        # Free cv_img memory early
        del cv_img

        # Step 5: Detect lines in the image
        line_count = _detect_lines(edges)

        # Free edges memory
        del edges

        # Step 6: Get file size information
        file_size = _get_file_size(image_path)

        # Step 7: Force garbage collection for memory optimization
        gc.collect()

        # Step 8: Generate formatted feature text
        return _generate_feature_text(width, height, avg_color, line_count, file_size)

    except Exception as e:
        logger.error(f"Error extracting image features: {str(e)}")
        return "Error extracting image features"
