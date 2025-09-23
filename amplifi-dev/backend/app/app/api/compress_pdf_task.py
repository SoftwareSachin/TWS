import io
import os
import tempfile
from typing import Any, Dict
from uuid import UUID

import fitz
import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session

from app.be_core.celery import celery
from app.be_core.logger import logger
from app.db.session import SyncSessionLocal
from app.models.file_model import File
from app.schemas.file_schema import FileStatusEnum


@celery.task(name="tasks.compress_pdf_task", bind=True, max_retries=3, acks_late=True)
def compress_pdf_task(
    self,
    file_id: UUID,
    workspace_id: UUID,
    user_id: UUID = None,
    quality: int = 60,
    dpi: int = 72,
) -> Dict[str, Any]:
    """
    Simple PDF compression with robust table detection and rotation.
    """
    logger.info(f"Starting PDF compression for file_id: {file_id}")

    db_session: Session = SyncSessionLocal()
    temp_compressed_path = None

    try:
        # Get file from database
        file_record = (
            db_session.query(File)
            .filter(
                File.id == file_id,
                File.workspace_id == workspace_id,
                File.deleted_at.is_(None),
            )
            .first()
        )

        if not file_record:
            logger.error(f"File not found: {file_id}")
            return {"success": False, "error": "File not found"}

        original_path = file_record.file_path
        if not os.path.exists(original_path):
            logger.error(f"File path does not exist: {original_path}")
            file_record.status = FileStatusEnum.Failed
            db_session.commit()
            return {"success": False, "error": "File path does not exist"}

        # Update status to processing
        file_record.status = FileStatusEnum.Processing
        db_session.commit()

        # Create temporary file for compressed version
        filename_without_ext = os.path.splitext(file_record.filename)[0]
        temp_fd, temp_compressed_path = tempfile.mkstemp(
            suffix=f"_{filename_without_ext}_compressed.pdf",
            dir=os.path.dirname(original_path),
        )
        os.close(temp_fd)  # Close file descriptor, we only need the path

        # Perform compression with aggressive memory limits
        compression_result = compress_pdf_with_table_detection(
            original_path,
            temp_compressed_path,
            quality=quality,  # Increase for better table text
            dpi=dpi,  # Keep user request
            max_analysis_width=400,
            max_processing_width=1000,  # Slightly smaller for memory
            preserve_table_quality=True,  # Enable smart DPI
        )

        # Replace original file with compressed version
        os.remove(original_path)
        os.rename(temp_compressed_path, original_path)
        temp_compressed_path = None

        # Update file size and status
        compressed_size = os.path.getsize(original_path)
        file_record.size = compressed_size
        file_record.status = FileStatusEnum.Uploaded
        db_session.commit()

        logger.info(
            f"PDF compression completed for {file_id}. "
            f"Original: {compression_result['original_size']/1024:.2f}KB, "
            f"Compressed: {compressed_size/1024:.2f}KB, "
            f"Reduction: {compression_result['compression_ratio']:.1f}%"
        )

        return {
            "success": True,
            "original_size": compression_result["original_size"],
            "compressed_size": compressed_size,
            "compression_ratio": compression_result["compression_ratio"],
            "file_id": str(file_id),
        }

    except Exception as e:
        logger.error(
            f"PDF compression failed for file_id {file_id}: {str(e)}", exc_info=True
        )

        # Clean up temp file
        if temp_compressed_path and os.path.exists(temp_compressed_path):
            try:
                os.remove(temp_compressed_path)
            except Exception as e_cleanup:
                logger.warning(
                    f"Failed to clean up temporary file {temp_compressed_path}: {e_cleanup}"
                )

        # Update file status to failed
        if "file_record" in locals():
            file_record.status = FileStatusEnum.Failed
            db_session.commit()

        # Retry logic
        if self.request.retries < self.max_retries:
            countdown = 60 * (2**self.request.retries)
            logger.info(
                f"Retrying PDF compression for file_id {file_id}, attempt {self.request.retries + 1}"
            )
            raise self.retry(exc=e, countdown=countdown)

        return {"success": False, "error": str(e), "file_id": str(file_id)}

    finally:
        db_session.close()


def compress_pdf_with_table_detection(
    input_path: str,
    output_path: str,
    quality: int = 60,
    dpi: int = 72,
    max_analysis_width: int = 400,
    max_processing_width: int = 1200,
    preserve_table_quality: bool = True,
) -> Dict[str, Any]:
    """
    PDF compression with smart DPI selection for table preservation.
    """
    original_size = os.path.getsize(input_path)

    with fitz.open(input_path) as doc:
        canvas_obj = canvas.Canvas(output_path, pagesize=A4)

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height

                # Quick content type detection with minimal memory
                if preserve_table_quality:
                    analysis_pix = page.get_pixmap(dpi=25)
                    analysis_img = Image.frombytes(
                        "RGB",
                        [analysis_pix.width, analysis_pix.height],
                        analysis_pix.samples,
                    )
                    content_type = detect_content_type(analysis_img)
                    analysis_img.close()
                    analysis_pix = None
                else:
                    content_type = "mixed"

                # Calculate smart DPI
                optimal_dpi = calculate_smart_dpi(
                    page_width, page_height, dpi, max_processing_width, content_type
                )

                logger.debug(
                    f"Page {page_num + 1}: Content type: {content_type}, "
                    f"Using DPI {optimal_dpi} (requested: {dpi})"
                )

                # Convert page to image with the calculated optimal DPI
                pix = page.get_pixmap(dpi=optimal_dpi)

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Clean up immediately
                pix.ib = None
                pix = None
                page = None

                logger.debug(
                    f"Page {page_num + 1}: Processing image {img.width}x{img.height} at {optimal_dpi} DPI"
                )

                # Create very small image for analysis only
                analysis_img = None
                if img.width > max_analysis_width:
                    ratio = max_analysis_width / img.width
                    analysis_size = (max_analysis_width, int(img.height * ratio))
                    analysis_img = img.resize(
                        analysis_size, Image.NEAREST
                    )  # Faster resize
                    logger.debug(f"Page {page_num + 1}: Analysis image {analysis_size}")
                else:
                    analysis_img = img

                # Detect content orientation using the small analysis image
                needs_rotation, rotation_degrees = detect_content_orientation(
                    analysis_img, page_num
                )

                # Clean up analysis image if different
                if analysis_img is not img:
                    analysis_img.close()
                    analysis_img = None

                # Apply rotation if needed
                if needs_rotation and rotation_degrees > 0:
                    logger.debug(f"Page {page_num + 1}: Rotating {rotation_degrees}°")
                    rotated_img = img.rotate(rotation_degrees, expand=True)
                    img.close()
                    img = rotated_img

                # Compress image with memory-efficient approach
                img_bytes = io.BytesIO()

                # Further reduce size if still too large
                if img.width > max_processing_width:
                    ratio = max_processing_width / img.width
                    new_size = (max_processing_width, int(img.height * ratio))
                    resized_img = img.resize(new_size, Image.LANCZOS)
                    img.close()
                    img = resized_img
                    logger.debug(
                        f"Page {page_num + 1}: Resized to {new_size} for processing"
                    )

                # Save with aggressive compression
                img.save(img_bytes, format="JPEG", quality=quality, optimize=True)

                # Store dimensions and close image immediately
                final_width = img.width
                final_height = img.height
                img.close()
                img = None

                # Force garbage collection
                import gc

                gc.collect()

                # Add to PDF
                img_bytes.seek(0)
                img_reader = ImageReader(img_bytes)

                # Choose appropriate page size
                if needs_rotation and rotation_degrees > 0:
                    page_width, page_height = calculate_optimal_page_size(
                        final_width, final_height
                    )
                    logger.debug(
                        f"Page {page_num + 1}: Dynamic page size {page_width:.0f}x{page_height:.0f}"
                    )
                else:
                    page_width, page_height = A4

                canvas_obj.setPageSize((page_width, page_height))

                # Calculate scaling
                scale_x = page_width / final_width
                scale_y = page_height / final_height
                scale = min(scale_x, scale_y)

                new_width = final_width * scale
                new_height = final_height * scale
                x_offset = (page_width - new_width) / 2
                y_offset = (page_height - new_height) / 2

                canvas_obj.drawImage(
                    img_reader, x_offset, y_offset, width=new_width, height=new_height
                )
                canvas_obj.showPage()

                # Clean up
                img_bytes.close()
                img_reader = None

                logger.debug(f"Processed page {page_num + 1}/{len(doc)}")

            except Exception as e:
                logger.warning(f"Failed to process page {page_num + 1}: {e}")
                # Clean up any remaining objects
                for obj_name in [
                    "img",
                    "analysis_img",
                    "rotated_img",
                    "resized_img",
                    "img_bytes",
                    "pix",
                    "page",
                ]:
                    try:
                        obj = locals().get(obj_name)
                        if obj and hasattr(obj, "close"):
                            obj.close()
                    except Exception as e_close:
                        logger.warning(
                            f"Failed to close object {obj_name} on page {page_num + 1}: {e_close}"
                        )
                # Add blank page and continue
                canvas_obj.showPage()
                continue

        canvas_obj.save()

    compressed_size = os.path.getsize(output_path)
    compression_ratio = (
        (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    )

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
    }


def calculate_optimal_page_size(
    content_width: int, content_height: int
) -> tuple[float, float]:
    """
    Calculate optimal page size for rotated table content.

    Args:
        content_width: Width of the rotated content
        content_height: Height of the rotated content

    Returns:
        Tuple of (page_width, page_height)
    """
    # Ensure minimum readable scale
    min_scale = 0.8
    max_page_height = 1191  # A3 height for very wide tables

    # Calculate required page size to maintain readable scale
    required_width = content_width / min_scale
    required_height = content_height / min_scale

    # Choose appropriate page dimensions
    if required_width <= 595:  # Fits in A4 portrait width
        return A4  # 595x842
    elif required_width <= 842:  # Fits in A4 landscape width
        page_width = 842
        page_height = min(max_page_height, max(842, required_height))
        return page_width, page_height
    else:  # Very wide table, use A3-like dimensions
        page_width = min(1191, required_width)
        page_height = min(max_page_height, max(842, required_height))
        return page_width, page_height


def detect_content_orientation(img: Image.Image, page_num: int) -> tuple[bool, int]:
    """
    Enhanced content orientation detection that distinguishes between:
    - Normal text content
    - Real tables that need rotation
    - Index/list content that looks like tables but shouldn't be rotated
    - Mixed content with rotated elements

    Args:
        img: PIL Image object
        page_num: Page number for logging

    Returns:
        tuple: (needs_rotation, rotation_degrees)
        - needs_rotation: True if content should be rotated
        - rotation_degrees: Degrees to rotate (90, 180, 270)
    """
    original_width = img.width
    original_height = img.height
    is_page_landscape = original_width > original_height

    logger.debug(
        f"Page {page_num + 1}: Analyzing content orientation ({original_width}x{original_height})"
    )

    # Simple case: Page-level landscape
    if is_page_landscape:
        logger.debug(f"Page {page_num + 1}: Page-level landscape detected")
        return True, 270  # Clockwise rotation

    # Complex case: Portrait page with potentially rotated content
    try:
        # Convert to grayscale and apply edge detection
        gray_img = img.convert("L")
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        img_array = np.array(edges)

        # Calculate edge density ratios
        horizontal_edges = np.sum(np.diff(img_array, axis=0) ** 2)
        vertical_edges = np.sum(np.diff(img_array, axis=1) ** 2)
        edge_ratio = horizontal_edges / (vertical_edges + 1)

        logger.debug(
            f"Page {page_num + 1}: Edge analysis - H:{horizontal_edges:.0f}, V:{vertical_edges:.0f}, Ratio:{edge_ratio:.2f}"
        )

        # Only analyze for rotation if edge ratio suggests landscape content
        if edge_ratio < 1.0:  # Landscape-leaning content

            # Multi-feature table analysis
            features = analyze_table_features(
                img_array, original_width, original_height, page_num
            )

            # Enhanced analysis to distinguish real tables from indexes
            table_type_analysis = analyze_table_vs_index(
                img_array, original_width, original_height, features
            )

            # Calculate table confidence
            table_confidence = calculate_table_confidence(features)
            border_score = features["border_score"]

            # Check if this is likely an index/list rather than a real table
            is_likely_index = table_type_analysis["is_index"]
            real_table_confidence = table_type_analysis["table_confidence"]

            logger.debug(
                f"Page {page_num + 1}: Table confidence: {table_confidence:.3f}, "
                f"Border score: {border_score:.3f}, "
                f"Real table confidence: {real_table_confidence:.3f}, "
                f"Likely index: {is_likely_index}"
            )

            # Enhanced decision logic with index detection
            should_rotate = False
            reason = ""

            # Don't rotate if it's likely an index/list
            if is_likely_index:
                should_rotate = False
                reason = "Detected index/list content - not rotating"

            # High confidence real table
            elif real_table_confidence > 0.6:
                should_rotate = True
                reason = "High confidence real table"

            # Strong border with landscape tendency (but not if it's an index)
            elif border_score > 0.8 and edge_ratio < 0.98 and not is_likely_index:
                should_rotate = True
                reason = "Strong border with landscape structure"

            # Medium confidence table with clear landscape signal
            elif real_table_confidence > 0.4 and edge_ratio < 0.95:
                should_rotate = True
                reason = "Medium confidence table with strong landscape signal"

            # Very strong landscape signal with some table features
            elif border_score > 0.6 and edge_ratio < 0.90 and table_confidence > 0.2:
                should_rotate = True
                reason = "Strong landscape signal with table features"

            if should_rotate:
                logger.debug(f"Page {page_num + 1}: {reason} - rotating 270°")
                return True, 270
            else:
                logger.debug(
                    f"Page {page_num + 1}: {reason if reason else 'Insufficient evidence for rotation'}"
                )
                return False, 0

        # Check for extremely horizontal content that might need different rotation
        elif edge_ratio > 3.0:
            logger.debug(f"Page {page_num + 1}: Highly horizontal structure detected")
            return True, 90  # Counter-clockwise for inverted content

        # Check content distribution for unbalanced layouts
        unbalanced_rotation = check_content_distribution(
            img_array, original_width, original_height, page_num
        )
        if unbalanced_rotation != 0:
            return True, unbalanced_rotation

        logger.debug(
            f"Page {page_num + 1}: No rotation needed - content appears properly oriented"
        )
        return False, 0

    except Exception as e:
        logger.warning(
            f"Page {page_num + 1}: Content analysis failed ({e}), falling back to basic detection"
        )
        return is_page_landscape, 270 if is_page_landscape else 0


def analyze_table_features(
    img_array: np.ndarray, width: int, height: int, page_num: int
) -> Dict[str, float]:
    """
    Analyze multiple features to detect table structure.

    Returns:
        Dictionary with feature scores (0.0 to 1.0)
    """
    features = {
        "grid_regularity": analyze_grid_regularity(img_array, width, height),
        "text_alignment": analyze_text_alignment(img_array, width, height),
        "whitespace_structure": analyze_whitespace_structure(img_array, width, height),
        "density_variance": analyze_density_variance(img_array, width, height),
        "border_score": detect_table_borders(img_array, width, height),
    }

    logger.debug(f"Page {page_num + 1}: Feature analysis:")
    for feature, score in features.items():
        logger.debug(f"  - {feature}: {score:.3f}")

    return features


def calculate_table_confidence(features: Dict[str, float]) -> float:
    """
    Calculate overall table confidence based on feature scores.

    Args:
        features: Dictionary of feature scores

    Returns:
        Confidence score (0.0 to 1.0)
    """
    # Count indicators above threshold
    table_indicators = sum(1 for score in features.values() if score > 0.3)

    # Calculate weighted total score
    weights = {
        "grid_regularity": 0.2,
        "text_alignment": 0.25,
        "whitespace_structure": 0.2,
        "density_variance": 0.15,
        "border_score": 0.2,
    }

    total_score = sum(features[feature] * weight for feature, weight in weights.items())

    # Combine score with indicator count
    confidence = total_score * (table_indicators / len(features))

    return confidence


def analyze_grid_regularity(img_array: np.ndarray, width: int, height: int) -> float:
    """Analyze grid regularity - tables have regular cell patterns."""
    try:
        # Memory-efficient: Sample only key intersection points
        sample_points = []
        step_x = max(10, width // 15)  # Fewer samples for memory efficiency
        step_y = max(10, height // 15)

        for y in range(step_y, height - step_y, step_y):
            for x in range(step_x, width - step_x, step_x):
                if y < height and x < width:
                    sample_points.append(img_array[y, x])

        if len(sample_points) > 4:
            # Regular grid should have more consistent intersection intensities
            std_dev = np.std(sample_points)
            mean_val = np.mean(sample_points)
            regularity = 1.0 - min(1.0, std_dev / (mean_val + 1))
            return regularity
        return 0.0
    except Exception:
        return 0.0


def analyze_text_alignment(img_array: np.ndarray, width: int, height: int) -> float:
    """Analyze text alignment patterns - tables have column alignment."""
    try:
        # Memory-efficient: Sample fewer rows
        start_positions = []
        step = max(20, height // 25)  # Adaptive sampling based on height

        for y in range(0, height, step):
            if y + 5 < height:
                row = img_array[y : y + 5, :]
                # Find first non-white pixel
                dark_pixels = np.where(row < 200)
                if len(dark_pixels[1]) > 0:
                    start_positions.append(dark_pixels[1][0])

        if len(start_positions) > 3:
            # Check for consistent column starts
            start_positions = np.array(start_positions)
            consistency = 1.0 - min(1.0, np.std(start_positions) / (width * 0.1))

            # Check for multiple consistent column positions
            unique_positions = []
            for pos in start_positions:
                if not any(abs(pos - upos) < width * 0.05 for upos in unique_positions):
                    unique_positions.append(pos)

            # More unique column positions suggest table structure
            column_bonus = min(1.0, len(unique_positions) / 5.0)
            return consistency * column_bonus

        return 0.0
    except Exception:
        return 0.0


def analyze_whitespace_structure(
    img_array: np.ndarray, width: int, height: int
) -> float:
    """Analyze whitespace structure - tables have structured whitespace."""
    try:
        # Memory-efficient: Sample fewer points
        step_y = max(15, height // 20)  # Adaptive sampling
        step_x = max(15, width // 20)

        # Analyze horizontal whitespace (between rows)
        horizontal_whitespace = []
        for y in range(0, height, step_y):
            if y < height:
                row_whitespace = np.sum(img_array[y, :] > 240) / width
                horizontal_whitespace.append(row_whitespace)

        # Analyze vertical whitespace (between columns)
        vertical_whitespace = []
        for x in range(0, width, step_x):
            if x < width:
                col_whitespace = np.sum(img_array[:, x] > 240) / height
                vertical_whitespace.append(col_whitespace)

        # Calculate structure score
        structure_score = 0
        if len(horizontal_whitespace) > 2:
            h_consistency = 1.0 - min(
                1.0,
                np.std(horizontal_whitespace) / (np.mean(horizontal_whitespace) + 0.1),
            )
            structure_score += h_consistency * 0.5

        if len(vertical_whitespace) > 2:
            v_consistency = 1.0 - min(
                1.0, np.std(vertical_whitespace) / (np.mean(vertical_whitespace) + 0.1)
            )
            structure_score += v_consistency * 0.5

        return structure_score
    except Exception:
        return 0.0


def analyze_density_variance(img_array: np.ndarray, width: int, height: int) -> float:
    """Analyze density variance - tables have varied content density."""
    try:
        # Memory-efficient: Use fewer, larger cells
        cell_width = width // 4  # Fewer cells
        cell_height = height // 4
        densities = []

        for y in range(0, height - cell_height, cell_height):
            for x in range(0, width - cell_width, cell_width):
                cell = img_array[y : y + cell_height, x : x + cell_width]
                # Sample only part of the cell for memory efficiency
                sample_size = min(cell.size, 1000)  # Limit sample size
                if sample_size < cell.size:
                    flat_cell = cell.flatten()
                    indices = np.random.choice(
                        len(flat_cell), sample_size, replace=False
                    )
                    density = np.sum(flat_cell[indices] < 200) / sample_size
                else:
                    density = np.sum(cell < 200) / cell.size
                densities.append(density)

        if len(densities) > 2:  # Reduced threshold
            variance = np.var(densities)
            mean_density = np.mean(densities)

            if mean_density > 0.1:
                return min(1.0, variance * 10)

        return 0.0
    except Exception:
        return 0.0


def detect_table_borders(img_array: np.ndarray, width: int, height: int) -> float:
    """Detect table borders - tables often have visible borders."""
    try:
        # Memory-efficient: Sample fewer lines
        horizontal_lines = 0
        step_y = max(20, height // 15)  # Adaptive sampling

        for y in range(10, height - 10, step_y):
            row = img_array[y, :]
            dark_pixels = row < 200
            runs = []
            current_run = 0

            for pixel in dark_pixels:
                if pixel:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)

            # Count significant horizontal lines
            long_runs = [r for r in runs if r > width * 0.3]
            horizontal_lines += len(long_runs)

        # Detect vertical lines (sample fewer)
        vertical_lines = 0
        step_x = max(20, width // 15)  # Adaptive sampling

        for x in range(10, width - 10, step_x):
            col = img_array[:, x]
            dark_pixels = col < 200
            runs = []
            current_run = 0

            for pixel in dark_pixels:
                if pixel:
                    current_run += 1
                else:
                    if current_run > 0:
                        runs.append(current_run)
                    current_run = 0
            if current_run > 0:
                runs.append(current_run)

            # Count significant vertical lines
            long_runs = [r for r in runs if r > height * 0.2]
            vertical_lines += len(long_runs)

        # Combine border detection
        total_lines = horizontal_lines + vertical_lines
        border_score = min(1.0, total_lines / 8.0)  # Adjusted for fewer samples

        return border_score
    except Exception:
        return 0.0


def check_content_distribution(
    img_array: np.ndarray, width: int, height: int, page_num: int
) -> int:
    """
    Check for unbalanced content distribution that might indicate rotation need.

    Returns:
        Rotation degrees (0, 90, 180, 270) or 0 if no rotation needed
    """
    try:
        # Sample top, middle, and bottom sections
        height_third = height // 3
        top_section = img_array[:height_third, :]
        middle_section = img_array[height_third : 2 * height_third, :]
        bottom_section = img_array[2 * height_third :, :]

        # Calculate content density in each section
        top_density = np.sum(top_section < 240) / top_section.size
        middle_density = np.sum(middle_section < 240) / middle_section.size
        bottom_density = np.sum(bottom_section < 240) / bottom_section.size

        logger.debug(
            f"Page {page_num + 1}: Content density - Top:{top_density:.3f}, Mid:{middle_density:.3f}, Bot:{bottom_density:.3f}"
        )

        # Check for heavily concentrated content
        max_density = max(top_density, middle_density, bottom_density)
        min_density = min(top_density, middle_density, bottom_density)

        if max_density > 0.1 and max_density / (min_density + 0.001) > 5:
            logger.debug(
                f"Page {page_num + 1}: Unbalanced content distribution detected"
            )

            # Determine rotation based on content concentration
            if top_density == max_density:
                return 180  # Content at top, rotate 180°
            elif bottom_density == max_density:
                return 0  # Content at bottom, might be OK
            else:
                return 270  # Content in middle, try 270°

        return 0
    except Exception:
        return 0


def calculate_smart_dpi(
    page_width: int,
    page_height: int,
    requested_dpi: int,
    max_processing_width: int,
    content_type: str = "mixed",
) -> int:
    """
    Calculate optimal DPI balancing memory usage and quality.

    Args:
        page_width: Original page width in points
        page_height: Original page height in points
        requested_dpi: User requested DPI
        max_processing_width: Maximum allowed pixel width
        content_type: "text", "table", "image", or "mixed"

    Returns:
        Optimal DPI value
    """

    # Calculate DPI that would fit within memory limits
    memory_limited_dpi = int((max_processing_width / page_width) * 72)

    # Set minimum DPI based on content type
    min_dpi_by_content = {
        "text": 50,  # Readable text
        "table": 60,  # Clear table structure
        "image": 50,  # Images can be lower
        "mixed": 55,  # Safe default
    }

    min_dpi = min_dpi_by_content.get(content_type, 55)

    # Choose the best compromise
    if memory_limited_dpi >= min_dpi:
        # Can use higher DPI within memory limits
        optimal_dpi = min(requested_dpi, memory_limited_dpi)
    else:
        # Memory constraint is tight, but respect minimum quality
        logger.warning(
            f"Memory constraint requires DPI {memory_limited_dpi}, "
            f"but content type '{content_type}' needs minimum {min_dpi}. "
            f"Using {min_dpi} DPI for quality preservation."
        )
        optimal_dpi = min_dpi

    # Absolute minimum for any content
    optimal_dpi = max(40, optimal_dpi)

    return optimal_dpi


def detect_content_type(img: Image.Image) -> str:
    """
    Quick content type detection for DPI optimization.
    """
    try:
        # Convert small sample to analyze
        sample_img = img.resize((200, 200), Image.NEAREST)
        edges = sample_img.filter(ImageFilter.FIND_EDGES)
        edge_density = np.mean(np.array(edges))

        if edge_density > 30:
            return "table"  # High structure = likely table
        elif edge_density > 15:
            return "text"  # Medium structure = text
        else:
            return "image"  # Low structure = image/photo

    except Exception:
        return "mixed"  # Safe default


def analyze_table_vs_index(
    img_array: np.ndarray, width: int, height: int, features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Enhanced analysis to distinguish real tables from indexes/lists that look like tables.

    Real tables typically have:
    - Multiple columns with varied content
    - Consistent cell structure
    - Headers and data separation
    - Mixed data types (text, numbers, symbols)

    Indexes typically have:
    - Mostly left-aligned text
    - Right-aligned page numbers
    - Consistent formatting throughout
    - Primarily text content

    Returns:
        Dict with analysis results
    """
    try:
        analysis = {
            "is_index": False,
            "table_confidence": 0.0,
            "index_indicators": 0,
            "table_indicators": 0,
        }

        # Feature 1: Column structure analysis
        column_analysis = analyze_column_structure(img_array, width, height)

        # Feature 2: Content type distribution
        content_distribution = analyze_content_distribution_detailed(
            img_array, width, height
        )

        # Feature 3: Alignment patterns
        alignment_patterns = analyze_alignment_patterns(img_array, width, height)

        # Feature 4: Row consistency
        row_consistency = analyze_row_consistency(img_array, width, height)

        # Index indicators
        index_score = 0

        # Strong left-right alignment pattern (typical of indexes)
        if (
            alignment_patterns["left_heavy"] > 0.7
            and alignment_patterns["right_heavy"] > 0.5
        ):
            index_score += 0.3
            analysis["index_indicators"] += 1

        # Consistent row spacing (typical of lists/indexes)
        if row_consistency["spacing_consistency"] > 0.8:
            index_score += 0.2
            analysis["index_indicators"] += 1

        # Mostly text content with numbers at edges
        if (
            content_distribution["text_percentage"] > 0.7
            and content_distribution["edge_numbers"] > 0.3
        ):
            index_score += 0.2
            analysis["index_indicators"] += 1

        # Limited column variety (2-3 columns max for indexes)
        if column_analysis["distinct_columns"] <= 3:
            index_score += 0.1
            analysis["index_indicators"] += 1

        # Low border score (indexes often have no borders)
        if features["border_score"] < 0.3:
            index_score += 0.1
            analysis["index_indicators"] += 1

        # Table indicators
        table_score = 0

        # Multiple distinct columns (real tables have more variety)
        if column_analysis["distinct_columns"] >= 4:
            table_score += 0.3
            analysis["table_indicators"] += 1

        # Mixed content distribution
        if (
            content_distribution["mixed_content"] > 0.6
            and content_distribution["data_variety"] > 0.5
        ):
            table_score += 0.3
            analysis["table_indicators"] += 1

        # Strong grid regularity
        if features["grid_regularity"] > 0.4:
            table_score += 0.2
            analysis["table_indicators"] += 1

        # Multiple alignment patterns (not just left-right)
        if alignment_patterns["center_content"] > 0.3:
            table_score += 0.2
            analysis["table_indicators"] += 1

        # High border score
        if features["border_score"] > 0.7:
            table_score += 0.2
            analysis["table_indicators"] += 1

        # Decision logic
        analysis["table_confidence"] = table_score

        # Strong index indicators with weak table indicators
        if index_score > 0.5 and table_score < 0.4:
            analysis["is_index"] = True

        # Very strong index pattern
        elif index_score > 0.7:
            analysis["is_index"] = True

        # Ambiguous case - err on side of not rotating
        elif index_score > 0.4 and table_score < 0.6:
            analysis["is_index"] = True

        return analysis

    except Exception as e:
        logger.warning(f"Error analyzing table vs index: {e}")
        # Safe fallback
        return {
            "is_index": False,
            "table_confidence": features.get("border_score", 0) * 0.5,
            "index_indicators": 0,
            "table_indicators": 0,
        }


def analyze_column_structure(
    img_array: np.ndarray, width: int, height: int
) -> Dict[str, float]:
    """Analyze column structure to detect table vs index patterns."""
    try:
        # Sample vertical strips to find column boundaries
        step_x = max(20, width // 15)
        column_densities = []

        for x in range(0, width, step_x):
            if x < width:
                col_density = np.sum(img_array[:, x] < 200) / height
                column_densities.append(col_density)

        # Find distinct columns (peaks in density)
        if len(column_densities) > 2:
            # Simple peak detection
            peaks = []
            for i in range(1, len(column_densities) - 1):
                if (
                    column_densities[i] > column_densities[i - 1]
                    and column_densities[i] > column_densities[i + 1]
                    and column_densities[i] > 0.1
                ):
                    peaks.append(i)

            distinct_columns = len(peaks)
        else:
            distinct_columns = 1

        return {
            "distinct_columns": distinct_columns,
            "column_separation": (
                np.std(column_densities) if len(column_densities) > 1 else 0
            ),
        }

    except Exception:
        return {"distinct_columns": 2, "column_separation": 0}


def analyze_content_distribution_detailed(
    img_array: np.ndarray, width: int, height: int
) -> Dict[str, float]:
    """Analyze content distribution to distinguish data types."""
    try:
        # Sample different regions
        left_third = img_array[:, : width // 3]
        middle_third = img_array[:, width // 3 : 2 * width // 3]
        right_third = img_array[:, 2 * width // 3 :]

        # Calculate densities
        left_density = np.sum(left_third < 200) / left_third.size
        middle_density = np.sum(middle_third < 200) / middle_third.size
        right_density = np.sum(right_third < 200) / right_third.size

        # Estimate content characteristics
        text_percentage = min(1.0, (left_density + middle_density) / 2)
        edge_numbers = min(1.0, right_density * 2)  # Numbers often on right
        mixed_content = 1.0 - abs(left_density - right_density)
        data_variety = np.std([left_density, middle_density, right_density]) * 3

        return {
            "text_percentage": text_percentage,
            "edge_numbers": edge_numbers,
            "mixed_content": mixed_content,
            "data_variety": min(1.0, data_variety),
        }

    except Exception:
        return {
            "text_percentage": 0.5,
            "edge_numbers": 0.3,
            "mixed_content": 0.5,
            "data_variety": 0.5,
        }


def analyze_alignment_patterns(
    img_array: np.ndarray, width: int, height: int
) -> Dict[str, float]:
    """Analyze text alignment patterns."""
    try:
        # Sample rows to find content start/end positions
        step_y = max(15, height // 20)
        left_starts = []
        right_ends = []
        center_content = 0

        for y in range(0, height, step_y):
            if y + 5 < height:
                row = img_array[y : y + 5, :]
                dark_pixels = np.where(row < 200)

                if len(dark_pixels[1]) > 0:
                    start_pos = dark_pixels[1][0]
                    end_pos = dark_pixels[1][-1]

                    left_starts.append(start_pos / width)
                    right_ends.append(end_pos / width)

                    # Check for center content
                    if 0.3 < (start_pos + end_pos) / (2 * width) < 0.7:
                        center_content += 1

        if len(left_starts) > 0:
            left_heavy = 1.0 - np.mean(left_starts)  # Content starting near left
            right_heavy = np.mean(right_ends)  # Content ending near right
            center_content = center_content / len(left_starts)
        else:
            left_heavy = right_heavy = center_content = 0

        return {
            "left_heavy": left_heavy,
            "right_heavy": right_heavy,
            "center_content": center_content,
        }

    except Exception:
        return {"left_heavy": 0.5, "right_heavy": 0.5, "center_content": 0.3}


def analyze_row_consistency(
    img_array: np.ndarray, width: int, height: int
) -> Dict[str, float]:
    """Analyze row spacing consistency."""
    try:
        # Find row boundaries by looking for horizontal whitespace
        step_y = max(10, height // 30)
        row_densities = []

        for y in range(0, height, step_y):
            if y < height:
                row_density = np.sum(img_array[y, :] < 200) / width
                row_densities.append(row_density)

        if len(row_densities) > 2:
            # Measure consistency in row spacing
            spacing_consistency = 1.0 - min(
                1.0, np.std(row_densities) / (np.mean(row_densities) + 0.1)
            )
        else:
            spacing_consistency = 0.5

        return {"spacing_consistency": spacing_consistency}

    except Exception:
        return {"spacing_consistency": 0.5}
