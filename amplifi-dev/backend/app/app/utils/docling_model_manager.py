"""
Utility module for managing docling model downloading and initialization.
Handles automatic model downloading at startup and provides model management functions.
Designed to work reliably in both FastAPI and Celery environments.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Set

import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.be_core.config import settings
from app.be_core.logger import logger
from app.models.document_model import DocumentTypeEnum


class DoclingModelManager:
    """
    Manages docling model downloading and initialization.
    Thread-safe singleton pattern for use in both FastAPI and Celery environments.
    """

    def __init__(self):
        self._downloaded_models: Set[str] = set()
        self._pdf_converter: Optional[DocumentConverter] = None
        self._other_converter: Optional[DocumentConverter] = None
        self._initialization_lock = (
            False  # Simple lock to prevent concurrent initialization
        )

    def get_converter(self, document_type: DocumentTypeEnum) -> DocumentConverter:
        """
        Get the appropriate DocumentConverter instance based on document type.
        If not initialized, attempts to initialize models and create converter.

        Args:
            document_type: Type of document to get converter for

        Returns:
            DocumentConverter instance

        Raises:
            RuntimeError: If unable to create converter after initialization attempts
        """
        # First try to get existing converter
        converter = self._get_cached_converter(document_type)

        if converter is not None:
            return converter

        # Fallback: Try to initialize models and create converter
        logger.warning(
            f"Docling converter not available for {document_type}, attempting to initialize..."
        )

        try:
            # Attempt to initialize models
            initialize_success = self.download_models_at_startup(force_download=False)

            if initialize_success:
                # Try to get converter again after initialization
                converter = self._get_cached_converter(document_type)
                if converter is not None:
                    logger.info(
                        f"Successfully initialized and retrieved docling converter for {document_type}"
                    )
                    return converter

            # If still not available, try to create converter directly
            logger.warning(
                f"Models initialized but converter still not available for {document_type}, creating new converter..."
            )

            converter = self._create_converter_directly(document_type)

            if converter is not None:
                # Store the converter for future use
                self._store_converter(document_type, converter)
                logger.info(
                    f"Successfully created and stored docling converter for {document_type}"
                )
                return converter

            # If all attempts failed, raise error
            raise RuntimeError(
                f"Failed to create docling converter for {document_type} after initialization attempts"
            )

        except Exception as e:
            error_msg = (
                f"Error initializing docling converter for {document_type}: {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _get_cached_converter(
        self, document_type: DocumentTypeEnum
    ) -> Optional[DocumentConverter]:
        """Get cached converter if available."""
        if document_type == DocumentTypeEnum.PDF:
            return self._pdf_converter
        else:
            return self._other_converter

    def _store_converter(
        self, document_type: DocumentTypeEnum, converter: DocumentConverter
    ) -> None:
        """Store converter in cache."""
        if document_type == DocumentTypeEnum.PDF:
            self._pdf_converter = converter
        else:
            self._other_converter = converter

    def _create_converter_directly(
        self, document_type: DocumentTypeEnum
    ) -> Optional[DocumentConverter]:
        """Create converter directly without initialization."""
        if document_type == DocumentTypeEnum.PDF:
            return self._create_pdf_converter()
        else:
            return self._create_other_converter()

    def download_models_at_startup(self, force_download: bool = False) -> bool:
        """
        Download required docling models at startup.

        Args:
            force_download: If True, force re-download even if models exist

        Returns:
            True if successful, False otherwise
        """
        # Prevent concurrent initialization
        if self._initialization_lock:
            logger.info("Docling initialization already in progress, waiting...")
            return self.is_ready()

        self._initialization_lock = True

        try:
            logger.info("Starting docling model download at startup...")

            # Check if models should be downloaded
            if not force_download and self._are_models_already_downloaded():
                logger.info("Docling models already downloaded, skipping download")
                return True

            # Initialize converters to trigger model download
            logger.info("Initializing docling DocumentConverters to download models...")
            start_time = time.time()

            # Initialize PDF converter
            pdf_converter = self._create_pdf_converter()
            if not pdf_converter:
                logger.error("Failed to initialize PDF DocumentConverter")
                return False

            # Initialize other document types converter
            other_converter = self._create_other_converter()
            if not other_converter:
                logger.error("Failed to initialize other DocumentConverter")
                return False

            download_time = time.time() - start_time
            logger.info(
                f"Docling models downloaded successfully in {download_time:.2f} seconds"
            )

            # Store the converters for reuse
            self._pdf_converter = pdf_converter
            self._other_converter = other_converter

            return True

        except Exception as e:
            logger.error(f"Error downloading docling models: {str(e)}")
            return False
        finally:
            self._initialization_lock = False

    def _create_pdf_converter(self) -> Optional[DocumentConverter]:
        """
        Create a DocumentConverter instance specifically for PDF documents.

        Returns:
            DocumentConverter instance for PDF or None if failed
        """
        try:
            # torch.set_default_device("cpu")
            # Configure PDF-specific options
            accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options
            pipeline_options.images_scale = settings.IMAGE_RESOLUTION_SCALE
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_page_images = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
            else:
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    force_full_page_ocr=False,
                    lang=settings.SUPPORTED_OCR_LANGUAGES,
                )

            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }

            logger.info("Testing PDF docling converter initialization...")
            return DocumentConverter(format_options=format_options)

        except Exception as e:
            logger.error(f"Error creating PDF DocumentConverter: {str(e)}")
            return None

    def _create_other_converter(self) -> Optional[DocumentConverter]:
        """
        Create a DocumentConverter instance for all other document types (DOCX, XLSX, PPTX, etc.).

        Returns:
            DocumentConverter instance for other formats or None if failed
        """
        try:
            # torch.set_default_device("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create format options for all supported non-PDF formats
            format_options = {}

            # Add support for DOCX, XLSX, PPTX, MD, HTML, CSV
            # These don't need special pipeline options like PDF does
            supported_formats = [
                InputFormat.DOCX,
                InputFormat.XLSX,
                InputFormat.PPTX,
                InputFormat.MD,
                InputFormat.HTML,
                InputFormat.CSV,
            ]

            for fmt in supported_formats:
                format_options[fmt] = None  # Use default options for these formats

            logger.info(
                "Testing other document types docling converter initialization..."
            )
            return DocumentConverter(format_options=format_options)

        except Exception as e:
            logger.error(f"Error creating other DocumentConverter: {str(e)}")
            return None

    def _are_models_already_downloaded(self) -> bool:
        """
        Check if docling models are already downloaded.

        Returns:
            True if models exist, False otherwise
        """
        try:
            # Check common docling model cache directories
            cache_dirs = [
                os.path.expanduser("~/.cache/docling"),
                os.path.expanduser("~/.cache/huggingface"),
                os.path.join(tempfile.gettempdir(), "docling_cache"),
            ]

            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    # Check if there are model files
                    model_files = list(Path(cache_dir).rglob("*.bin")) + list(
                        Path(cache_dir).rglob("*.safetensors")
                    )
                    if model_files:
                        logger.info(f"Found existing docling models in {cache_dir}")
                        return True

            return False

        except Exception as e:
            logger.warning(f"Error checking for existing models: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """
        Check if the model manager is ready (models downloaded and converters available).

        Returns:
            True if ready, False otherwise
        """
        return self._pdf_converter is not None and self._other_converter is not None

    def get_status(self) -> Dict[str, any]:
        """
        Get the current download status and information.

        Returns:
            Dictionary with status information
        """
        return {
            "ready": self.is_ready(),
            "downloaded_models": list(self._downloaded_models),
            "pdf_converter_initialized": self._pdf_converter is not None,
            "other_converter_initialized": self._other_converter is not None,
            "initialization_in_progress": self._initialization_lock,
        }

    def clear_cache(self) -> None:
        """Clear cached converters (useful for testing or memory management)."""
        self._pdf_converter = None
        self._other_converter = None
        logger.info("Cleared docling converter cache")


# Global singleton instance
_docling_model_manager = DoclingModelManager()


# Public API functions - these provide a clean interface for both FastAPI and Celery
def get_docling_converter(document_type: DocumentTypeEnum) -> DocumentConverter:
    """
    Get the appropriate DocumentConverter instance based on document type.
    This is the main function to use in both FastAPI and Celery code.

    Args:
        document_type: Type of document to get converter for

    Returns:
        DocumentConverter instance

    Raises:
        RuntimeError: If unable to create converter after initialization attempts
    """
    return _docling_model_manager.get_converter(document_type)


def initialize_docling_models(force_download: bool = False) -> bool:
    """
    Initialize docling models at startup.
    This function is primarily used during application startup.

    Args:
        force_download: If True, force re-download even if models exist

    Returns:
        True if successful, False otherwise
    """
    return _docling_model_manager.download_models_at_startup(force_download)


def is_docling_ready() -> bool:
    """
    Check if docling is ready for use.
    Useful for health checks and status monitoring.

    Returns:
        True if ready, False otherwise
    """
    return _docling_model_manager.is_ready()


def get_docling_status() -> Dict[str, any]:
    """
    Get docling model status.
    Useful for monitoring and debugging.

    Returns:
        Dictionary with status information
    """
    return _docling_model_manager.get_status()


def clear_docling_cache() -> None:
    """
    Clear cached docling converters.
    Useful for testing or memory management.

    Returns:
        None
    """
    _docling_model_manager.clear_cache()


# Backward compatibility functions (deprecated but kept for existing code)
def get_docling_converter_legacy(
    document_type: DocumentTypeEnum,
) -> Optional[DocumentConverter]:
    """
    Legacy function for backward compatibility.
    Returns Optional[DocumentConverter] instead of DocumentConverter.

    DEPRECATED: Use get_docling_converter() instead.
    """
    try:
        return get_docling_converter(document_type)
    except RuntimeError:
        return None
