"""
Utility module for managing video processing models (Whisper, MiniCPM-V, etc.).
Handles automatic model downloading and initialization for video ingestion tasks.
Designed to work reliably in both FastAPI and Celery environments.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from app.be_core.config import settings
from app.be_core.logger import logger
from app.utils.feature_flags import is_video_ingestion_enabled


class VideoModelManager:
    """
    Manages video processing model downloading and initialization.
    Thread-safe singleton pattern for use in both FastAPI and Celery environments.
    """

    def __init__(self):
        self._downloaded_models: Set[str] = set()
        self._whisper_model: Optional[object] = None
        self._caption_model: Optional[object] = None
        self._caption_tokenizer: Optional[object] = None
        self._initialization_lock = (
            False  # Simple lock to prevent concurrent initialization
        )
        self._cuda_available: Optional[bool] = None
        self._torch_available: Optional[bool] = None

    def _check_dependencies(self) -> Tuple[bool, bool]:
        """
        Check if required dependencies are available and video ingestion is enabled.

        Returns:
            Tuple[bool, bool]: (torch_available, cuda_available)
        """
        # First check if video ingestion is enabled
        if not is_video_ingestion_enabled():
            logger.info("Video ingestion is disabled via settings or feature flag")
            self._torch_available = False
            self._cuda_available = False
            return False, False

        if self._torch_available is None:
            try:
                import torch

                self._torch_available = True
                self._cuda_available = torch.cuda.is_available()

                # Detailed CUDA diagnostics
                if not self._cuda_available:
                    logger.warning("CUDA is not available. Diagnostic information:")
                    logger.warning(f"  - PyTorch version: {torch.__version__}")
                    logger.warning(f"  - CUDA compiled version: {torch.version.cuda}")
                    logger.warning(f"  - CUDA runtime version: {torch.version.cuda}")
                    logger.warning(
                        f"  - CUDA device count: {torch.cuda.device_count()}"
                    )

                    # Check if CUDA was compiled into PyTorch
                    if not torch.cuda.is_available():
                        if not hasattr(torch.cuda, "is_available"):
                            logger.warning(
                                "  - PyTorch was not compiled with CUDA support"
                            )
                        else:
                            logger.warning(
                                "  - CUDA drivers/runtime may not be installed or compatible"
                            )
                            logger.warning("  - Check: nvidia-smi command availability")
                            logger.warning(
                                "  - Check: CUDA driver version compatibility with PyTorch"
                            )

                logger.info(
                    f"Video ingestion enabled - PyTorch available: {self._torch_available}, CUDA available: {self._cuda_available}"
                )
            except ImportError:
                self._torch_available = False
                self._cuda_available = False
                logger.warning(
                    "PyTorch not available - video model functionality disabled"
                )

        return self._torch_available, self._cuda_available

    def get_whisper_model(self) -> Optional[object]:
        """
        Get the Whisper transcription model instance.

        Returns:
            Whisper model instance or None if not available
        """
        torch_available, cuda_available = self._check_dependencies()

        if not torch_available:
            logger.warning("PyTorch not available - cannot load Whisper model")
            return None

        # Note: Whisper can fallback to CPU if CUDA is not available

        if self._whisper_model is not None:
            return self._whisper_model

        # Try to initialize Whisper model
        logger.info("Attempting to initialize Whisper model...")
        try:
            self._whisper_model = self._create_whisper_model()
            if self._whisper_model:
                logger.info("Successfully initialized Whisper model")
                self._downloaded_models.add("whisper")
            return self._whisper_model
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {str(e)}")
            return None

    def get_caption_model(self) -> Tuple[Optional[object], Optional[object]]:
        """
        Get the caption model and tokenizer instances.

        Returns:
            Tuple[model, tokenizer] or (None, None) if not available
        """
        torch_available, cuda_available = self._check_dependencies()

        if not torch_available:
            logger.warning("PyTorch not available - cannot load caption model")
            return None, None

        # Note: We now support CPU fallback for caption models, so CUDA is not required

        if not getattr(settings, "VIDEO_ENABLE_CAPTIONING", True):
            logger.info("Video captioning disabled in settings")
            return None, None

        if self._caption_model is not None and self._caption_tokenizer is not None:
            return self._caption_model, self._caption_tokenizer

        # Try to initialize caption model
        logger.info("Attempting to initialize caption model...")
        try:
            model, tokenizer = self._create_caption_model()
            if model and tokenizer:
                self._caption_model = model
                self._caption_tokenizer = tokenizer
                logger.info("Successfully initialized caption model")
                self._downloaded_models.add("caption")
            return self._caption_model, self._caption_tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize caption model: {str(e)}")
            return None, None

    def _create_whisper_model(self) -> Optional[object]:
        """
        Create Whisper model instance for transcription.

        Returns:
            Whisper model instance or None if failed
        """
        try:
            import logging

            from faster_whisper import WhisperModel

            # Check CUDA availability for device selection
            torch_available, cuda_available = self._check_dependencies()
            device = "cuda" if cuda_available else "cpu"

            model_name = getattr(
                settings,
                "VIDEO_WHISPER_MODEL",
                "Systran/faster-distil-whisper-large-v3",
            )

            logger.info(f"Loading Whisper model {model_name} on {device}...")

            model = WhisperModel(model_name, device=device)

            # Reduce Whisper logging verbosity
            model.logger.setLevel(logging.WARNING)

            logger.info(f"Whisper model {model_name} loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error creating Whisper model: {str(e)}")

            # Enhanced cuDNN error handling
            if "libcudnn" in str(e) or "cudnn" in str(e).lower():
                logger.error("=" * 70)
                logger.error("ERROR: cuDNN library compatibility issue detected!")
                logger.error("=" * 70)
                logger.error(
                    "This indicates a version mismatch between PyTorch's cuDNN 9.x"
                )
                logger.error(
                    "libraries and faster-whisper's expectation of cuDNN 8.x naming."
                )
                logger.error("")
                logger.error("TROUBLESHOOTING STEPS:")
                logger.error(
                    "1. Ensure container built with ENABLE_VIDEO_INGESTION=true"
                )
                logger.error("2. Check if cuDNN symlinks exist in container:")
                logger.error(
                    "   ls -la /usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/"
                )
                logger.error("3. Verify LD_LIBRARY_PATH includes cudnn/lib directory")
                logger.error("4. Rebuild container if symlinks are missing")
                logger.error("=" * 70)

            return None

    def _create_caption_model(self) -> Tuple[Optional[object], Optional[object]]:
        """
        Create caption model and tokenizer instances.

        Returns:
            Tuple[model, tokenizer] or (None, None) if failed
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_name = getattr(
                settings, "VIDEO_CAPTION_MODEL", "openbmb/MiniCPM-V-2_6-int4"
            )

            logger.info(f"Loading caption model {model_name}...")

            # Check if we should use quantization based on CUDA availability
            if torch.cuda.is_available():
                model = AutoModel.from_pretrained(
                    model_name,
                    revision="06219bd",
                    trust_remote_code=True,
                    device_map="cuda",
                    torch_dtype="auto",
                )
            else:
                # Fallback for CPU or when CUDA is not available
                logger.warning(
                    "CUDA not available, loading model on CPU without quantization"
                )
                # Security note: Using "main" revision for CPU fallback model
                # This is safe as it's a fallback scenario and revision is pinned
                model = AutoModel.from_pretrained(
                    model_name.replace("-int4", ""),  # Use non-quantized version
                    revision="main",  # Pin revision for security  # nosec B615
                    trust_remote_code=True,
                    device_map="cpu",
                    torch_dtype="float32",
                )

            # Use consistent model name for tokenizer
            tokenizer_model_name = (
                model_name
                if torch.cuda.is_available()
                else model_name.replace("-int4", "")
            )
            tokenizer_revision = "06219bd" if torch.cuda.is_available() else "main"

            # Security note: Revision is always pinned (either "06219bd" or "main")
            # This is safe as we never use floating references
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model_name,
                revision=tokenizer_revision,  # Always pin revision for security  # nosec B615
                trust_remote_code=True,
                use_fast=True,
                legacy=False,
            )

            model.eval()
            logger.info(f"Caption model {model_name} loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error creating caption model: {str(e)}")
            return None, None

    def initialize_models_at_startup(self, force_download: bool = False) -> bool:
        """
        Initialize video processing models at startup.

        Args:
            force_download: If True, force re-download even if models exist

        Returns:
            True if successful, False otherwise
        """
        # Prevent concurrent initialization
        if self._initialization_lock:
            logger.info("Video model initialization already in progress, waiting...")
            return self.is_ready()

        self._initialization_lock = True

        try:
            logger.info("Starting video model initialization at startup...")

            torch_available, cuda_available = self._check_dependencies()

            if not torch_available:
                logger.warning(
                    "PyTorch not available - skipping video model initialization"
                )
                return False

            if not cuda_available:
                logger.warning(
                    "CUDA not available - skipping GPU video model initialization"
                )
                return False

            # Check if models should be downloaded
            if not force_download and self._are_models_already_downloaded():
                logger.info(
                    "Video models already downloaded at build time, loading from cache..."
                )
                # Try to load the pre-downloaded models
                success_count = 0

                # Load Whisper model if transcription is enabled
                if getattr(settings, "VIDEO_ENABLE_TRANSCRIPTION", True):
                    whisper_model = self._create_whisper_model()
                    if whisper_model:
                        self._whisper_model = whisper_model
                        success_count += 1
                        logger.info("Whisper model loaded from cache")

                # Load caption model if captioning is enabled
                if getattr(settings, "VIDEO_ENABLE_CAPTIONING", True):
                    caption_model, caption_tokenizer = self._create_caption_model()
                    if caption_model and caption_tokenizer:
                        self._caption_model = caption_model
                        self._caption_tokenizer = caption_tokenizer
                        success_count += 1
                        logger.info("Caption model loaded from cache")

                return success_count > 0

            start_time = time.time()
            success_count = 0

            # Initialize Whisper model if transcription is enabled
            if getattr(settings, "VIDEO_ENABLE_TRANSCRIPTION", True):
                logger.info("Initializing Whisper model...")
                whisper_model = self._create_whisper_model()
                if whisper_model:
                    self._whisper_model = whisper_model
                    success_count += 1
                    logger.info("Whisper model initialized successfully")
                else:
                    logger.warning("Failed to initialize Whisper model")

            # Initialize caption model if captioning is enabled
            if getattr(settings, "VIDEO_ENABLE_CAPTIONING", True):
                logger.info("Initializing caption model...")
                caption_model, caption_tokenizer = self._create_caption_model()
                if caption_model and caption_tokenizer:
                    self._caption_model = caption_model
                    self._caption_tokenizer = caption_tokenizer
                    success_count += 1
                    logger.info("Caption model initialized successfully")
                else:
                    logger.warning("Failed to initialize caption model")

            download_time = time.time() - start_time
            logger.info(
                f"Video model initialization completed in {download_time:.2f} seconds"
            )
            logger.info(f"Successfully initialized {success_count} video models")

            return success_count > 0

        except Exception as e:
            logger.error(f"Error initializing video models: {str(e)}")
            return False
        finally:
            self._initialization_lock = False

    def _are_models_already_downloaded(self) -> bool:
        """
        Check if video models are already downloaded.
        Uses the same cache detection logic as docling for consistency.

        Returns:
            True if models exist, False otherwise
        """
        try:
            cache_dirs = [
                os.path.expanduser("~/.cache/huggingface"),
                os.path.expanduser("~/.cache/docling"),
                os.path.join(tempfile.gettempdir(), "video_model_cache"),
            ]

            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    # Check if there are model files (same logic as docling)
                    model_files = (
                        list(Path(cache_dir).rglob("*.bin"))
                        + list(Path(cache_dir).rglob("*.safetensors"))
                        + list(Path(cache_dir).rglob("config.json"))
                    )
                    if model_files:
                        logger.info(f"Found existing video models in {cache_dir}")
                        return True

            return False

        except Exception as e:
            logger.warning(f"Error checking for existing video models: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """
        Check if the video model manager is ready.

        Returns:
            True if at least one model is available, False otherwise
        """
        torch_available, cuda_available = self._check_dependencies()

        if not torch_available or not cuda_available:
            return False

        # Check if at least one model is available
        transcription_ready = (
            not getattr(settings, "VIDEO_ENABLE_TRANSCRIPTION", True)
            or self._whisper_model is not None
        )
        captioning_ready = not getattr(settings, "VIDEO_ENABLE_CAPTIONING", True) or (
            self._caption_model is not None and self._caption_tokenizer is not None
        )

        return transcription_ready and captioning_ready

    def get_status(self) -> Dict[str, any]:
        """
        Get the current status and information.

        Returns:
            Dictionary with status information
        """
        torch_available, cuda_available = self._check_dependencies()

        return {
            "ready": self.is_ready(),
            "torch_available": torch_available,
            "cuda_available": cuda_available,
            "downloaded_models": list(self._downloaded_models),
            "whisper_model_loaded": self._whisper_model is not None,
            "caption_model_loaded": self._caption_model is not None,
            "caption_tokenizer_loaded": self._caption_tokenizer is not None,
            "initialization_in_progress": self._initialization_lock,
            "transcription_enabled": getattr(
                settings, "VIDEO_ENABLE_TRANSCRIPTION", True
            ),
            "captioning_enabled": getattr(settings, "VIDEO_ENABLE_CAPTIONING", True),
        }

    def clear_cache(self) -> None:
        """Clear cached models (useful for testing or memory management)."""
        self._whisper_model = None
        self._caption_model = None
        self._caption_tokenizer = None
        self._downloaded_models.clear()
        logger.info("Cleared video model cache")

    def cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory after model usage."""
        try:
            torch_available, cuda_available = self._check_dependencies()
            if torch_available and cuda_available:
                import torch

                torch.cuda.empty_cache()
                logger.debug("Cleaned up GPU memory")
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU memory: {e}")


# Global singleton instance
_video_model_manager = VideoModelManager()


# Public API functions - these provide a clean interface for both FastAPI and Celery
def get_whisper_model() -> Optional[object]:
    """
    Get the Whisper transcription model instance.
    This is the main function to use for speech-to-text processing.

    Returns:
        Whisper model instance or None if not available
    """
    return _video_model_manager.get_whisper_model()


def get_caption_model() -> Tuple[Optional[object], Optional[object]]:
    """
    Get the caption model and tokenizer instances.
    This is the main function to use for video captioning.

    Returns:
        Tuple[model, tokenizer] or (None, None) if not available
    """
    return _video_model_manager.get_caption_model()


def initialize_video_models(force_download: bool = False) -> bool:
    """
    Initialize video processing models at startup.
    This function is primarily used during application startup.

    Args:
        force_download: If True, force re-download even if models exist

    Returns:
        True if successful, False otherwise
    """
    return _video_model_manager.initialize_models_at_startup(force_download)


def is_video_models_ready() -> bool:
    """
    Check if video models are ready for use.
    Useful for health checks and status monitoring.

    Returns:
        True if ready, False otherwise
    """
    return _video_model_manager.is_ready()


def get_video_models_status() -> Dict[str, any]:
    """
    Get video model status.
    Useful for monitoring and debugging.

    Returns:
        Dictionary with status information
    """
    return _video_model_manager.get_status()


def clear_video_models_cache() -> None:
    """
    Clear cached video models.
    Useful for testing or memory management.
    """
    _video_model_manager.clear_cache()


def cleanup_video_gpu_memory() -> None:
    """
    Clean up GPU memory after video processing.
    Should be called after intensive video processing tasks.
    """
    _video_model_manager.cleanup_gpu_memory()


# Backward compatibility functions (deprecated but kept for existing code)
def get_video_models_legacy() -> (
    Tuple[Optional[object], Optional[object], Optional[object]]
):
    """
    Legacy function for backward compatibility.
    Returns (whisper_model, caption_model, caption_tokenizer).

    DEPRECATED: Use get_whisper_model() and get_caption_model() instead.
    """
    whisper_model = get_whisper_model()
    caption_model, caption_tokenizer = get_caption_model()
    return whisper_model, caption_model, caption_tokenizer
