"""
Startup initialization for video processing models.
This module provides functions to initialize video models during application startup.
"""

from app.be_core.config import settings
from app.be_core.logger import logger
from app.utils.feature_flags import is_video_ingestion_enabled
from app.utils.video_model_manager import (
    get_video_models_status,
    initialize_video_models,
    is_video_models_ready,
)


def initialize_video_models_on_startup() -> bool:
    """
    Initialize video processing models during application startup.
    This function should be called during FastAPI startup events.

    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Check if video ingestion is enabled first
        if not is_video_ingestion_enabled():
            logger.info(
                "Video ingestion disabled via settings or feature flag - skipping model initialization"
            )
            return True

        # Check if models are already ready before attempting initialization
        if is_video_models_ready():
            logger.info("Video models are already ready - skipping initialization")
            return True

        logger.info(
            "Video ingestion enabled - initializing video processing models on startup..."
        )

        video_transcription_enabled = getattr(
            settings, "VIDEO_ENABLE_TRANSCRIPTION", True
        )
        video_captioning_enabled = getattr(settings, "VIDEO_ENABLE_CAPTIONING", True)

        if not video_transcription_enabled and not video_captioning_enabled:
            logger.info(
                "Both video transcription and captioning disabled - skipping model initialization"
            )
            return True

        # Initialize models
        success = initialize_video_models(force_download=False)

        if success:
            logger.info("Video models initialized successfully on startup")

            # Log status for monitoring
            status = get_video_models_status()
            logger.info(f"Video models status: {status}")

            return True
        else:
            logger.warning("Video model initialization failed on startup")
            logger.info("Video processing will attempt to load models on-demand")
            return False

    except Exception as e:
        logger.error(f"Error during video model startup initialization: {str(e)}")
        logger.info("Video processing will attempt to load models on-demand")
        return False


def check_video_models_health() -> dict:
    """
    Check the health status of video processing models.
    This function can be used for health check endpoints.

    Returns:
        Dictionary with health status information
    """
    try:
        status = get_video_models_status()
        ready = is_video_models_ready()

        return {
            "video_models_ready": ready,
            "status": status,
            "health": "healthy" if ready else "degraded",
        }

    except Exception as e:
        logger.error(f"Error checking video models health: {str(e)}")
        return {
            "video_models_ready": False,
            "status": {"error": str(e)},
            "health": "unhealthy",
        }
