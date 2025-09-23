"""
Feature flag utilities for managing Flagsmith feature toggles.

This module provides centralized functions for checking feature flags
across the application, making the code more modular and reusable.
"""

from flagsmith import Flagsmith

from app.be_core.config import settings
from app.be_core.logger import logger


async def is_feature_enabled_for_workspace(*, feature_name: str) -> bool:
    """
    Check if a Flagsmith feature is enabled for the organization that owns the workspace.

    This function provides a centralized way to check feature flags across the application.
    It uses the organization name as identity and returns False if Flagsmith is not configured.

    Args:
        feature_name (str): The name of the feature flag to check

    Returns:
        bool: True if the feature is enabled, False otherwise or if Flagsmith is not configured

    Example:
        >>> enabled = await is_feature_enabled_for_workspace(feature_name="groove_connector_feature")
        >>> if enabled:
        ...     # Execute feature-specific logic
        ...     pass
    """
    try:
        env_key = (settings.FLAGSMITH_ENVIRONMENT_KEY or "").strip()
        if not env_key:
            logger.debug("Flagsmith environment key not configured")
            return False

        fs = Flagsmith(environment_key=env_key)
        flags = fs.get_environment_flags()
        is_enabled = bool(flags.is_feature_enabled(feature_name))

        logger.debug(
            f"Feature flag '{feature_name}' is {'enabled' if is_enabled else 'disabled'}"
        )
        return is_enabled

    except Exception as e:
        logger.warning(f"Flagsmith feature check failed for '{feature_name}': {e}")
        return False


def is_feature_enabled_sync(*, feature_name: str) -> bool:
    """
    Synchronous version of feature flag check.

    Use this when you need to check feature flags in synchronous contexts.

    Args:
        feature_name (str): The name of the feature flag to check

    Returns:
        bool: True if the feature is enabled, False otherwise or if Flagsmith is not configured
    """
    try:
        env_key = (settings.FLAGSMITH_ENVIRONMENT_KEY or "").strip()
        if not env_key:
            logger.debug("Flagsmith environment key not configured")
            return False

        fs = Flagsmith(environment_key=env_key)
        flags = fs.get_environment_flags()
        is_enabled = bool(flags.is_feature_enabled(feature_name))

        logger.debug(
            f"Feature flag '{feature_name}' is {'enabled' if is_enabled else 'disabled'}"
        )
        return is_enabled

    except Exception as e:
        logger.warning(f"Flagsmith feature check failed for '{feature_name}': {e}")
        return False


def is_video_ingestion_enabled() -> bool:
    """
    Check if video ingestion is enabled via both settings and feature flags.

    This function combines the traditional settings-based check with Flagsmith feature flags
    to provide a comprehensive enablement check for video ingestion functionality.

    Returns:
        bool: True if video ingestion is enabled via both settings AND feature flag, False otherwise
    """
    try:
        # First check the settings-based flag (backward compatibility)
        settings_enabled = getattr(settings, "ENABLE_VIDEO_INGESTION", False)
        if not settings_enabled:
            logger.debug("Video ingestion disabled via ENABLE_VIDEO_INGESTION setting")
            return False

        # Then check the Flagsmith feature flag
        # feature_flag_enabled = is_feature_enabled_sync(
        #     feature_name=FeatureFlags.VIDEO_INGESTION_FEATURE
        # )

        # if not feature_flag_enabled:
        #     logger.debug("Video ingestion disabled via Flagsmith feature flag")
        #     return False

        logger.debug("Video ingestion enabled via both settings and feature flag")
        return True

    except Exception as e:
        logger.warning(f"Error checking video ingestion enablement: {e}")
        # Fallback to settings-only check if feature flag system fails
        return getattr(settings, "ENABLE_VIDEO_INGESTION", False)


class FeatureFlags:
    """
    Feature flag constants for better code organization and typo prevention.

    Define your feature flag names here to avoid string literals throughout the codebase.
    """

    GROOVE_CONNECTOR_FEATURE = "groove_connector_feature"
    VIDEO_INGESTION_FEATURE = "video_ingestion_feature"
    # Add more feature flags here as needed
    # EXAMPLE_FEATURE = "example_feature"
