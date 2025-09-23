import logfire
from logfire.exceptions import LogfireConfigError

from app.be_core.config import settings
from app.be_core.logger import logger


class LogfireManager:
    """Manages logfire configuration and initialization"""

    _instance = None
    _configured = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            self._configure_logfire()

    def _configure_logfire(self):
        """Configure logfire with error handling for test environments"""
        try:
            if hasattr(settings, "LOGFIRE_API_KEY") and settings.LOGFIRE_API_KEY:
                logfire.configure(
                    token=settings.LOGFIRE_API_KEY,
                    environment=settings.DEPLOYED_ENV,
                    scrubbing=logfire.ScrubbingOptions(
                        callback=self.scrubbing_callback
                    ),
                )
                self._configured = True
                logger.info("Logfire configured successfully")
            else:
                logger.warning("LOGFIRE_API_KEY not found in settings")
        except LogfireConfigError as e:
            logger.warning(f"Logfire configuration failed: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error configuring Logfire: {e}")

    @staticmethod
    def scrubbing_callback(m: logfire.ScrubMatch):
        """Custom scrubbing callback to preserve chat_session_id"""
        if m.path == ("attributes", "chat_session_id"):
            return m.value
        return None

    @property
    def is_configured(self) -> bool:
        """Check if logfire is properly configured"""
        return self._configured

    def instrument_pydantic_ai(self, agent):
        """Instrument PydanticAI agent with logfire if configured"""
        if self._configured:
            try:
                logfire.instrument_pydantic_ai(agent)
                logger.info("PydanticAI agent instrumented with logfire")
            except Exception as e:
                logger.warning(f"Failed to instrument PydanticAI agent: {e}")
        else:
            logger.warning("Logfire not configured, skipping agent instrumentation")

    def span(self, name: str, **kwargs):
        """Create a logfire span if configured, otherwise return a no-op context manager"""
        if self._configured:
            return logfire.span(name, **kwargs)
        else:
            # Return a no-op context manager for tests
            from contextlib import nullcontext

            return nullcontext()


# Create a singleton instance
logfire_manager = LogfireManager()
