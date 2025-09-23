import asyncio
import time
from datetime import datetime
from functools import wraps
from typing import Dict, Literal, Optional

from opencensus.ext.azure.log_exporter import AzureLogHandler

from app.be_core.config import settings
from app.be_core.logger import logger

# Configure Azure handler once at module level
_azure_handler_configured = False


def _ensure_azure_handler():
    """Ensure Azure handler is configured once."""
    global _azure_handler_configured

    if not _azure_handler_configured and settings.APPLICATIONINSIGHTS_CONNECTION_STRING:
        try:
            # Check if handler already exists
            has_azure_handler = any(
                isinstance(handler, AzureLogHandler) for handler in logger.handlers
            )

            if not has_azure_handler:
                logger.addHandler(
                    AzureLogHandler(
                        connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING,
                        logging_sampling_rate=1.0,
                    )
                )
            _azure_handler_configured = True
        except Exception as e:
            logger.error(f"Failed to initialize Azure Monitor: {str(e)}")


class IngestionBenchmark:
    """Utility class for benchmarking different types of ingestion operations."""

    def __init__(
        self,
        file_id: str,
        ingestion_type: Literal["pdf", "image", "split"] = "pdf",
        split_id: Optional[str] = None,
        parent_document_id: Optional[str] = None,
    ):
        self.file_id = file_id
        self.split_id = split_id
        self.parent_document_id = parent_document_id
        self.ingestion_type = ingestion_type
        self.benchmarks: Dict[str, float] = {}
        self.start_times: Dict[str, float] = {}

        # Check if benchmarking is enabled
        self.benchmark_enabled = settings.ENABLE_BENCHMARK_LOGS

        # Only setup Azure logging if benchmarking is enabled
        if self.benchmark_enabled:
            _ensure_azure_handler()
            self.azure_logger = logger
        else:
            self.azure_logger = None

    def start(self, stage: str):
        """Start timing a specific stage."""
        if self.benchmark_enabled:
            self.start_times[stage] = time.time()

    def end(self, stage: str):
        """End timing a specific stage and record duration."""
        if not self.benchmark_enabled:
            return

        if stage in self.start_times:
            duration = time.time() - self.start_times[stage]
            self.benchmarks[stage] = duration
            self._log_metric(stage, duration)

    def _log_metric(self, stage: str, duration: float):
        """Log a benchmark metric to Azure Monitor."""
        # Skip logging if benchmarking is disabled
        if not self.benchmark_enabled or not self.azure_logger:
            return

        try:
            # Azure Monitor custom dimensions for filtering
            custom_dimensions = {
                "logType": "ingestion_benchmark",
                "ingestionType": self.ingestion_type,
                "fileId": self.file_id,
                "splitId": self.split_id,
                "parentDocumentId": self.parent_document_id,
                "stage": stage,
                "duration": round(duration, 3),
                "timestamp": datetime.now().isoformat(),
            }

            self.azure_logger.info(
                f"Ingestion Benchmark: {self.ingestion_type} - {stage} - {round(duration, 3)}s",
                extra=custom_dimensions,
            )

        except Exception as e:
            logger.error(f"Failed to send benchmark to Azure Monitor: {str(e)}")

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all benchmarks."""
        if not self.benchmark_enabled:
            return {}
        return {k: round(v, 3) for k, v in self.benchmarks.items()}


def benchmark_stage(stage_name: str):
    """Decorator to benchmark a specific stage."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Skip benchmarking if disabled
            if not settings.ENABLE_BENCHMARK_LOGS:
                return await func(*args, **kwargs)

            file_id = kwargs.get("file_id") or getattr(args[0], "file_id", None)
            split_id = kwargs.get("split_id")
            benchmark = IngestionBenchmark(
                str(file_id), str(split_id) if split_id else None
            )

            benchmark.start(stage_name)
            result = await func(*args, **kwargs)
            benchmark.end(stage_name)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Skip benchmarking if disabled
            if not settings.ENABLE_BENCHMARK_LOGS:
                return func(*args, **kwargs)

            file_id = kwargs.get("file_id") or getattr(args[0], "file_id", None)
            split_id = kwargs.get("split_id")
            benchmark = IngestionBenchmark(
                str(file_id), str(split_id) if split_id else None
            )

            benchmark.start(stage_name)
            result = func(*args, **kwargs)
            benchmark.end(stage_name)

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
