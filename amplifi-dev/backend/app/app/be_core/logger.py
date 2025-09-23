import logging
import typing as t
from logging import DEBUG, INFO

from app.be_core.config import settings


class LoggingEndpointFilter(logging.Filter):
    def __init__(
        self,
        path: str,
        *args: t.Any,
        **kwargs: t.Any,
    ):
        super().__init__(*args, **kwargs)
        self._path = path

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find(self._path) == -1


if settings.DEPLOYED_ENV == "local":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, force=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(DEBUG)
elif settings.DEPLOYED_ENV == "azure_dev" or settings.DEPLOYED_ENV == "azure_prod":
    # from opencensus.ext.azure.log_exporter import AzureLogHandler
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry.sdk.resources import Resource

    configure_azure_monitor(
        # Set logger_name to the name of the logger you want to capture logging telemetry with
        logger_name="amplifi-be-logger",
        enable_live_metrics=False,
        connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING,
        resource=Resource.create({"service.name": "amplifi-be"}),
    )

    # Configure logging
    logger = logging.getLogger("amplifi-be-logger")
    logger.addFilter(LoggingEndpointFilter(path="/docs"))

    if settings.DEPLOYED_ENV == "azure_dev":
        logger.setLevel(DEBUG)
    elif settings.DEPLOYED_ENV == "azure_prod":
        logger.setLevel(INFO)
    # logger.addHandler(
    #     AzureLogHandler(
    #         connection_string=settings.APPLICATIONINSIGHTS_CONNECTION_STRING
    #     )
    # )

# Disable few logs at unicorn level
uvicorn_logger = logging.getLogger("uvicorn.access")
if uvicorn_logger:
    uvicorn_logger.addFilter(LoggingEndpointFilter(path="/docs"))
    uvicorn_logger.addFilter(LoggingEndpointFilter(path="/health"))
