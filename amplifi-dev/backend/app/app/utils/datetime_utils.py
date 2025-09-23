import datetime
import inspect
from enum import Enum
from typing import Optional
from uuid import UUID

from app.be_core import logger


def get_utcnow() -> datetime.datetime:
    """
    Get current UTC time as a timezone-naive datetime.
    This should be used instead of datetime.utcnow() or datetime.now() throughout the app.
    """
    # Get UTC time but return it without timezone info
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def ensure_naive_datetime(
    dt: Optional[datetime.datetime],
) -> Optional[datetime.datetime]:
    """
    Ensure a datetime is timezone-naive by removing any timezone info.
    """
    if dt is None:
        return None

    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)

    return dt


def serialize_datetime(obj):
    """
    Serialize various types to JSON-compatible formats.

    Args:
        obj: The object to serialize

    Returns:
        JSON serializable version of the object

    Raises:
        TypeError: If the object cannot be serialized
    """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif hasattr(obj, "__dict__"):
        # Handle objects that can be converted to dict
        try:
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception as e:
            # Log the error and continue to next serialization attempt
            logger.debug(f"Failed to serialize object {type(obj).__name__}: {str(e)}")

    # Provide detailed error information to help debugging
    obj_type = type(obj).__name__
    obj_repr = repr(obj)[:100]  # Truncate long representations
    obj_module = (
        inspect.getmodule(type(obj)).__name__
        if inspect.getmodule(type(obj))
        else "unknown"
    )

    error_msg = (
        f"Type not serializable: {obj_type} from {obj_module} module: {obj_repr}"
    )
    raise TypeError(error_msg)
