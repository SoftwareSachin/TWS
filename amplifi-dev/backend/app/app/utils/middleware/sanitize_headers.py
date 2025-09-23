import re

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


def sanitize_header_value(value: str) -> str:
    """
    Remove CRLF characters to prevent HTTP response splitting.
    """
    return re.sub(r"[\r\n]", "", value)


class HeaderSanitizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # Copy existing headers, sanitize them
        original_headers = dict(response.headers)
        for key in original_headers:
            sanitized_value = sanitize_header_value(original_headers[key])
            response.headers[key] = sanitized_value  # overwrite safely

        return response
