import re
import time
from typing import List

from openai import APIError, AzureOpenAI, RateLimitError

from app.api.deps import get_async_gpt4o_client
from app.be_core.config import settings
from app.be_core.logger import logger

client_4o = get_async_gpt4o_client()


def _is_rate_limit_error(error_str: str) -> bool:
    """Check if the error is a rate limit (429) error"""
    return (
        "429" in error_str
        or "rate limit" in error_str.lower()
        or "quota" in error_str.lower()
        or "exceeded" in error_str.lower()
        or "too many requests" in error_str.lower()
    )


def _extract_retry_delay(error_str: str) -> int:
    """Extract retry delay from error message, default to exponential backoff"""
    # Look for "retry after X seconds" pattern
    match = re.search(r"retry after (\d+) seconds?", error_str.lower())
    if match:
        return int(match.group(1))
    return None


def retry_openai_call(max_retries: int = 3, base_delay: int = 2):
    """
    Decorator to retry OpenAI API calls with exponential backoff for rate limit errors

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    last_exception = e

                    # Only retry for rate limit errors
                    if not _is_rate_limit_error(error_str):
                        logger.error(
                            f"Non-retryable error in {func.__name__}: {error_str}"
                        )
                        raise e

                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) reached for {func.__name__}: {error_str}"
                        )
                        raise e

                    # Calculate delay - check error response headers first, then error message, then exponential backoff
                    suggested_delay = None

                    # Try to get delay from response headers
                    if (
                        hasattr(e, "response")
                        and hasattr(e.response, "headers")
                        and e.response.headers
                    ):
                        retry_after = e.response.headers.get("retry-after")
                        if retry_after:
                            try:
                                suggested_delay = int(retry_after)
                            except (ValueError, TypeError):
                                pass

                    # If no header delay, try to extract from error message
                    if suggested_delay is None:
                        suggested_delay = _extract_retry_delay(error_str)

                    # Use suggested delay or exponential backoff
                    if suggested_delay:
                        delay = suggested_delay
                    else:
                        delay = base_delay * (2**attempt)  # Exponential backoff

                    logger.warning(
                        f"Rate limit error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {delay} seconds: {error_str}"
                    )
                    time.sleep(delay)

            # Should never reach here, but just in case
            raise last_exception

        return wrapper

    return decorator


def generate_embedding_with_retry(text: str) -> List[float]:
    """Generate embedding with retry logic for rate limit errors"""

    @retry_openai_call()
    def _make_embedding_call():
        # Call the core embedding function without error handling
        return _generate_embedding_core(text)

    try:
        return _make_embedding_call()
    except Exception as e:
        logger.error(f"Error generating embedding after retries: {str(e)}")
        # Return empty embedding on failure to prevent chunk creation from failing
        return [0.0] * settings.EMBEDDING_DIMENSIONS


def _generate_embedding_core(text: str) -> List[float]:
    """Core embedding generation without error handling - used by retry wrapper"""
    # Handle case where a dictionary is passed instead of a string
    if isinstance(text, dict):
        if "object_name" in text:
            # If it's an object detection result, use the object name
            text = text.get("object_name", "")
        else:
            # Convert dict to string representation for embedding
            text = str(text)
        logger.warning(
            f"Dictionary passed to generate_embedding, converted to string: {text[:50]}..."
        )

    if not text or (isinstance(text, str) and not text.strip()):
        logger.warning("Empty text provided for embedding generation")
        return [0.0] * settings.EMBEDDING_DIMENSIONS

    client = get_openai_client()
    embedding_model = settings.EMBEDDING_MODEL_NAME

    response = client.embeddings.create(
        input=text, model=embedding_model, dimensions=settings.EMBEDDING_DIMENSIONS
    )
    return response.data[0].embedding


def chat_completion_with_retry(
    messages: List[dict],
    model: str = None,
    max_tokens: int = None,
    temperature: float = None,
    client=None,
    **kwargs,
) -> str:
    """Generate chat completion with retry logic for rate limit errors"""

    @retry_openai_call()
    def _make_chat_completion_call():
        # Use provided client or get default one
        openai_client = client or get_openai_client()

        # Set up parameters with defaults
        completion_params = {
            "model": model or settings.AZURE_GPT_4o_DEPLOYMENT_NAME,
            "messages": messages,
        }

        # Add optional parameters if provided
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens
        if temperature is not None:
            completion_params["temperature"] = temperature

        # Add any additional kwargs
        completion_params.update(kwargs)

        response = openai_client.chat.completions.create(**completion_params)
        return response.choices[0].message.content.strip()

    try:
        return _make_chat_completion_call()
    except Exception as e:
        logger.error(f"Error generating chat completion after retries: {str(e)}")
        # Return a fallback response
        return "Unable to generate response due to API error."


def get_openai_client() -> AzureOpenAI:
    """Get an initialized OpenAI client"""
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_GPT_4o_URL,
        api_key=settings.AZURE_GPT_4o_KEY,
        api_version=settings.AZURE_GPT_4o_VERSION,
    )


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for the provided text (with error handling for backward compatibility)"""
    try:
        return _generate_embedding_core(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return empty embedding in case of error
        return [0.0] * settings.EMBEDDING_DIMENSIONS


async def generate_embedding_async(text: str):
    try:
        response = await client_4o.embeddings.create(
            input=text,
            model=settings.EMBEDDING_MODEL_NAME,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        return response.data[0].embedding
    except RateLimitError as e:
        raise ValueError("Rate limit exceeded. Please wait and try again.") from e
    except APIError as e:
        raise ValueError(f"OpenAI API error: {str(e)}") from e
    except Exception as e:
        raise ValueError(f"Embedding generation failed: {str(e)}") from e


def generate_text2sql_embedding(text: str) -> List[float]:
    """Generate embedding for text2sql with fixed 1024 dimensions"""
    try:
        # Handle case where a dictionary is passed instead of a string
        if isinstance(text, dict):
            if "object_name" in text:
                # If it's an object detection result, use the object name
                text = text.get("object_name", "")
            else:
                # Convert dict to string representation for embedding
                text = str(text)
            logger.warning(
                f"Dictionary passed to generate_embedding, converted to string: {text[:50]}..."
            )

        if not text or (isinstance(text, str) and not text.strip()):
            logger.warning("Empty text provided for embedding generation")
            return [0.0] * 1024  # Fixed dimension for text2sql

        client = get_openai_client()
        embedding_model = settings.EMBEDDING_MODEL_NAME

        response = client.embeddings.create(
            input=text, model=embedding_model, dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        # Return empty embedding in case of error
        return [0.0] * 1024  # Fixed dimension for text2sql
