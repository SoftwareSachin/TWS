"""
Unit tests for OpenAI retry logic functionality.
Tests the centralized retry decorators and rate limiting handling.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding

from app.utils.openai_utils import (
    generate_embedding_with_retry,
    retry_openai_call,
)


class TestRetryLogic:
    """Test suite for OpenAI retry logic"""

    def test_retry_decorator_success_first_attempt(self):
        """Test that successful API calls work without retry"""

        @retry_openai_call()
        def mock_api_call():
            return {"result": "success"}

        result = mock_api_call()
        assert result == {"result": "success"}

    def test_retry_decorator_rate_limit_recovery(self):
        """Test that 429 rate limit errors trigger retry with success"""
        call_count = 0

        @retry_openai_call()
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Simulate rate limit error with retry-after header
                error = Exception("Error code: 429")
                error.response = Mock()
                error.response.headers = {"retry-after": "1"}
                raise error
            return {"result": "success_after_retry"}

        result = mock_api_call()
        assert result == {"result": "success_after_retry"}
        assert call_count == 3  # Failed twice, succeeded on third

    def test_retry_decorator_max_retries_exceeded(self):
        """Test that max retries are respected"""
        call_count = 0

        @retry_openai_call()
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            # Always fail with rate limit
            error = Exception("Error code: 429")
            error.response = Mock()
            error.response.headers = {"retry-after": "1"}
            raise error

        with pytest.raises(
            Exception
        ):  # Should raise the rate limit exception after max retries
            mock_api_call()

        # Should try 4 times total (initial + 3 retries)
        assert call_count == 4

    def test_retry_decorator_non_rate_limit_error(self):
        """Test that non-429 errors don't trigger retry"""
        call_count = 0

        @retry_openai_call()
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            raise Exception("Error code: 500")  # Not a rate limit error

        with pytest.raises(Exception) as exc_info:
            mock_api_call()

        assert "Error code: 500" in str(exc_info.value)
        assert call_count == 1  # No retry for non-rate-limit errors

    def test_retry_decorator_with_suggested_delay(self):
        """Test that API-suggested retry delays are respected"""
        call_count = 0
        start_time = time.time()

        @retry_openai_call()
        def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with suggested 2-second delay
                error = Exception("Error code: 429")
                error.response = Mock()
                error.response.headers = {"retry-after": "2"}
                raise error
            return {"result": "success"}

        result = mock_api_call()
        elapsed = time.time() - start_time

        assert result == {"result": "success"}
        assert call_count == 2
        # Should have waited at least 2 seconds (with some tolerance)
        assert elapsed >= 1.8

    @patch("app.utils.openai_utils.get_openai_client")
    def test_generate_embedding_with_retry_success(self, mock_get_client):
        """Test successful embedding generation with retry wrapper"""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Create a realistic embedding response
        mock_embedding = Embedding(
            embedding=[0.1, 0.2, 0.3], index=0, object="embedding"
        )
        mock_response = CreateEmbeddingResponse(
            data=[mock_embedding],
            model="text-embedding-3-large",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
        mock_client.embeddings.create.return_value = mock_response

        result = generate_embedding_with_retry("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch("app.utils.openai_utils.get_openai_client")
    def test_generate_embedding_with_retry_rate_limit(self, mock_get_client):
        """Test embedding generation with rate limit retry"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Fail first two attempts with rate limit
                error = Exception("Error code: 429")
                error.response = Mock()
                error.response.headers = {"retry-after": "1"}
                raise error

            # Succeed on third attempt
            mock_embedding = Embedding(
                embedding=[0.4, 0.5, 0.6], index=0, object="embedding"
            )
            return CreateEmbeddingResponse(
                data=[mock_embedding],
                model="text-embedding-3-large",
                object="list",
                usage={"prompt_tokens": 10, "total_tokens": 10},
            )

        mock_client.embeddings.create.side_effect = mock_create

        result = generate_embedding_with_retry("test text")

        assert result == [0.4, 0.5, 0.6]
        assert call_count == 3

    def test_error_message_parsing(self):
        """Test that different error message formats are properly detected"""

        @retry_openai_call()
        def test_error_detection(error_msg):
            raise Exception(error_msg)

        # Test various rate limit error formats
        rate_limit_errors = [
            "Error code: 429",
            "Rate limit exceeded: Error code: 429",
            "OpenAI API error: 429 Too Many Requests",
            "Error analyzing image: Error code: 429 - Rate limit exceeded",
        ]

        for error_msg in rate_limit_errors:
            call_count = 0

            @retry_openai_call()
            def mock_call():
                nonlocal call_count
                call_count += 1
                if call_count <= 1:
                    error = Exception(error_msg)
                    error.response = Mock()
                    error.response.headers = {"retry-after": "1"}
                    raise error
                return "success"

            result = mock_call()
            assert result == "success"
            assert call_count == 2  # Should have retried


class TestRateLimitScenarios:
    """Test real-world rate limiting scenarios"""

    @patch("app.utils.openai_utils.get_openai_client")
    def test_concurrent_requests_rate_limiting(self, mock_get_client):
        """Test behavior under concurrent rate limit scenarios"""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Simulate staggered failures and recovery
        failure_pattern = [True, True, False]  # Fail twice, then succeed
        call_index = 0

        def mock_create(*args, **kwargs):
            nonlocal call_index
            should_fail = (
                call_index < len(failure_pattern) and failure_pattern[call_index]
            )
            call_index += 1

            if should_fail:
                error = Exception("Error code: 429 - Rate limit exceeded")
                error.response = Mock()
                error.response.headers = {"retry-after": "2"}
                raise error

            # Success case
            mock_embedding = Embedding(
                embedding=[0.7, 0.8, 0.9], index=0, object="embedding"
            )
            return CreateEmbeddingResponse(
                data=[mock_embedding],
                model="text-embedding-3-large",
                object="list",
                usage={"prompt_tokens": 10, "total_tokens": 10},
            )

        mock_client.embeddings.create.side_effect = mock_create

        result = generate_embedding_with_retry("concurrent test")
        assert result == [0.7, 0.8, 0.9]

    def test_exponential_backoff_timing(self):
        """Test that exponential backoff timing works correctly"""
        call_times = []
        start_time = time.time()

        @retry_openai_call()
        def mock_api_call():
            call_times.append(time.time() - start_time)
            if len(call_times) <= 2:
                error = Exception("Error code: 429")
                error.response = Mock()
                error.response.headers = {}  # No retry-after, use exponential backoff
                raise error
            return "success"

        result = mock_api_call()
        assert result == "success"
        assert len(call_times) == 3

        # Check that delays approximately follow exponential backoff (2, 4 seconds)
        # Allow some tolerance for execution time
        assert call_times[1] - call_times[0] >= 1.8  # ~2 seconds
        assert call_times[2] - call_times[1] >= 3.8  # ~4 seconds


if __name__ == "__main__":
    # Run with: python -m pytest test_openai_retry.py -v
    pass
