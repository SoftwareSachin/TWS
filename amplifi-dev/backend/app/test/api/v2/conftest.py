"""
Video ingestion test configuration and fixtures.
Provides common fixtures and utilities for video ingestion tests.
"""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

"""Fixtures specific to v2 API tests."""

import random

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

# V2 API base URL
url_v2 = "http://localhost:8085/api/v2"


@pytest_asyncio.fixture(scope="module")
async def test_client_v2():
    """Test client configured for V2 API endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url=url_v2) as client:
        yield client


@pytest_asyncio.fixture(scope="module")
async def test_client_admin_v2(test_client_v2, auth_token):
    """Authenticated test client for V2 API with admin permissions."""
    test_client_v2.headers.update({"Authorization": f"Bearer {auth_token}"})
    yield test_client_v2


@pytest_asyncio.fixture(scope="module")
async def dataset_id_v2(test_client_admin_v2, workspace_id, upload_file_to_workspace):
    """
    Create a dataset specifically for v2 API testing.
    """
    unique_suffix = random.randint(100, 999)
    file_id = upload_file_to_workspace["id"]
    dataset_data = {
        "name": f"V2 Test Dataset {unique_suffix}",
        "description": "Dataset created for V2 API testing",
        "file_ids": [file_id],
    }

    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset", json=dataset_data
    )
    response.raise_for_status()
    dataset_id = response.json()["data"]["id"]
    yield dataset_id


@pytest_asyncio.fixture(scope="module")
async def workspace_tool_id_v2(test_client_admin, workspace_id, tool_id):
    """
    Create a workspace tool specifically for v2 API testing.
    Uses V1 client since tool assignment might only be available in V1.
    """
    import time

    request_data = {
        "name": f"V2 Test Tool {int(time.time())}",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    response.raise_for_status()
    yield response.json()["data"]["id"]


@pytest_asyncio.fixture(scope="module")
async def agent_id_v2(test_client_admin, workspace_id, workspace_tool_id_v2):
    """
    Create an agent specifically for v2 API testing.
    Uses V1 client since agent creation might only be available in V1.
    """
    agent_data = {
        "name": "Test Agent V2",
        "description": "Agent for V2 API testing",
        "prompt_instructions": "Follow these instructions for V2 testing",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt for V2 testing",
        "memory_enabled": True,
        "agent_metadata": {"test": "v2"},
        "workspace_tool_ids": [workspace_tool_id_v2],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=agent_data
    )
    response.raise_for_status()
    yield response.json()["data"]["id"]


@pytest_asyncio.fixture(scope="module")
async def chatapp_id_v2(test_client_admin_v2, workspace_id, agent_id_v2):
    """
    Create a chatapp specifically for v2 API testing.
    """
    unique_suffix = random.randint(100, 999)
    chatapp_data = {
        "name": f"V2 Test ChatApp {unique_suffix}",
        "description": "ChatApp created for V2 API testing",
        "agent_id": agent_id_v2,
    }

    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/chat_app", json=chatapp_data
    )
    response.raise_for_status()
    chatapp_id = response.json()["data"]["id"]
    yield chatapp_id


@pytest_asyncio.fixture(scope="module")
async def workspace_id_v2(workspace_id):
    """
    Alias for workspace_id to maintain V2 API test consistency.
    V2 API uses the same workspace concept as V1.
    """
    yield workspace_id


@pytest_asyncio.fixture(scope="function")
async def upload_video_to_workspace(test_client_admin, workspace_id):
    """Upload video file to workspace using V1 API"""
    import io

    # Create a test video file
    video_content = b"fake video content " * 1000  # ~18KB video file

    files = [
        (
            "files",
            ("test_video.mp4", io.BytesIO(video_content), "video/mp4"),
        ),
    ]

    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/file_upload",
        files=files,
    )

    response.raise_for_status()
    return response.json()["data"][0]  # Return the file dict


@pytest.fixture
def temp_video_dir():
    """Create a temporary directory for video test files

    NOTE: Used for mock video testing only - no actual video processing occurs
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_video_file(temp_video_dir):
    """Create a mock video file for testing

    NOTE: Creates dummy bytes, not actual video - used because GitHub Actions
    cannot run GPU-based video processing in pytest workflow
    """
    video_path = Path(temp_video_dir) / "test_video.mp4"
    # Create a dummy video file with some content
    video_content = b"fake video content for testing " * 1000  # ~30KB
    video_path.write_bytes(video_content)
    return str(video_path)


@pytest.fixture
def mock_large_video_file(temp_video_dir):
    """Create a mock large video file for testing

    NOTE: Creates dummy bytes, not actual video - used for size validation tests
    without requiring actual video processing in GitHub Actions pytest workflow
    """
    video_path = Path(temp_video_dir) / "large_video.mp4"
    # Create a large dummy video file (~60MB)
    video_content = b"fake large video content " * (60 * 1024 * 20)
    video_path.write_bytes(video_content)
    return str(video_path)


@pytest.fixture
def mock_video_segments():
    """Mock video segment data for testing"""
    return {
        0: {
            "segment_name": "test_video_segment_0",
            "transcript": "Hello, this is the first segment of the video.",
            "caption": "A person speaking in front of a computer screen.",
            "start_time": 0,
            "end_time": 30,
            "combined_content": "Hello, this is the first segment of the video. A person speaking in front of a computer screen.",
        },
        1: {
            "segment_name": "test_video_segment_1",
            "transcript": "This is the second segment with more content.",
            "caption": "The person continues speaking with charts visible in the background.",
            "start_time": 30,
            "end_time": 60,
            "combined_content": "This is the second segment with more content. The person continues speaking with charts visible in the background.",
        },
    }


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing"""
    mock_model = Mock()

    # Mock transcription results
    mock_segments = [
        Mock(text="Hello, this is a test transcription."),
        Mock(text="This is another segment of transcribed text."),
    ]
    mock_model.transcribe.return_value = (mock_segments, None)

    return mock_model


@pytest.fixture
def mock_caption_model():
    """Mock caption model and tokenizer for testing"""
    mock_model = Mock()
    mock_tokenizer = Mock()

    # Mock captioning results
    mock_model.chat.return_value = (
        "This is a test caption describing the video content."
    )

    return mock_model, mock_tokenizer


@pytest.fixture
def mock_video_clip():
    """Mock VideoFileClip for testing"""
    mock_clip = Mock()
    mock_clip.duration = 60.0
    mock_clip.fps = 30

    # Mock audio
    mock_audio = Mock()
    mock_audio.duration = 60.0
    mock_clip.audio = mock_audio

    # Mock frame extraction
    import numpy as np

    test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock_clip.get_frame.return_value = test_frame

    # Mock subclip creation
    mock_subclip = Mock()
    mock_clip.subclip.return_value = mock_subclip

    return mock_clip


@pytest.fixture
def video_ingestion_config():
    """Video ingestion configuration for testing"""
    return {
        "video_segment_length": 30,
        "rough_num_frames_per_segment": 5,
        "audio_output_format": "mp3",
        "video_output_format": "mp4",
        "enable_captioning": True,
        "enable_transcription": True,
        "whisper_model_name": "Systran/faster-distil-whisper-large-v3",
        "caption_model_name": "openbmb/MiniCPM-V-2_6-int4",
        "transcription_max_workers": 3,
        "captioning_batch_size": 3,
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    mock_client = Mock()
    mock_client.exists.return_value = False
    mock_client.set.return_value = True
    mock_client.get.return_value = None
    mock_client.delete.return_value = True
    mock_client.expire.return_value = True
    return mock_client


@pytest.fixture
def mock_database_session():
    """Mock database session for testing"""
    mock_session = Mock()
    mock_session.add = Mock()
    mock_session.commit = Mock()
    mock_session.refresh = Mock()
    mock_session.query.return_value.filter.return_value.first.return_value = None
    return mock_session


@pytest.fixture
def mock_document():
    """Mock document for testing"""
    mock_doc = Mock()
    mock_doc.id = uuid.uuid4()
    mock_doc.file_id = str(uuid.uuid4())
    mock_doc.dataset_id = str(uuid.uuid4())
    mock_doc.document_type = "Video"
    mock_doc.processing_status = "processing"
    mock_doc.metadata = {"test": "metadata"}
    return mock_doc


@pytest.fixture
def mock_document_chunks():
    """Mock document chunks for testing"""
    chunks = []
    for i in range(2):
        chunk = Mock()
        chunk.id = uuid.uuid4()
        chunk.content = f"Video segment {i}: Test content for chunk {i}"
        chunk.metadata = {
            "segment_name": f"test_video_segment_{i}",
            "start_time": i * 30,
            "end_time": (i + 1) * 30,
            "transcript": f"Test transcript {i}",
            "caption": f"Test caption {i}",
        }
        chunk.chunk_type = "video_segment"
        chunks.append(chunk)
    return chunks


@pytest.fixture
def enable_video_ingestion():
    """Enable video ingestion for testing"""
    with patch("app.utils.feature_flags.is_video_ingestion_enabled", return_value=True):
        yield


@pytest.fixture
def disable_video_ingestion():
    """Disable video ingestion for testing"""
    with patch(
        "app.utils.feature_flags.is_video_ingestion_enabled", return_value=False
    ):
        yield


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability for testing"""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA unavailability for testing"""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_video_processing_success():
    """Mock successful video processing pipeline"""
    with (
        patch("app.api.video_ingestion_task.split_video") as mock_split,
        patch("app.api.video_ingestion_task.speech_to_text") as mock_transcribe,
        patch("app.api.video_ingestion_task.segment_caption") as mock_caption,
        patch("app.api.video_ingestion_task.merge_segment_information") as mock_merge,
    ):

        # Mock video splitting
        mock_split.return_value = (
            {0: "segment_0", 1: "segment_1"},
            {
                0: {"timestamp": (0, 30), "frame_times": [0, 15, 30]},
                1: {"timestamp": (30, 60), "frame_times": [30, 45, 60]},
            },
        )

        # Mock transcription
        mock_transcribe.return_value = {
            0: "First segment transcript",
            1: "Second segment transcript",
        }

        # Mock captioning
        def mock_caption_side_effect(*args, **kwargs):
            caption_result = kwargs["caption_result"]
            caption_result[0] = "First segment caption"
            caption_result[1] = "Second segment caption"

        mock_caption.side_effect = mock_caption_side_effect

        # Mock merging
        mock_merge.return_value = [
            {
                "segment_name": "segment_0",
                "transcript": "First segment transcript",
                "caption": "First segment caption",
                "start_time": 0,
                "end_time": 30,
                "combined_content": "First segment transcript. First segment caption",
            },
            {
                "segment_name": "segment_1",
                "transcript": "Second segment transcript",
                "caption": "Second segment caption",
                "start_time": 30,
                "end_time": 60,
                "combined_content": "Second segment transcript. Second segment caption",
            },
        ]

        yield {
            "split": mock_split,
            "transcribe": mock_transcribe,
            "caption": mock_caption,
            "merge": mock_merge,
        }


@pytest.fixture
def mock_openai_embedding():
    """Mock OpenAI embedding generation"""
    with patch(
        "app.utils.openai_utils.generate_embedding_with_retry"
    ) as mock_embedding:
        # Return a mock embedding vector (1536 dimensions)
        mock_embedding.return_value = [0.1] * 1536
        yield mock_embedding


@pytest.fixture
def video_test_files():
    """Create various video test files with different formats"""
    temp_dir = tempfile.mkdtemp()

    files = {}
    formats = [
        ("test.mp4", "video/mp4"),
        ("test.avi", "video/x-msvideo"),
        ("test.mov", "video/quicktime"),
        ("test.webm", "video/webm"),
        ("test.mkv", "video/x-matroska"),
        ("test.flv", "video/x-flv"),
        ("test.wmv", "video/x-ms-wmv"),
    ]

    for filename, mimetype in formats:
        file_path = Path(temp_dir) / filename
        file_path.write_bytes(b"fake video content " * 100)
        files[filename] = {"path": str(file_path), "mimetype": mimetype}

    yield files

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


class VideoTestHelpers:
    """Helper class for video ingestion tests"""

    @staticmethod
    def create_test_video_bytes(size_mb=1):
        """Create test video content as bytes"""
        content = b"fake video content for testing " * (size_mb * 1024 * 32)
        return content

    @staticmethod
    def create_mock_video_results(num_segments=2):
        """Create mock video processing results"""
        segments = []
        for i in range(num_segments):
            segments.append(
                {
                    "segment_name": f"test_video_segment_{i}",
                    "transcript": f"Test transcript for segment {i}",
                    "caption": f"Test caption for segment {i}",
                    "start_time": i * 30,
                    "end_time": (i + 1) * 30,
                    "combined_content": f"Test transcript for segment {i}. Test caption for segment {i}",
                }
            )

        return {
            "segments": segments,
            "total_duration": num_segments * 30,
            "processing_time": 45.5,
        }

    @staticmethod
    def assert_video_chunk_structure(chunk):
        """Assert that a video chunk has the expected structure"""
        required_fields = ["id", "content", "metadata"]
        for field in required_fields:
            assert hasattr(chunk, field), f"Chunk missing required field: {field}"

        # Check metadata structure for video chunks
        metadata = chunk.metadata
        video_metadata_fields = ["segment_name", "start_time", "end_time"]
        for field in video_metadata_fields:
            assert field in metadata, f"Chunk metadata missing field: {field}"

    @staticmethod
    def assert_video_ingestion_result(result):
        """Assert that a video ingestion result has the expected structure"""
        required_fields = ["success", "status", "document_type"]
        for field in required_fields:
            assert field in result, f"Result missing required field: {field}"

        assert result["document_type"] == "Video"

        if result["success"]:
            success_fields = ["segments_processed", "total_duration"]
            for field in success_fields:
                assert field in result, f"Successful result missing field: {field}"


@pytest.fixture
def video_test_helpers():
    """Provide video test helpers"""
    return VideoTestHelpers
