"""
Video ingestion test configuration and fixtures.
Provides common fixtures and utilities for video ingestion tests.
"""

import asyncio
import io
import json
import os
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path
from test.test_files.aristotle import aristotle_bytes
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image
from typing_extensions import deprecated

from app.be_core.config import settings
from app.be_core.logger import logger
from app.main import app
from app.models.chat_app_generation_config_model import ChatAppGenerationConfigBase
from app.schemas.chat_schema import (
    IChatAppCreate,
    IChatAppRead,
    IChatAppTypeEnum,
    IChatSessionRead,
)
from app.schemas.user_schema import IUserCreate

url = "http://localhost:8085/api/v2"


@pytest.fixture(scope="session")
def event_loop(request) -> Generator:  # noqa: indirect usage
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def test_client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url=url) as client:
        yield client


@pytest_asyncio.fixture(scope="module")
async def auth_token(test_client):
    form_data = {
        "username": settings.TEST_USER_EMAIL,
        "password": settings.TEST_USER_PASSWORD,
        "securityanswer[Project Name]": "Amplifi",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = await test_client.post(
        "/login/access-token", data=form_data, headers=headers
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    yield token


@pytest_asyncio.fixture(scope="module")
async def test_client_admin(test_client, auth_token):
    test_client.headers.update({"Authorization": f"Bearer {auth_token}"})
    yield test_client


@pytest_asyncio.fixture(scope="module")
async def test_client_v2():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost:8085/api/v2"
    ) as client:
        yield client


@pytest_asyncio.fixture(scope="module")
async def test_client_admin_v2(test_client_v2, auth_token):
    test_client_v2.headers.update({"Authorization": f"Bearer {auth_token}"})
    yield test_client_v2


@pytest_asyncio.fixture(scope="session")
async def organization_id():
    yield str(settings.TEST_ORG_UUID)


@pytest_asyncio.fixture(scope="module")
async def workspace_id(organization_id, test_client_admin):
    import time

    unique_suffix = f"{random.randint(100, 999)}-{int(time.time())}"  # More unique name
    unique_name = f"test-ws-{unique_suffix}"
    response = await test_client_admin.post(
        f"/organization/{organization_id}/workspace",
        json={
            "name": unique_name,
            "description": "Workspace for testing",
            "is_active": True,
        },
    )
    response.raise_for_status()
    workspace_id = response.json()["data"]["id"]
    yield workspace_id


@pytest_asyncio.fixture(scope="module")
async def user_id(test_client_admin, organization_id):
    user_data = IUserCreate(
        first_name="Test",
        last_name="User",
        email=f"testuser_{uuid.uuid4()}@example.com",
        password="password123",
        role_id=settings.TEST_ROLE_ID,
        organization_id=organization_id,
    )
    response = await test_client_admin.post(
        "/user", json=user_data.model_dump(mode="json")
    )
    response.raise_for_status()

    user_id = response.json()["data"]["id"]
    yield user_id


@pytest_asyncio.fixture(scope="module")
async def destination_id(organization_id, test_client_admin):
    response = await test_client_admin.post(
        f"/organization/{organization_id}/destination",
        json={
            "name": "test_destination",
            "description": "destination for testing",
            "is_active": True,
            "pg_vector": {
                "host": "database",
                "port": 5432,
                "database_name": "amplifi_db",
                "table_name": "sample_table",
                "username": "postgres",
                "password": "postgres",
            },
        },
    )
    response.raise_for_status()
    destination_id = response.json()["data"]["id"]
    yield destination_id


@pytest_asyncio.fixture(scope="module")
async def workflow_id(organization_id, workspace_id, destination_id, test_client_admin):
    new_workflow_body = {
        "name": "test_workflow",
        "description": "workflow for testing",
        "is_active": True,
        "workspace_id": workspace_id,
        "destination_id": destination_id,
        "schedule_config": {
            "cron_expression": "cron_expression str in schedule_config"
        },
    }

    response = await test_client_admin.post(
        f"/organization/{organization_id}/workflow",
        json=new_workflow_body,
    )
    response.raise_for_status()
    workflow_id = response.json()["data"]["id"]
    yield workflow_id


@deprecated("Use the new upload file functionality that supports multiple files")
@pytest_asyncio.fixture(scope="module")
async def upload_file_to_workspace(test_client_admin, workspace_id):
    file_content = aristotle_bytes
    file_like_object = io.BytesIO(file_content)
    file_like_object.name = "aristotle.pdf"

    files = [
        (
            "files",
            ("aristotle.pdf", io.BytesIO(file_content)),
        ),
    ]

    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/file_upload",
        files=files,
    )

    response.raise_for_status()
    yield response.json()["data"][0]  # Return the dict


@deprecated("Use the new upload file functionality that supports multiple files")
@pytest_asyncio.fixture(scope="module")
async def upload_pdf_to_workspace(test_client_admin, workspace_id):

    with open("test/test_files/Socrates_file.pdf", "rb") as pdf_file:
        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/file_upload",
            files=[("files", pdf_file)],
        )
    response.raise_for_status()
    yield response.json()["data"][0]  ##dict


@pytest_asyncio.fixture(scope="module")
async def dataset_id(test_client_admin, workspace_id, upload_file_to_workspace):
    file_id = upload_file_to_workspace["id"]
    dataset_name = f"Test Dataset {uuid.uuid4()}"  # Ensuring unique dataset name

    dataset_body = {
        "name": dataset_name,
        "description": "This is a test dataset",
        "file_ids": [file_id],  # Passing file id
    }

    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/dataset",
        json=dataset_body,
    )

    response.raise_for_status()
    yield response.json()["data"]["id"]


@pytest_asyncio.fixture(scope="module")
async def dataset_id_v2_empty(test_client_admin_v2, workspace_id):
    dataset_name = f"Test Dataset {uuid.uuid4()}"  # Ensuring unique dataset name

    dataset_body = {
        "name": dataset_name,
        "description": "This is a test dataset",
        "file_ids": [],
    }

    response = await test_client_admin_v2.post(
        f"/workspace/{workspace_id}/dataset",
        json=dataset_body,
    )
    response.raise_for_status()
    yield response.json()["data"]["id"]


@pytest_asyncio.fixture(scope="module")
async def dataset_id_generator(
    test_client_admin, workspace_id, upload_file_to_workspace
):
    async def dataset_id():
        file_id = upload_file_to_workspace["id"]
        dataset_body = {
            "name": "Test Dataset",
            "description": "This is a test dataset",
            "file_ids": [file_id],  # Passing file id
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset",
            json=dataset_body,
        )

        response.raise_for_status()
        return response.json()["data"]["id"]  # Returning dataset id

    yield dataset_id


@pytest_asyncio.fixture(scope="module")
async def ingest_dataset(test_client_admin, dataset_id):
    ingest_body = {"name": "Test Ingestion"}

    response = await test_client_admin.post(
        f"/dataset/{dataset_id}/ingest",
        json=ingest_body,
    )

    response.raise_for_status()
    yield response.json()


@pytest_asyncio.fixture(scope="module")
async def ingest_dataset_by_id(test_client_admin: AsyncClient):
    ingest_body = {"name": "Test Ingestion"}

    async def ingest_dataset(dataset_id):
        response = await test_client_admin.post(
            f"/dataset/{dataset_id}/ingest",
            json=ingest_body,
        )
        response.raise_for_status()

        for i in range(300):
            response = await test_client_admin.get(
                f"/dataset/{dataset_id}/ingestion_status"
            )
            files = response.json()["data"]
            if all(file["status"] == "Success" for file in files):
                return response.json()
            for file in files:
                if file["status"] in ["Failed", "Exception"]:
                    raise Exception("File Upload Failed")
            time.sleep(2)
            if i % 30 == 0:
                print(f"Most recent status: {[file["status"] for file in files]}")
        print(f"Response before timeout: {files}")
        response = await test_client_admin.get(
            f"/dataset/{dataset_id}/ingestion_status"
        )
        print(f"One last call: {response.json()}")
        raise Exception("File Upload took longer than 12 mins, aborting...")

    yield ingest_dataset


@pytest_asyncio.fixture(scope="module")
async def ingest_5_datasets(dataset_id_generator, ingest_dataset_by_id):
    dataset_ids = []
    for _ in range(5):
        id = await dataset_id_generator()
        dataset_ids.append(id)
        await ingest_dataset_by_id(id)
    assert len(dataset_ids) == 5
    yield dataset_ids


@pytest.fixture(scope="function")
def create_chatapp_request_one_dataset(dataset_id):
    unique_name = f"TestChatApp_{uuid.uuid4().hex[:8]}"
    chatapp = IChatAppCreate(
        chat_app_type=IChatAppTypeEnum.unstructured_chat_app,
        name=unique_name,
        description="chatapp for testing",
        generation_config=ChatAppGenerationConfigBase(
            llm_model="GPT4o", max_chunks_retrieved=5, max_tokens_to_sample=1024
        ),
        datasets=[dataset_id],
        voice_enabled=False,
        graph_enabled=False,
    )
    yield json.loads(chatapp.model_dump_json())


@pytest_asyncio.fixture(scope="function")
async def chatapp_id_one_dataset(
    test_client_admin: AsyncClient, workspace_id, create_chatapp_request_one_dataset
):
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/chat_app",
        json=create_chatapp_request_one_dataset,
    )
    response.raise_for_status()
    yield IChatAppRead.model_validate(response.json()["data"]).id


@pytest_asyncio.fixture(scope="function")
async def chatsession_id_one_dataset(
    test_client_admin: AsyncClient, chatapp_id_one_dataset
):
    chatapp_id = chatapp_id_one_dataset
    response = await test_client_admin.post(
        f"/chat_app/{chatapp_id}/chat_session",
        json={"title": "TestChat"},
    )
    response.raise_for_status()
    yield IChatSessionRead.model_validate(response.json()["data"]).id


@pytest_asyncio.fixture(scope="module")
async def tool_id(test_client_admin: AsyncClient):
    import time

    unique_suffix = str(int(time.time() * 1000))  # Use timestamp for better uniqueness
    tool_data = {
        "name": f"Test Tool {unique_suffix}",
        "description": "A tool for testing",
        "tool_kind": "system",
        "dataset_required": False,
        "system_tool": {
            "python_module": "builtins",
            "function_name": "len",
            "is_async": False,
        },
        "deprecated": False,
    }
    response = await test_client_admin.post("/tool", json=tool_data)
    response.raise_for_status()
    tool_id = response.json()["data"]["id"]

    yield tool_id


@pytest_asyncio.fixture(scope="module")
async def workspace_tool_id(test_client_admin: AsyncClient, workspace_id, tool_id):
    """
    Fixture to create a workspace tool and return its ID.
    """
    request_data = {
        "name": "Test Workspace Tool for Fixture",
        "tool_ids": [tool_id],
        "dataset_ids": [],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/tool", json=request_data
    )
    response.raise_for_status()
    yield response.json()["data"]["id"]


@pytest_asyncio.fixture(scope="module")
async def agent_id(test_client_admin: AsyncClient, workspace_id, workspace_tool_id):
    """
    Fixture to create an agent and return its ID.
    """
    import time

    unique_suffix = f"{random.randint(100, 999)}-{int(time.time())}"
    agent_data = {
        "name": f"Test Agent {unique_suffix}",
        "description": "Agent for testing",
        "prompt_instructions": "Follow these instructions",
        "llm_provider": "OpenAI",
        "llm_model": "GPT4o",
        "temperature": 0.7,
        "system_prompt": "System prompt example",
        "memory_enabled": True,
        "agent_metadata": {"key": "value"},
        "workspace_tool_ids": [workspace_tool_id],
    }
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/agent", json=agent_data
    )
    response.raise_for_status()
    yield response.json()["data"]["id"]


# Auto-mock GPU environment for video tests only
@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Automatically mock GPU environment for video tests.

    This ensures video tests run consistently on CPU-only systems while
    testing the logic that would normally use GPU resources.
    """
    # Create a mock torch module
    mock_torch = Mock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "Mocked GPU"

    # Only patch torch, not the entire sys.modules
    with patch.dict("sys.modules", {"torch": mock_torch}):
        yield mock_torch


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
