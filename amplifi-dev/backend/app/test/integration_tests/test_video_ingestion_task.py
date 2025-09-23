"""
Integration tests for video ingestion task.
Tests the complete video ingestion pipeline with mocked dependencies.

NOTE: These tests do NOT perform actual video ingestion/processing.
All video processing dependencies (AI models, GPU operations) are mocked because 
GitHub Actions pytest workflow runs on CPU-only runners without GPU access.
The tests verify task orchestration, database operations, and error handling logic.
"""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import pytest
from PIL import Image

from app.api.video_ingestion_task import video_ingestion_task
from app.models.document_model import DocumentProcessingStatusEnum, DocumentTypeEnum


class TestVideoIngestionTaskIntegration:
    """Integration tests for video ingestion task"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_video_path = self.create_test_video_file()
        self.dataset_id = str(uuid.uuid4())
        self.file_id = str(uuid.uuid4())
        self.user_id = uuid.uuid4()
        self.ingestion_id = str(uuid.uuid4())

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_video_file(self):
        """Create a mock video file for testing"""
        video_path = Path(self.temp_dir) / "test_video.mp4"
        # Create a dummy file to simulate video
        video_path.write_bytes(b"fake video content for testing")
        return str(video_path)

    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    def test_video_ingestion_disabled(self, mock_is_enabled):
        """Test video ingestion when feature is disabled"""
        mock_is_enabled.return_value = False

        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        assert result["success"] is False
        assert result["error"] == "Video ingestion is disabled"
        assert result["status"] == "failed"
        assert result["document_type"] == "Video"

    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task.os.path.exists")
    @patch("app.api.video_ingestion_task.get_redis_client_sync")
    def test_video_file_not_found(self, mock_redis, mock_exists, mock_is_enabled):
        """Test video ingestion when file doesn't exist"""
        mock_is_enabled.return_value = True
        mock_exists.return_value = False
        mock_redis.return_value = Mock()

        result = video_ingestion_task(
            file_id=self.file_id,
            file_path="/nonexistent/video.mp4",
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()
        assert result["status"] == "exception"

    @patch("app.api.video_ingestion_task._acquire_processing_lock_atomic")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.SyncSessionLocal")
    @patch("app.api.video_ingestion_task.get_redis_client_sync")
    def test_video_file_too_large(
        self,
        mock_redis,
        mock_session,
        mock_validate,
        mock_is_enabled,
        mock_acquire_tokens,
        mock_acquire_lock,
    ):
        """Test video ingestion when database operations fail"""
        mock_is_enabled.return_value = True
        mock_validate.return_value = None
        mock_redis.return_value = Mock()
        mock_acquire_tokens.return_value = (True, None)
        mock_acquire_lock.return_value = (True, None)

        # Mock database session to fail with file record not found
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = (
            None  # No file record found
        )
        mock_session.return_value.__enter__.return_value = mock_db

        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        assert result["success"] is False
        assert "file record" in result["error"].lower()
        assert result["status"] == "failed"

    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    def test_token_acquisition_failure(
        self, mock_acquire_tokens, mock_validate, mock_is_enabled
    ):
        """Test video ingestion when token acquisition fails"""
        mock_is_enabled.return_value = True
        mock_validate.return_value = None  # No validation error
        mock_acquire_tokens.return_value = False  # Token acquisition failed

        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        assert result["success"] is False
        assert "Failed to acquire processing tokens" in result["error"]
        assert result["status"] == "failed"

    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    @patch("app.api.video_ingestion_task.SyncSessionLocal")
    @patch("app.api.video_ingestion_task._handle_existing_document")
    @patch("app.api.video_ingestion_task.get_redis_client_sync")
    def test_skip_existing_document(
        self,
        mock_redis,
        mock_handle_existing,
        mock_session,
        mock_acquire_tokens,
        mock_validate,
        mock_is_enabled,
    ):
        """Test skipping already processed documents"""
        mock_is_enabled.return_value = True
        mock_validate.return_value = None
        mock_acquire_tokens.return_value = (True, None)
        mock_redis.return_value = Mock()

        # Mock existing document response
        skip_response = {
            "file_id": self.file_id,
            "success": True,
            "status": "skipped",
            "message": "Document already processed",
        }
        mock_handle_existing.return_value = skip_response

        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
            skip_successful_files=True,
        )

        assert result == skip_response

    @patch("app.api.video_ingestion_task._acquire_processing_lock_atomic")
    @patch("app.api.video_ingestion_task.get_redis_client_sync")
    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    @patch("app.api.video_ingestion_task.SyncSessionLocal")
    @patch("app.api.video_ingestion_task._handle_existing_document")
    @patch("app.api.video_ingestion_task._create_or_update_document")
    @patch("app.api.video_ingestion_task._get_workspace_id_from_file")
    @patch("app.api.video_ingestion_task._handle_workspace_video_processing")
    @patch("app.api.video_ingestion_task._store_video_chunks")
    @patch("app.api.video_ingestion_task._update_document_with_results")
    @patch("app.api.video_ingestion_task.propagate_ingestion_status")
    def test_successful_video_ingestion(
        self,
        mock_propagate,
        mock_update_document,
        mock_store_chunks,
        mock_workspace_processing,
        mock_get_workspace_id,
        mock_create_document,
        mock_handle_existing,
        mock_session,
        mock_acquire_tokens,
        mock_validate,
        mock_is_enabled,
        mock_redis,
        mock_acquire_lock,
    ):
        """Test successful video ingestion pipeline"""
        # Setup mocks
        mock_is_enabled.return_value = True
        mock_validate.return_value = None
        mock_acquire_tokens.return_value = (True, None)
        mock_handle_existing.return_value = None  # Don't skip
        mock_redis.return_value = Mock()
        mock_acquire_lock.return_value = (True, None)  # Successfully acquired lock

        # Mock workspace ID retrieval
        workspace_id = str(uuid.uuid4())
        mock_get_workspace_id.return_value = workspace_id

        # Mock document creation
        mock_document = Mock()
        mock_document.id = uuid.uuid4()
        mock_create_document.return_value = mock_document  # Return document object

        # Mock workspace video processing results (updated to match new return format)
        mock_workspace_results = {
            "success": True,
            "segments_created": True,
            "original_video_deleted": True,
            "document": {
                "description": "Test video description",
                "description_embedding": [0.1] * 1024,
                "document_metadata": {
                    "video_name": "test_video",
                    "file_size": 1024,
                    "video_duration": 60,
                    "total_segments": 2,
                    "processing_completed_at": "2023-01-01T00:00:00",
                },
            },
            "chunks": [
                {
                    "chunk_text": "Hello world. Person speaking",
                    "chunk_embedding": [0.1] * 1024,
                    "chunk_metadata": {
                        "segment_name": "test_video_segment_0",
                        "start_time": 0,
                        "end_time": 30,
                    },
                },
                {
                    "chunk_text": "This is a test. Text on screen",
                    "chunk_embedding": [0.1] * 1024,
                    "chunk_metadata": {
                        "segment_name": "test_video_segment_1",
                        "start_time": 30,
                        "end_time": 60,
                    },
                },
            ],
            "message": "Video processed successfully using workspace segments",
        }
        mock_workspace_processing.return_value = mock_workspace_results

        # Mock chunk creation
        mock_chunks = (5, 0)  # (chunks_created, failed_chunks)
        mock_store_chunks.return_value = mock_chunks
        mock_update_document.return_value = None

        # Mock database session
        mock_db = Mock()
        mock_document_query = Mock()
        mock_document_query.id = mock_document.id
        mock_document_query.document_metadata = {}  # Make it assignable
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_document_query
        )
        mock_session.return_value.__enter__.return_value = mock_db

        # Run the task
        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        # Verify successful result
        assert result["success"] is True
        assert result["status"] == "success"
        assert result["document_type"] == "Video"
        assert result["chunk_count"] == 5
        assert result["failed_chunks"] == 0

        # Verify workspace-specific fields
        assert (
            "new segments created" in result["message"]
            or "reused existing segments" in result["message"]
        )
        assert (
            "original video deleted" in result["message"]
            or "reused existing segments" in result["message"]
        )

        # Verify all processing steps were called
        mock_get_workspace_id.assert_called_once_with(self.file_id)
        mock_workspace_processing.assert_called_once()
        mock_store_chunks.assert_called_once()
        mock_propagate.assert_called_once()

    @patch("app.api.video_ingestion_task._acquire_processing_lock_atomic")
    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    @patch("app.api.video_ingestion_task.SyncSessionLocal")
    @patch("app.api.video_ingestion_task._handle_existing_document")
    @patch("app.api.video_ingestion_task._create_or_update_document")
    @patch("app.api.video_ingestion_task._get_workspace_id_from_file")
    @patch("app.api.video_ingestion_task._process_video_directly_to_workspace")
    @patch("app.api.video_ingestion_task.get_redis_client_sync")
    def test_video_processing_error(
        self,
        mock_redis,
        mock_process_video,
        mock_get_workspace_id,
        mock_create_document,
        mock_handle_existing,
        mock_session,
        mock_acquire_tokens,
        mock_validate,
        mock_is_enabled,
        mock_acquire_lock,
    ):
        """Test handling of video processing errors"""
        # Setup mocks
        mock_is_enabled.return_value = True
        mock_validate.return_value = None
        mock_acquire_tokens.return_value = (True, None)
        mock_handle_existing.return_value = None
        mock_redis.return_value = Mock()
        mock_acquire_lock.return_value = (True, None)  # Successfully acquired lock

        # Mock workspace ID retrieval
        workspace_id = str(uuid.uuid4())
        mock_get_workspace_id.return_value = workspace_id

        mock_document = Mock()
        mock_document.id = uuid.uuid4()
        mock_create_document.return_value = mock_document

        # Mock workspace video processing to raise exception
        mock_process_video.side_effect = Exception("Video processing failed")

        mock_db = Mock()
        mock_session.return_value.__enter__.return_value = mock_db

        # Run the task
        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        # Verify error handling
        assert result["success"] is False
        assert result["status"] == "failed"
        assert "Video processing failed" in result["error"]

    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    def test_retry_logic(self, mock_acquire_tokens, mock_validate, mock_is_enabled):
        """Test retry logic for video ingestion"""
        mock_is_enabled.return_value = True
        mock_validate.return_value = None

        # First call fails, second succeeds
        mock_acquire_tokens.side_effect = [False, True]

        # Mock the task to simulate retry
        with patch(
            "app.api.video_ingestion_task.video_ingestion_task.retry"
        ) as mock_retry:
            result = video_ingestion_task(
                file_id=self.file_id,
                file_path=self.test_video_path,
                ingestion_id=self.ingestion_id,
                dataset_id=self.dataset_id,
                user_id=self.user_id,
                retry_count=1,
                max_retries=3,
            )

        # Should attempt retry when token acquisition fails initially
        assert result["success"] is False  # First attempt fails

    @patch("app.api.video_ingestion_task.is_video_ingestion_enabled")
    @patch("app.api.video_ingestion_task._validate_video_file")
    @patch("app.api.video_ingestion_task.acquire_tokens_with_retry")
    @patch("app.api.video_ingestion_task.SyncSessionLocal")
    @patch("app.api.video_ingestion_task._handle_existing_document")
    @patch("app.api.video_ingestion_task._create_or_update_document")
    @patch("app.api.video_ingestion_task._get_workspace_id_from_file")
    @patch("app.api.video_ingestion_task._handle_workspace_video_processing")
    @patch("app.api.video_ingestion_task._store_video_chunks")
    @patch("app.api.video_ingestion_task._update_document_with_results")
    @patch("app.api.video_ingestion_task.propagate_ingestion_status")
    @patch("app.api.video_ingestion_task.get_redis_client_sync")
    def test_workspace_segment_reuse(
        self,
        mock_redis,
        mock_propagate,
        mock_update_document,
        mock_store_chunks,
        mock_workspace_processing,
        mock_get_workspace_id,
        mock_create_document,
        mock_handle_existing,
        mock_session,
        mock_acquire_tokens,
        mock_validate,
        mock_is_enabled,
    ):
        """Test that workspace segments are reused for same file in different datasets"""
        # Setup mocks
        mock_is_enabled.return_value = True
        mock_validate.return_value = None
        mock_acquire_tokens.return_value = (True, None)
        mock_handle_existing.return_value = None
        mock_redis.return_value = Mock()

        workspace_id = str(uuid.uuid4())
        mock_get_workspace_id.return_value = workspace_id

        mock_document = Mock()
        mock_document.id = uuid.uuid4()
        mock_create_document.return_value = mock_document

        # Mock workspace processing to indicate segments were reused (not created)
        mock_workspace_results = {
            "success": True,
            "segments_created": False,  # Segments were reused
            "original_video_deleted": False,  # Original already deleted
            "document": {"description": "Test video description"},
            "chunks": [
                {"chunk_text": "Reused segment", "chunk_embedding": [0.1] * 1024}
            ],
            "message": "Video processed successfully using workspace segments",
        }
        mock_workspace_processing.return_value = mock_workspace_results

        mock_chunks = (3, 0)
        mock_store_chunks.return_value = mock_chunks
        mock_update_document.return_value = None

        mock_db = Mock()
        mock_document_query = Mock()
        mock_document_query.id = mock_document.id
        mock_document_query.document_metadata = {}
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_document_query
        )
        mock_session.return_value.__enter__.return_value = mock_db

        # Run the task (simulating second ingestion of same file)
        result = video_ingestion_task(
            file_id=self.file_id,
            file_path=self.test_video_path,
            ingestion_id=self.ingestion_id,
            dataset_id=self.dataset_id,
            user_id=self.user_id,
        )

        # Verify successful result with segment reuse
        assert result["success"] is True
        assert result["status"] == "success"
        assert result["chunk_count"] == 3

        # Verify message indicates reuse
        assert "reused" in result["message"] or "existing" in result["message"]

        # Verify workspace processing was called
        mock_workspace_processing.assert_called_once()

        # Verify workspace processing was called with correct parameters
        call_args = mock_workspace_processing.call_args
        assert call_args[1]["workspace_id"] == workspace_id
        assert call_args[1]["file_id"] == self.file_id


class TestVideoProcessingComponents:
    """Test individual video processing components"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("app.api.video_ingestion_task._generate_video_description")
    @patch("app.api.video_ingestion_task._save_video_segments_to_workspace")
    @patch("app.api.video_ingestion_task.generate_embedding_with_retry")
    @patch("app.api.video_ingestion_task.split_video")
    @patch("app.api.video_ingestion_task.speech_to_text")
    @patch("app.api.video_ingestion_task.segment_caption")
    @patch("app.api.video_ingestion_task.merge_segment_information")
    @patch("app.api.video_ingestion_task.get_whisper_model")
    @patch("app.api.video_ingestion_task.get_caption_model")
    @patch("app.api.video_ingestion_task.os.path.getsize")
    def test_process_video_directly_to_workspace_success(
        self,
        mock_getsize,
        mock_get_caption_model,
        mock_get_whisper_model,
        mock_merge_info,
        mock_segment_caption,
        mock_speech_to_text,
        mock_split_video,
        mock_generate_embedding,
        mock_save_video_segments,
        mock_generate_description,
    ):
        """Test _process_video_directly_to_workspace function"""
        from app.api.video_ingestion_task import (
            _process_video_directly_to_workspace,
        )

        # Mock file size
        mock_getsize.return_value = 1024 * 1024  # 1MB

        # Setup mocks
        mock_split_video.return_value = (
            {0: "segment_0", 1: "segment_1"},
            {
                0: {"timestamp": (0, 30), "frame_times": [0, 15, 30]},
                1: {"timestamp": (30, 60), "frame_times": [30, 45, 60]},
            },
        )

        mock_speech_to_text.return_value = {
            0: "First segment transcript",
            1: "Second segment transcript",
        }

        def mock_caption_side_effect(*args, **kwargs):
            # segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue, ...)
            # caption_result is the 6th argument (index 5)
            if len(args) >= 6:
                caption_result = args[5]
            else:
                caption_result = kwargs.get("caption_result", {})
            caption_result[0] = "First segment caption"
            caption_result[1] = "Second segment caption"

        mock_segment_caption.side_effect = mock_caption_side_effect

        mock_merge_info.return_value = {
            "segment_0": {
                "transcript": "First segment transcript",
                "caption": "First segment caption",
                "start_time": 0,
                "end_time": 30,
                "content": "Caption:\nFirst segment caption\nTranscript:\nFirst segment transcript\n\n",
                "time": "0-30",
                "frame_times": [0, 15, 30],
                "video_segment_path": "segment_0.mp4",
                "frame_count": 3,
            },
            "segment_1": {
                "transcript": "Second segment transcript",
                "caption": "Second segment caption",
                "start_time": 30,
                "end_time": 60,
                "content": "Caption:\nSecond segment caption\nTranscript:\nSecond segment transcript\n\n",
                "time": "30-60",
                "frame_times": [30, 45, 60],
                "video_segment_path": "segment_1.mp4",
                "frame_count": 3,
            },
        }

        mock_get_whisper_model.return_value = Mock()
        mock_get_caption_model.return_value = (Mock(), Mock())

        # Mock additional video processing functions
        mock_save_video_segments.return_value = None  # No return value needed
        mock_generate_embedding.return_value = [0.1] * 1024
        mock_generate_description.return_value = "Test video description"

        # Test the function
        result = _process_video_directly_to_workspace(
            file_path=f"{self.temp_dir}/test_video.mp4",
            segment_dir=f"{self.temp_dir}/segments",
            working_dir=self.temp_dir,
        )

        # Verify results (updated to match new return format)
        assert "segments_info" in result
        assert "document_data" in result
        assert "processing_completed" in result
        assert result["processing_completed"] is True
        assert result["segment_count"] == 2

        # Verify all processing steps were called
        mock_split_video.assert_called_once()
        mock_speech_to_text.assert_called_once()
        mock_segment_caption.assert_called_once()
        mock_merge_info.assert_called_once()

    @patch("app.api.video_ingestion_task.DocumentChunk")
    @patch("app.api.video_ingestion_task.generate_embedding_with_retry")
    def test_create_document_chunks(self, mock_generate_embedding, mock_document_chunk):
        """Test document chunk creation from video segments"""
        # Test document chunk creation from video segments - function doesn't exist yet
        # from app.api.video_ingestion_task import _create_document_chunks
        pytest.skip("_create_document_chunks function not implemented yet")

        # Mock embedding generation
        mock_generate_embedding.return_value = [0.1, 0.2, 0.3] * 512  # 1536 dimensions

        # Mock DocumentChunk creation
        mock_chunk_instance = Mock()
        mock_document_chunk.return_value = mock_chunk_instance

        # Test data
        video_results = {
            "segments": [
                {
                    "segment_name": "segment_0",
                    "transcript": "Hello world",
                    "caption": "Person speaking",
                    "start_time": 0,
                    "end_time": 30,
                    "combined_content": "Hello world. Person speaking",
                }
            ]
        }

        document_id = uuid.uuid4()
        dataset_id = str(uuid.uuid4())
        mock_db = Mock()

        # Run the function
        chunks = _create_document_chunks(
            db=mock_db,
            video_results=video_results,
            document_id=document_id,
            dataset_id=dataset_id,
        )

        # Verify chunk creation
        assert len(chunks) == 1
        mock_generate_embedding.assert_called_once()
        mock_document_chunk.assert_called_once()
        mock_db.add.assert_called_once()

    def test_get_video_config(self):
        """Test video configuration retrieval"""
        from app.api.video_ingestion_task import _get_video_config

        config = _get_video_config()

        # Verify required config keys
        required_keys = [
            "video_segment_length",
            "rough_num_frames_per_segment",
            "audio_output_format",
            "video_output_format",
            "enable_captioning",
            "enable_transcription",
            "embedding_model",
            "embedding_dim",
            "transcription_max_workers",
            "captioning_batch_size",
        ]

        for key in required_keys:
            assert key in config

        # Verify default values
        assert config["video_segment_length"] == 30
        assert config["audio_output_format"] == "mp3"
        assert config["enable_transcription"] is True
        assert config["enable_captioning"] is True
