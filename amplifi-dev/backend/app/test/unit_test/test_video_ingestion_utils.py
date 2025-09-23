"""
Unit tests for video ingestion utility functions.
Tests video splitting, transcription, and captioning utilities.

NOTE: These tests do NOT perform actual video ingestion/processing.
All video processing dependencies (MoviePy, PyTorch, Whisper models, etc.) are mocked
because GitHub Actions pytest workflow cannot run GPU-based video processing.
The tests focus on verifying business logic, API contracts, and error handling.
"""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID

import numpy as np
import pytest
from PIL import Image

from app.utils.video_ingestion.utils import (
    merge_segment_information,
    segment_caption,
    speech_to_text,
    split_video,
)


class TestVideoSplitUtils:
    """Test video splitting and segmentation utilities"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_name = "test_video"
        self.working_dir = self.temp_dir

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_video_clip(self, duration=60):
        """Create a mock VideoFileClip for testing"""
        mock_video = Mock()
        mock_video.duration = duration
        mock_video.fps = 30
        mock_video.audio = Mock()
        mock_video.audio.duration = duration

        # Mock subclip to return a proper mock with duration
        def mock_subclip(start, end):
            subclip_mock = Mock()
            subclip_mock.duration = end - start
            subclip_mock.audio = Mock()  # Add audio attribute for subvideo
            return subclip_mock

        mock_video.subclip = Mock(side_effect=mock_subclip)
        mock_video.audio.subclip = Mock()

        return mock_video

    @patch("app.utils.video_ingestion.utils.video_split.VideoFileClip")
    def test_split_video_basic_functionality(self, mock_video_clip_class):
        """Test basic video splitting functionality"""
        # Setup mock
        mock_video = self.create_mock_video_clip(duration=90)  # 90 second video
        mock_video_clip_class.return_value.__enter__.return_value = mock_video

        # The create_mock_video_clip already sets up subclip properly
        # We just need to track the write_audiofile calls
        write_audiofile_mock = Mock()

        # Override the side_effect to use our tracked mock
        def mock_subclip_with_tracking(start, end):
            subclip_mock = Mock()
            subclip_mock.duration = end - start
            subclip_mock.audio = Mock()
            subclip_mock.audio.write_audiofile = write_audiofile_mock
            return subclip_mock

        mock_video.subclip.side_effect = mock_subclip_with_tracking

        # Test parameters
        segment_length = 30
        num_frames_per_segment = 5
        audio_format = "mp3"

        # Run the function
        segment_index2name, segment_times_info = split_video(
            video_path=f"{self.temp_dir}/test_video.mp4",
            working_dir=self.working_dir,
            segment_length=segment_length,
            num_frames_per_segment=num_frames_per_segment,
            audio_output_format=audio_format,
        )

        # Verify results
        assert len(segment_index2name) == 3  # 90 seconds / 30 seconds = 3 segments
        assert len(segment_times_info) == 3

        # Check segment naming
        for i in range(3):
            assert str(i) in segment_index2name
            # Segment names contain timestamp and index info

        # Check segment timing info
        for i in range(3):
            assert str(i) in segment_times_info
            assert "timestamp" in segment_times_info[str(i)]
            assert "frame_times" in segment_times_info[str(i)]

        # Verify audio extraction was called (subvideo.audio, not video.audio.subclip)
        assert mock_video.subclip.call_count == 3
        assert write_audiofile_mock.call_count == 3

    @patch("app.utils.video_ingestion.utils.video_split.VideoFileClip")
    def test_split_video_edge_cases(self, mock_video_clip_class):
        """Test video splitting with edge cases"""
        # Test very short video
        mock_video = self.create_mock_video_clip(duration=15)  # 15 second video
        mock_video_clip_class.return_value.__enter__.return_value = mock_video

        mock_audio_segment = Mock()
        mock_video.audio.subclip.return_value = mock_audio_segment
        mock_audio_segment.write_audiofile = Mock()

        segment_index2name, segment_times_info = split_video(
            video_path=f"{self.temp_dir}/short_video.mp4",
            working_dir=self.working_dir,
            segment_length=30,
            num_frames_per_segment=5,
            audio_output_format="mp3",
        )

        # Should create 1 segment for short video
        assert len(segment_index2name) == 1
        assert len(segment_times_info) == 1


class TestTranscriptionUtils:
    """Test speech-to-text transcription utilities"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_name = "test_video"
        self.working_dir = self.temp_dir

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_audio_files(self):
        """Create test audio files for transcription"""
        cache_path = os.path.join(self.working_dir, "_cache", self.video_name)
        os.makedirs(cache_path, exist_ok=True)

        # Create dummy audio files
        for i in range(3):
            audio_file = os.path.join(cache_path, f"test_video_segment_{i}.mp3")
            Path(audio_file).touch()

        return cache_path

    @patch("builtins.__import__")
    def test_speech_to_text_no_cuda(self, mock_import):
        """Test transcription when CUDA is not available"""

        def side_effect(name, *args, **kwargs):
            if name == "torch":
                mock_torch = Mock()
                mock_torch.cuda.is_available.return_value = False
                return mock_torch
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        segment_index2name = {"0": "segment_0", "1": "segment_1", "2": "segment_2"}

        result = speech_to_text(
            video_name=self.video_name,
            working_dir=self.working_dir,
            segment_index2name=segment_index2name,
            audio_output_format="mp3",
            whisper_model=None,
        )

        # Should return empty transcripts when CUDA not available
        assert len(result) == 3
        assert all(transcript == "" for transcript in result.values())

    @patch("builtins.__import__")
    def test_speech_to_text_no_model(self, mock_import):
        """Test transcription when no model is provided"""

        def side_effect(name, *args, **kwargs):
            if name == "torch":
                mock_torch = Mock()
                mock_torch.cuda.is_available.return_value = True
                return mock_torch
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        segment_index2name = {"0": "segment_0", "1": "segment_1", "2": "segment_2"}

        result = speech_to_text(
            video_name=self.video_name,
            working_dir=self.working_dir,
            segment_index2name=segment_index2name,
            audio_output_format="mp3",
            whisper_model=None,
        )

        # Should return empty transcripts when no model provided
        assert len(result) == 3
        assert all(transcript == "" for transcript in result.values())

    @patch("concurrent.futures.ThreadPoolExecutor")
    def test_speech_to_text_with_model(self, mock_executor):
        """Test transcription with a valid model"""
        # Create a mock torch module
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True

        # Patch the torch import in the transcription module
        with patch.dict("sys.modules", {"torch": mock_torch}):

            # Create test audio files
            self.create_test_audio_files()

            # Mock whisper model
            mock_model = Mock()
            mock_segments = [Mock(text="Hello world"), Mock(text="This is a test")]
            mock_model.transcribe.return_value = (mock_segments, None)

            # Mock ThreadPoolExecutor
            mock_future1 = Mock()
            mock_future2 = Mock()
            mock_future1.result.return_value = ("0", "Transcribed text segment 0")
            mock_future2.result.return_value = ("1", "Transcribed text segment 1")

            mock_executor_instance = Mock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            mock_executor_instance.submit.side_effect = [mock_future1, mock_future2]

            segment_index2name = {
                "0": "test_video_segment_0",
                "1": "test_video_segment_1",
            }

            # Mock concurrent.futures.as_completed
            with patch("concurrent.futures.as_completed") as mock_as_completed:
                mock_as_completed.return_value = [mock_future1, mock_future2]

                result = speech_to_text(
                    video_name=self.video_name,
                    working_dir=self.working_dir,
                    segment_index2name=segment_index2name,
                    audio_output_format="mp3",
                    whisper_model=mock_model,
                    max_workers=2,
                )

            # Should return transcripts for each segment
            assert len(result) == 2
            assert all(
                transcript
                in ["Transcribed text segment 0", "Transcribed text segment 1"]
                for transcript in result.values()
            )

    @patch("builtins.__import__")
    def test_speech_to_text_no_audio_files(self, mock_import):
        """Test transcription when no audio files exist"""

        def side_effect(name, *args, **kwargs):
            if name == "torch":
                mock_torch = Mock()
                mock_torch.cuda.is_available.return_value = True
                return mock_torch
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        mock_model = Mock()
        segment_index2name = {"0": "segment_0", "1": "segment_1"}

        result = speech_to_text(
            video_name=self.video_name,
            working_dir=self.working_dir,
            segment_index2name=segment_index2name,
            audio_output_format="mp3",
            whisper_model=mock_model,
        )

        # Should return empty transcripts when no audio files exist
        assert len(result) == 2
        assert all(transcript == "" for transcript in result.values())


class TestCaptioningUtils:
    """Test video captioning and visual analysis utilities"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.video_name = "test_video"

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encode_video_frames_basic(self):
        """Test basic video frame encoding"""
        from app.utils.video_ingestion.utils.captioning import encode_video_frames

        # Create mock video
        mock_video = Mock()

        # Create test frames
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mock_video.get_frame.return_value = test_frame

        frame_times = [0.0, 15.0, 30.0]

        result = encode_video_frames(mock_video, frame_times)

        # Should return PIL Images
        assert len(result) == 3
        assert all(isinstance(frame, Image.Image) for frame in result)
        assert mock_video.get_frame.call_count == 3

    def test_encode_video_frames_error_handling(self):
        """Test frame encoding error handling"""
        from app.utils.video_ingestion.utils.captioning import encode_video_frames

        # Create mock video that raises exception
        mock_video = Mock()
        mock_video.get_frame.side_effect = Exception("Frame extraction error")

        frame_times = [0.0, 15.0]

        # Should handle errors gracefully
        with patch("app.utils.video_ingestion.utils.captioning.logger") as mock_logger:
            result = encode_video_frames(mock_video, frame_times)

            # Should log warning and return empty list or handle gracefully
            mock_logger.warning.assert_called()

    @patch("app.utils.video_ingestion.utils.captioning.VideoFileClip")
    @patch("app.utils.video_ingestion.utils.captioning.encode_video_frames")
    def test_segment_caption_basic(self, mock_encode_frames, mock_video_clip):
        """Test basic segment captioning functionality"""
        # Setup mocks
        mock_video = Mock()
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Mock frame encoding
        mock_frames = [Image.new("RGB", (100, 100), "red") for _ in range(3)]
        mock_encode_frames.return_value = mock_frames

        # Mock caption model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model.chat.return_value = "This is a test caption"

        # Test data
        segment_index2name = {0: "segment_0", 1: "segment_1"}
        transcripts = {0: "Hello world", 1: "Test transcript"}
        segment_times_info = {
            0: {"frame_times": [0.0, 15.0, 30.0]},
            1: {"frame_times": [30.0, 45.0, 60.0]},
        }
        caption_result = {}
        error_queue = Mock()

        # Run the function
        segment_caption(
            video_name=self.video_name,
            video_path=f"{self.temp_dir}/test_video.mp4",
            segment_index2name=segment_index2name,
            transcripts=transcripts,
            segment_times_info=segment_times_info,
            caption_result=caption_result,
            error_queue=error_queue,
            caption_model=mock_model,
            caption_tokenizer=mock_tokenizer,
            batch_size=2,
        )

        # Verify captions were generated
        assert len(caption_result) == 2
        assert 0 in caption_result
        assert 1 in caption_result

    def test_merge_segment_information(self):
        """Test merging segment information"""
        # Test data (using string keys to match actual implementation)
        segment_index2name = {"0": "segment_0", "1": "segment_1"}
        transcripts = {"0": "Hello world", "1": "Test transcript"}
        captions = {"0": "Visual: red background", "1": "Visual: blue background"}
        segment_times_info = {
            "0": {"timestamp": (0, 30), "frame_times": np.array([0.0, 15.0, 30.0])},
            "1": {"timestamp": (30, 60), "frame_times": np.array([30.0, 45.0, 60.0])},
        }

        result = merge_segment_information(
            segment_index2name=segment_index2name,
            transcripts=transcripts,
            captions=captions,
            segment_times_info=segment_times_info,
        )

        # Verify merged information - result is keyed by segment names
        assert len(result) == 2
        assert "segment_0" in result
        assert "segment_1" in result

        for segment_name in ["segment_0", "segment_1"]:
            assert "transcript" in result[segment_name]
            assert "caption" in result[segment_name]
            assert "start_time" in result[segment_name]
            assert "end_time" in result[segment_name]
            assert "content" in result[segment_name]  # The actual field name

        # Check content includes both transcript and caption
        assert "Hello world" in result["segment_0"]["content"]
        assert "Visual: red background" in result["segment_0"]["content"]


class TestVideoIngestionUtilsIntegration:
    """Integration tests for video ingestion utilities"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_video_processing_pipeline_mock(self):
        """Test the complete video processing pipeline with mocks"""
        # This test simulates the full pipeline without actual video files

        # Mock video splitting
        with patch(
            "app.utils.video_ingestion.utils.video_split.split_video"
        ) as mock_split:
            mock_split.return_value = (
                {"0": "segment_0", "1": "segment_1"},
                {
                    "0": {"timestamp": (0, 30), "frame_times": np.array([0, 15, 30])},
                    "1": {"timestamp": (30, 60), "frame_times": np.array([30, 45, 60])},
                },
            )

            # Mock transcription
            with patch(
                "app.utils.video_ingestion.utils.transcription.speech_to_text"
            ) as mock_transcribe:
                mock_transcribe.return_value = {
                    "0": "First segment transcript",
                    "1": "Second segment transcript",
                }

                # Mock captioning
                with patch(
                    "app.utils.video_ingestion.utils.captioning.segment_caption"
                ) as mock_caption:

                    def mock_caption_side_effect(*args, **kwargs):
                        caption_result = kwargs["caption_result"]
                        caption_result["0"] = "First segment caption"
                        caption_result["1"] = "Second segment caption"

                    mock_caption.side_effect = mock_caption_side_effect

                    # Simulate the pipeline
                    video_path = f"{self.temp_dir}/test_video.mp4"
                    working_dir = self.temp_dir

                    # Step 1: Split video (use mock result)
                    segment_index2name, segment_times_info = mock_split.return_value

                    # Step 2: Transcribe (use mocked result)
                    transcripts = mock_transcribe.return_value

                    # Step 3: Caption (use mocked result)
                    captions = {
                        "0": "First segment caption",
                        "1": "Second segment caption",
                    }

                    # Step 4: Merge
                    merged_info = merge_segment_information(
                        segment_index2name, segment_times_info, transcripts, captions
                    )

                    # Verify complete pipeline
                    assert len(merged_info) == 2
                    assert "segment_0" in merged_info
                    assert "segment_1" in merged_info
                    assert all("content" in info for info in merged_info.values())
                    assert (
                        "First segment transcript"
                        in merged_info["segment_0"]["content"]
                    )
                    assert (
                        "First segment caption" in merged_info["segment_0"]["content"]
                    )
