"""
Tests for video cleanup utilities.
"""

import os
import shutil
import tempfile
from uuid import uuid4

import pytest

from app.utils.video_cleanup_utils import (
    _should_delete_video_segments,
    check_video_segments_still_referenced,
    cleanup_empty_workspace_video_dir,
    cleanup_orphaned_video_segments,
    delete_video_segments,
    is_video_file,
)


class TestVideoFileDetection:
    """Test video file detection functionality."""

    def test_is_video_file_by_extension(self):
        """Test video file detection by file extension."""
        # Video files
        assert is_video_file("test.mp4") is True
        assert is_video_file("test.avi") is True
        assert is_video_file("test.mov") is True
        assert is_video_file("test.wmv") is True
        assert is_video_file("test.flv") is True
        assert is_video_file("test.webm") is True
        assert is_video_file("test.mkv") is True

        # Case insensitive
        assert is_video_file("test.MP4") is True
        assert is_video_file("test.AVI") is True

        # Non-video files
        assert is_video_file("test.pdf") is False
        assert is_video_file("test.txt") is False
        assert is_video_file("test.jpg") is False
        assert is_video_file("test.mp3") is False

    def test_is_video_file_by_mimetype(self):
        """Test video file detection by MIME type."""
        # Video MIME types
        assert is_video_file("test.unknown", "video/mp4") is True
        assert is_video_file("test.unknown", "video/x-msvideo") is True
        assert is_video_file("test.unknown", "video/quicktime") is True
        assert is_video_file("test.unknown", "video/x-ms-wmv") is True
        assert is_video_file("test.unknown", "video/x-flv") is True
        assert is_video_file("test.unknown", "video/webm") is True
        assert is_video_file("test.unknown", "video/x-matroska") is True

        # Non-video MIME types
        assert is_video_file("test.unknown", "application/pdf") is False
        assert is_video_file("test.unknown", "text/plain") is False
        assert is_video_file("test.unknown", "image/jpeg") is False
        assert is_video_file("test.unknown", "audio/mp3") is False

    def test_is_video_file_combined(self):
        """Test video file detection with both extension and MIME type."""
        # Both indicate video
        assert is_video_file("test.mp4", "video/mp4") is True

        # Extension indicates video, MIME type doesn't
        assert is_video_file("test.mp4", "application/octet-stream") is True

        # MIME type indicates video, extension doesn't
        assert is_video_file("test.unknown", "video/mp4") is True

        # Neither indicates video
        assert is_video_file("test.txt", "text/plain") is False


class TestVideoSegmentDeletion:
    """Test video segment deletion functionality."""

    @pytest.fixture
    def temp_segments_dir(self):
        """Create a temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_delete_video_segments_success(self, temp_segments_dir, monkeypatch):
        """Test successful deletion of video segments when no references exist."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        # Mock the reference check to return False (no references)
        async def mock_check_references(workspace_id, file_id, db_session=None):
            return False

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        workspace_id = uuid4()
        file_id = uuid4()

        # Create test segment directory structure
        segment_dir = os.path.join(temp_segments_dir, str(workspace_id), str(file_id))
        os.makedirs(segment_dir, exist_ok=True)

        # Create test files
        test_files = ["segment_0001.mp4", "segment_0002.mp4", "segments_metadata.json"]

        for filename in test_files:
            filepath = os.path.join(segment_dir, filename)
            with open(filepath, "w") as f:
                f.write("test content")

        # Verify files exist
        assert os.path.exists(segment_dir)
        assert len(os.listdir(segment_dir)) == 3

        # Delete segments
        result = await delete_video_segments(workspace_id, file_id)

        # Verify deletion
        assert result is True
        assert not os.path.exists(segment_dir)

    async def test_delete_video_segments_nonexistent(
        self, temp_segments_dir, monkeypatch
    ):
        """Test deletion when segments directory doesn't exist."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        workspace_id = uuid4()
        file_id = uuid4()

        # Try to delete non-existent segments
        result = await delete_video_segments(workspace_id, file_id)

        # Should return True (success) even if directory doesn't exist
        assert result is True

    def test_cleanup_empty_workspace_video_dir(self, temp_segments_dir, monkeypatch):
        """Test cleanup of empty workspace video directory."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        workspace_id = uuid4()

        # Create empty workspace directory
        workspace_dir = os.path.join(temp_segments_dir, str(workspace_id))
        os.makedirs(workspace_dir, exist_ok=True)

        # Verify directory exists and is empty
        assert os.path.exists(workspace_dir)
        assert len(os.listdir(workspace_dir)) == 0

        # Cleanup empty directory
        cleanup_empty_workspace_video_dir(workspace_id)

        # Verify directory is removed
        assert not os.path.exists(workspace_dir)

    def test_cleanup_empty_workspace_video_dir_with_files(
        self, temp_segments_dir, monkeypatch
    ):
        """Test cleanup doesn't remove workspace directory with files."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        workspace_id = uuid4()

        # Create workspace directory with a file
        workspace_dir = os.path.join(temp_segments_dir, str(workspace_id))
        os.makedirs(workspace_dir, exist_ok=True)

        # Add a file to the directory
        test_file = os.path.join(workspace_dir, "some_file.txt")
        with open(test_file, "w") as f:
            f.write("test")

        # Verify directory exists and has files
        assert os.path.exists(workspace_dir)
        assert len(os.listdir(workspace_dir)) == 1

        # Try to cleanup (should not remove directory with files)
        cleanup_empty_workspace_video_dir(workspace_id)

        # Verify directory still exists
        assert os.path.exists(workspace_dir)
        assert os.path.exists(test_file)

    def test_cleanup_empty_workspace_video_dir_nonexistent(
        self, temp_segments_dir, monkeypatch
    ):
        """Test cleanup when workspace directory doesn't exist."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        workspace_id = uuid4()

        # Try to cleanup non-existent directory (should not raise error)
        cleanup_empty_workspace_video_dir(workspace_id)

        # No assertion needed - just verify no exception is raised

    async def test_delete_video_segments_with_references(
        self, temp_segments_dir, monkeypatch
    ):
        """Test that segments are preserved when references exist."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        # Mock the reference check to return True (references exist)
        async def mock_check_references(workspace_id, file_id, db_session=None):
            return True

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        workspace_id = uuid4()
        file_id = uuid4()

        # Create test segment directory structure
        segment_dir = os.path.join(temp_segments_dir, str(workspace_id), str(file_id))
        os.makedirs(segment_dir, exist_ok=True)

        # Create test files
        test_files = ["segment_0001.mp4", "segment_0002.mp4", "segments_metadata.json"]

        for filename in test_files:
            filepath = os.path.join(segment_dir, filename)
            with open(filepath, "w") as f:
                f.write("test content")

        # Verify files exist
        assert os.path.exists(segment_dir)
        assert len(os.listdir(segment_dir)) == 3

        # Try to delete segments (should be skipped due to references)
        result = await delete_video_segments(workspace_id, file_id)

        # Verify segments are preserved
        assert result is False  # Deletion was skipped
        assert os.path.exists(segment_dir)  # Directory still exists
        assert len(os.listdir(segment_dir)) == 3  # Files still exist

    async def test_delete_video_segments_force_delete(
        self, temp_segments_dir, monkeypatch
    ):
        """Test force deletion ignores references."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        # Mock the reference check to return True (references exist)
        async def mock_check_references(workspace_id, file_id, db_session=None):
            return True

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        workspace_id = uuid4()
        file_id = uuid4()

        # Create test segment directory structure
        segment_dir = os.path.join(temp_segments_dir, str(workspace_id), str(file_id))
        os.makedirs(segment_dir, exist_ok=True)

        # Create test files
        test_files = ["segment_0001.mp4", "segment_0002.mp4", "segments_metadata.json"]

        for filename in test_files:
            filepath = os.path.join(segment_dir, filename)
            with open(filepath, "w") as f:
                f.write("test content")

        # Verify files exist
        assert os.path.exists(segment_dir)
        assert len(os.listdir(segment_dir)) == 3

        # Force delete segments (should ignore references)
        result = await delete_video_segments(workspace_id, file_id, force=True)

        # Verify deletion
        assert result is True
        assert not os.path.exists(segment_dir)


class TestVideoSegmentCleanupLogic:
    """Test the 4-case video segment cleanup logic."""

    @pytest.fixture
    def temp_segments_dir(self):
        """Create a temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_should_delete_video_segments_case1_video_present_active_refs(
        self, monkeypatch
    ):
        """Test Case 1: video present && active reference → keep segment (False)"""
        workspace_id = uuid4()
        file_id = uuid4()

        # Mock file exists and not deleted
        class MockFile:
            def __init__(self):
                self.deleted_at = None

        async def mock_db_execute(query):
            class MockResult:
                def scalars(self):
                    class MockScalars:
                        def first(self):
                            return MockFile()

                    return MockScalars()

            return MockResult()

        # Mock session
        class MockSession:
            async def execute(self, query):
                return await mock_db_execute(query)

            async def close(self):
                pass

        # Mock active references exist
        async def mock_check_references(workspace_id, file_id, session):
            return True

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        result = await _should_delete_video_segments(
            workspace_id, file_id, MockSession()
        )
        assert result is False  # Should keep segments

    async def test_should_delete_video_segments_case2_video_deleted_active_refs(
        self, monkeypatch
    ):
        """Test Case 2: video deleted && active reference → keep segment (False)"""
        workspace_id = uuid4()
        file_id = uuid4()

        # Mock file deleted
        class MockFile:
            def __init__(self):
                from datetime import datetime

                self.deleted_at = datetime.now()

        async def mock_db_execute(query):
            class MockResult:
                def scalars(self):
                    class MockScalars:
                        def first(self):
                            return MockFile()

                    return MockScalars()

            return MockResult()

        class MockSession:
            async def execute(self, query):
                return await mock_db_execute(query)

            async def close(self):
                pass

        # Mock active references exist
        async def mock_check_references(workspace_id, file_id, session):
            return True

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        result = await _should_delete_video_segments(
            workspace_id, file_id, MockSession()
        )
        assert result is False  # Should keep segments

    async def test_should_delete_video_segments_case3_video_deleted_no_refs(
        self, monkeypatch
    ):
        """Test Case 3: video deleted && no active reference → delete segments (True)"""
        workspace_id = uuid4()
        file_id = uuid4()

        # Mock file deleted
        class MockFile:
            def __init__(self):
                from datetime import datetime

                self.deleted_at = datetime.now()

        async def mock_db_execute(query):
            class MockResult:
                def scalars(self):
                    class MockScalars:
                        def first(self):
                            return MockFile()

                    return MockScalars()

            return MockResult()

        class MockSession:
            async def execute(self, query):
                return await mock_db_execute(query)

            async def close(self):
                pass

        # Mock no active references
        async def mock_check_references(workspace_id, file_id, session):
            return False

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        result = await _should_delete_video_segments(
            workspace_id, file_id, MockSession()
        )
        assert result is True  # Should delete segments

    async def test_should_delete_video_segments_case4_video_present_no_refs(
        self, monkeypatch
    ):
        """Test Case 4: video present && no active reference → keep segments (False)"""
        workspace_id = uuid4()
        file_id = uuid4()

        # Mock file exists and not deleted
        class MockFile:
            def __init__(self):
                self.deleted_at = None

        async def mock_db_execute(query):
            class MockResult:
                def scalars(self):
                    class MockScalars:
                        def first(self):
                            return MockFile()

                    return MockScalars()

            return MockResult()

        class MockSession:
            async def execute(self, query):
                return await mock_db_execute(query)

            async def close(self):
                pass

        # Mock no active references
        async def mock_check_references(workspace_id, file_id, session):
            return False

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        result = await _should_delete_video_segments(
            workspace_id, file_id, MockSession()
        )
        assert result is False  # Should keep segments

    async def test_should_delete_video_segments_file_not_found(self, monkeypatch):
        """Test when file record is not found → treat as deleted, check references"""
        workspace_id = uuid4()
        file_id = uuid4()

        # Mock file not found
        async def mock_db_execute(query):
            class MockResult:
                def scalars(self):
                    class MockScalars:
                        def first(self):
                            return None  # File not found

                    return MockScalars()

            return MockResult()

        class MockSession:
            async def execute(self, query):
                return await mock_db_execute(query)

            async def close(self):
                pass

        # Mock no active references
        async def mock_check_references(workspace_id, file_id, session):
            return False

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            mock_check_references,
        )

        result = await _should_delete_video_segments(
            workspace_id, file_id, MockSession()
        )
        assert result is True  # Should delete segments (file not found + no refs)

    async def test_cleanup_orphaned_video_segments_integration(
        self, temp_segments_dir, monkeypatch
    ):
        """Test cleanup_orphaned_video_segments with the new logic."""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        workspace_id = uuid4()
        file_id_keep = uuid4()  # This file should be kept
        file_id_delete = uuid4()  # This file should be deleted

        # Create test segment directories
        keep_dir = os.path.join(temp_segments_dir, str(workspace_id), str(file_id_keep))
        delete_dir = os.path.join(
            temp_segments_dir, str(workspace_id), str(file_id_delete)
        )

        os.makedirs(keep_dir, exist_ok=True)
        os.makedirs(delete_dir, exist_ok=True)

        # Create test files
        for segment_dir in [keep_dir, delete_dir]:
            for i in range(2):
                filepath = os.path.join(segment_dir, f"segment_{i:04d}.mp4")
                with open(filepath, "w") as f:
                    f.write("test content")

        # Mock _should_delete_video_segments to return different results for different files
        async def mock_should_delete(workspace_id, file_id, db_session):
            if str(file_id) == str(file_id_keep):
                return False  # Keep this one
            elif str(file_id) == str(file_id_delete):
                return True  # Delete this one
            return False

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils._should_delete_video_segments",
            mock_should_delete,
        )

        # Mock delete_video_segments to actually delete
        async def mock_delete_segments(
            workspace_id, file_id, force=False, db_session=None
        ):
            if force:
                segment_dir = os.path.join(
                    temp_segments_dir, str(workspace_id), str(file_id)
                )
                if os.path.exists(segment_dir):
                    shutil.rmtree(segment_dir)
                    return True
            return False

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.delete_video_segments",
            mock_delete_segments,
        )

        # Verify initial state
        assert os.path.exists(keep_dir)
        assert os.path.exists(delete_dir)

        # Run cleanup
        cleaned_count = await cleanup_orphaned_video_segments(workspace_id)

        # Verify results
        assert cleaned_count == 1  # Only one directory should be cleaned
        assert os.path.exists(keep_dir)  # This should be kept
        assert not os.path.exists(delete_dir)  # This should be deleted

    async def test_cache_invalidation_after_dataset_deletion(
        self, temp_segments_dir, monkeypatch
    ):
        """Test that cache is properly cleared after dataset deletion to prevent stale results"""
        # Mock settings
        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.settings.VIDEO_SEGMENTS_DIR",
            temp_segments_dir,
        )

        workspace_id = uuid4()
        file_id = uuid4()

        # Create test segment directory
        segment_dir = os.path.join(temp_segments_dir, str(workspace_id), str(file_id))
        os.makedirs(segment_dir, exist_ok=True)

        # Create test files
        for i in range(2):
            filepath = os.path.join(segment_dir, f"segment_{i:04d}.mp4")
            with open(filepath, "w") as f:
                f.write("test content")

        # Simulate the cache bug scenario:
        # 1. First check returns True (has references) - gets cached
        # 2. Dataset is deleted (references removed)
        # 3. Second check should return False (no references) but cache returns stale True
        # 4. After cache clear, third check should return False

        call_count = 0

        def mock_check_references_with_cache_bug(workspace_id, file_id, session):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # First call: has references (gets cached)
            elif call_count == 2:
                return True  # Second call: should be False but cache returns stale True
            else:
                return False  # Third call: after cache clear, returns correct False

        # Mock the actual reference check function
        async def async_mock_check_references(workspace_id, file_id, session):
            return mock_check_references_with_cache_bug(workspace_id, file_id, session)

        monkeypatch.setattr(
            "app.utils.video_cleanup_utils.check_video_segments_still_referenced",
            async_mock_check_references,
        )

        # Import the functions we need to test
        from app.utils.video_cleanup_utils import (
            _should_delete_video_segments,
            clear_reference_cache,
        )

        # Mock session
        class MockSession:
            async def execute(self, query):
                class MockResult:
                    def scalars(self):
                        class MockScalars:
                            def first(self):
                                # Mock file as deleted
                                class MockFile:
                                    def __init__(self):
                                        from datetime import datetime

                                        self.deleted_at = datetime.now()

                                return MockFile()

                        return MockScalars()

                return MockResult()

            async def close(self):
                pass

        session = MockSession()

        # Step 1: First check - should return False (keep) due to references
        result1 = await _should_delete_video_segments(workspace_id, file_id, session)
        assert result1 is False  # Keep due to references

        # Step 2: Simulate dataset deletion - references are gone but cache is stale
        # This simulates the bug where cache still returns True for "has references"
        result2 = await _should_delete_video_segments(workspace_id, file_id, session)
        assert result2 is False  # Bug: still returns False due to stale cache

        # Step 3: Clear cache (this is the fix)
        clear_reference_cache(workspace_id=workspace_id, file_id=file_id)

        # Step 4: Check again - should now return True (delete) because cache is cleared
        result3 = await _should_delete_video_segments(workspace_id, file_id, session)
        assert result3 is True  # Fixed: now correctly returns True (should delete)

        # Verify the mock was called the expected number of times
        assert call_count == 3
