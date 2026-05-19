"""Tests for FileManager."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from file_manager import FileManager


class TestInit:
    def test_creates_auto_save_dir(self, temp_dir: Path):
        fm = FileManager(base_dir=str(temp_dir))
        assert fm.auto_save_dir.exists()

    def test_auto_save_dir_is_correct_path(self, temp_dir: Path):
        fm = FileManager(base_dir=str(temp_dir))
        assert fm.auto_save_dir == temp_dir / "auto_saves"


class TestSaveAndLoad:
    def test_save_canvas_uses_timestamp_when_no_filename(self, file_manager: FileManager, blank_canvas):
        result = file_manager.save_canvas(blank_canvas)
        assert result is not None
        path = Path(result)
        assert path.suffix == ".jpg"
        assert "finger_drawing_" in path.name

    def test_save_canvas_returns_path_on_success(self, file_manager: FileManager, blank_canvas):
        result = file_manager.save_canvas(blank_canvas)
        assert result is not None
        assert Path(result).exists()

    def test_save_canvas_returns_none_when_canvas_is_none(self, file_manager: FileManager):
        assert file_manager.save_canvas(None) is None

    def test_load_canvas_returns_array(self, file_manager: FileManager, blank_canvas):
        saved = file_manager.save_canvas(blank_canvas)
        assert saved is not None
        loaded = file_manager.load_canvas(saved)
        assert loaded is not None
        assert loaded.shape == blank_canvas.shape

    def test_load_canvas_returns_none_for_missing_file(self, file_manager: FileManager):
        assert file_manager.load_canvas("nonexistent.jpg") is None


class TestAutoSave:
    def test_auto_save_returns_none_before_interval_elapses(self, file_manager: FileManager, blank_canvas):
        result = file_manager.auto_save_canvas(blank_canvas)
        assert result is not None
        result2 = file_manager.auto_save_canvas(blank_canvas)
        assert result2 is None

    def test_auto_save_returns_none_for_none_canvas(self, file_manager: FileManager):
        assert file_manager.auto_save_canvas(None) is None

    def test_auto_save_creates_file(self, file_manager: FileManager, blank_canvas):
        result = file_manager.auto_save_canvas(blank_canvas)
        assert result is not None
        assert Path(result).exists()

    def test_auto_save_filename_format(self, file_manager: FileManager, blank_canvas):
        result = file_manager.auto_save_canvas(blank_canvas)
        assert result is not None
        path = Path(result)
        assert "auto_save_" in path.name
        assert path.suffix == ".jpg"


class TestAutoSaveCleanup:
    def test_cleanup_removes_old_files(self, temp_dir: Path, blank_canvas):
        fm = FileManager(base_dir=str(temp_dir))
        fm.last_save_time = 0
        fm.max_auto_saves = 2
        with patch("time.time", return_value=100), \
             patch("time.strftime", return_value="20200101_000001"):
            fm.last_save_time = 0
            fm.auto_save_canvas(blank_canvas)
        with patch("time.time", return_value=200), \
             patch("time.strftime", return_value="20200101_000002"):
            fm.last_save_time = 0
            fm.auto_save_canvas(blank_canvas)
        with patch("time.time", return_value=300), \
             patch("time.strftime", return_value="20200101_000003"):
            fm.last_save_time = 0
            fm.auto_save_canvas(blank_canvas)
        # Only the 2 most recent should remain
        files = sorted(fm.auto_save_dir.glob("*.jpg"))
        assert len(files) == fm.max_auto_saves

    def test_get_auto_save_list_returns_sorted(self, temp_dir: Path, blank_canvas):
        fm = FileManager(base_dir=str(temp_dir))
        fm.last_save_time = 0
        with patch("time.time", return_value=100), \
             patch("time.strftime", return_value="20200101_000001"):
            fm.last_save_time = 0
            fm.auto_save_canvas(blank_canvas)
        with patch("time.time", return_value=200), \
             patch("time.strftime", return_value="20200101_000002"):
            fm.last_save_time = 0
            fm.auto_save_canvas(blank_canvas)
        saves = fm.get_auto_save_list()
        assert len(saves) == 2


class TestCustomFilename:
    def test_save_with_custom_filename_returns_path(self, file_manager: FileManager, blank_canvas):
        result = file_manager.save_canvas(blank_canvas, filename="mydrawing.jpg")
        assert result is not None
        path = Path(result)
        assert path.exists()
        assert path.name == "mydrawing.jpg"

    def test_save_with_custom_filename_stores_in_base_dir(self, file_manager: FileManager, blank_canvas):
        result = file_manager.save_canvas(blank_canvas, filename="mydrawing.jpg")
        assert result is not None
        path = Path(result)
        assert path.parent == file_manager.base_dir
