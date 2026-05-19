"""Shared fixtures for SmartBoard tests."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from canvas_manager import CanvasManager
from file_manager import FileManager


@pytest.fixture
def canvas() -> Generator[CanvasManager, None, None]:
    cm = CanvasManager(640, 480)
    yield cm


@pytest.fixture
def blank_canvas() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def file_manager(temp_dir: Path) -> Generator[FileManager, None, None]:
    fm = FileManager(base_dir=str(temp_dir))
    fm.last_save_time = 0  # Ensure first auto-save triggers
    yield fm


@pytest.fixture
def sample_landmarks():
    class FakeLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    return [FakeLandmark(0.5, 0.5) for _ in range(21)]
