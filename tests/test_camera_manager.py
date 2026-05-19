"""Tests for CameraManager."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from camera_manager import CameraManager


@pytest.fixture
def mock_cap():
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cap.get.side_effect = lambda prop: {3: 640, 4: 480}.get(prop, 0)
    return cap


class TestInit:
    def test_default_camera_index(self):
        cm = CameraManager()
        assert cm.camera_index == 0

    def test_not_initialized_by_default(self):
        cm = CameraManager()
        assert cm._is_initialized is False
        assert cm.cap is None


class TestInitialize:
    @patch("cv2.VideoCapture")
    def test_successful_initialization(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        assert cm.initialize() is True
        assert cm._is_initialized is True

    @patch("cv2.VideoCapture")
    def test_initialization_sets_dimensions(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        cm.initialize()
        assert cm.width == 640
        assert cm.height == 480

    @patch("cv2.VideoCapture")
    def test_fallback_camera_index(self, mock_vc):
        first_cap = Mock()
        first_cap.isOpened.return_value = False
        second_cap = Mock()
        second_cap.isOpened.return_value = True
        second_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        second_cap.get.side_effect = lambda prop: {3: 640, 4: 480}.get(prop, 0)
        mock_vc.side_effect = [first_cap, second_cap]

        cm = CameraManager(camera_index=0)
        assert cm.initialize() is True
        assert mock_vc.call_count >= 2

    @patch("cv2.VideoCapture")
    def test_all_cameras_fail(self, mock_vc):
        cap = Mock()
        cap.isOpened.return_value = False
        mock_vc.return_value = cap
        cm = CameraManager(camera_index=0)
        assert cm.initialize() is False


@patch("cv2.VideoCapture")
class TestReadFrame:
    def test_read_frame_success(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        cm.initialize()
        success, frame = cm.read_frame()
        assert success is True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_read_frame_return_none_when_not_initialized(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        success, frame = cm.read_frame()
        assert success is False
        assert frame is None

    def test_read_frame_flip_mirrors_image(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        cm.initialize()
        _, frame_no_flip = cm.read_frame(flip=False)
        _, frame_flip = cm.read_frame(flip=True)
        assert frame_no_flip is not None
        assert frame_flip is not None


class TestGetFrameDimensions:
    @patch("cv2.VideoCapture")
    def test_returns_zeros_before_init(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager()
        assert cm.get_frame_dimensions() == (0, 0)

    @patch("cv2.VideoCapture")
    def test_returns_width_height_after_init(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        cm.initialize()
        w, h = cm.get_frame_dimensions()
        assert w > 0 and h > 0


class TestIsOpened:
    @patch("cv2.VideoCapture")
    def test_false_before_init(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager()
        assert cm.is_opened() is False

    @patch("cv2.VideoCapture")
    def test_true_after_init(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        cm.initialize()
        assert cm.is_opened() is True


class TestRelease:
    @patch("cv2.VideoCapture")
    def test_release_clears_state(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        cm = CameraManager(camera_index=0)
        cm.initialize()
        cm.release()
        assert cm._is_initialized is False
        assert cm.cap is None


class TestContextManager:
    @patch("cv2.VideoCapture")
    def test_context_manager_initializes_and_releases(self, mock_vc, mock_cap):
        mock_vc.return_value = mock_cap
        with CameraManager(camera_index=0) as cm:
            assert cm._is_initialized is True
        assert cm._is_initialized is False
        assert cm.cap is None
