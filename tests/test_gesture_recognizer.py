"""Tests for GestureRecognizer logic methods.

MediaPipe initialization is mocked to avoid requiring the model file.
Tests focus on the pure-logic methods: detect_gesture, get_stable_gesture,
smooth_coordinates, get_finger_positions, and draw_hand_landmarks.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Must patch BEFORE importing GestureRecognizer
with patch("gesture_recognizer._ensure_model", return_value="dummy.task"), \
     patch("gesture_recognizer.HandLandmarker.create_from_options") as mock_create:
    mock_landmarker = MagicMock()
    mock_create.return_value = mock_landmarker
    from gesture_recognizer import GestureRecognizer


class FakeLandmark:
    """Simulates MediaPipe's NormalizedLandmark with x, y, z attributes."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(finger_states: list) -> list:
    """Create 21 landmarks that simulate given finger states.

    finger_states: list of 5 bools [thumb, index, middle, ring, pinky]
                   True = extended, False = curled
    """
    ls = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    # Thumb (landmarks 2, 4)
    ls[2] = FakeLandmark(0.48, 0.5)
    if finger_states[0]:
        ls[4] = FakeLandmark(0.52, 0.5)  # extended thumb (right)
    else:
        ls[4] = FakeLandmark(0.48, 0.5)  # curled thumb

    # Other fingers: tip y < mcp y = extended
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
    for i, (tip, mcp) in enumerate(pairs):
        if finger_states[i + 1]:
            ls[tip] = FakeLandmark(0.5, 0.4)  # above (extended)
        else:
            ls[tip] = FakeLandmark(0.5, 0.6)  # below (curled)
        ls[mcp] = FakeLandmark(0.5, 0.5)
    return ls


@pytest.fixture(autouse=True)
def recognizer():
    gr = GestureRecognizer()
    gr.reset_buffers()
    return gr


class TestDetectGesture:
    def test_all_fingers_extended_is_palm_erase(self, recognizer: GestureRecognizer):
        lm = make_landmarks([True, True, True, True, True])
        assert recognizer.detect_gesture(lm) == GestureRecognizer.GESTURE_PALM_ERASE

    def test_only_index_extended_is_drawing(self, recognizer: GestureRecognizer):
        lm = make_landmarks([False, True, False, False, False])
        assert recognizer.detect_gesture(lm) == GestureRecognizer.GESTURE_DRAWING

    def test_index_and_middle_extended_is_shape_mode(self, recognizer: GestureRecognizer):
        lm = make_landmarks([False, True, True, False, False])
        assert recognizer.detect_gesture(lm) == GestureRecognizer.GESTURE_SHAPE_MODE

    def test_no_fingers_extended_is_pause(self, recognizer: GestureRecognizer):
        lm = make_landmarks([False, False, False, False, False])
        assert recognizer.detect_gesture(lm) == GestureRecognizer.GESTURE_PAUSE

    def test_three_fingers_is_none(self, recognizer: GestureRecognizer):
        lm = make_landmarks([False, True, True, True, False])
        assert recognizer.detect_gesture(lm) == GestureRecognizer.GESTURE_NONE

    def test_thumb_and_ring_is_none(self, recognizer: GestureRecognizer):
        lm = make_landmarks([True, False, False, True, False])
        assert recognizer.detect_gesture(lm) == GestureRecognizer.GESTURE_NONE


class TestGetStableGesture:
    def test_returns_current_when_buffer_not_full(self, recognizer: GestureRecognizer):
        result = recognizer.get_stable_gesture(GestureRecognizer.GESTURE_DRAWING)
        assert result == GestureRecognizer.GESTURE_DRAWING

    def test_switches_after_majority_vote(self, recognizer: GestureRecognizer):
        for _ in range(6):
            recognizer.get_stable_gesture(GestureRecognizer.GESTURE_DRAWING)
        result = recognizer.get_stable_gesture(GestureRecognizer.GESTURE_PAUSE)
        # 4 DRAWING + 1 PAUSE in buffer of 6 -> should still be DRAWING
        assert result == GestureRecognizer.GESTURE_DRAWING

    def test_switches_when_new_gesture_wins_majority(self, recognizer: GestureRecognizer):
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_DRAWING)
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_DRAWING)
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_PAUSE)
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_PAUSE)
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_PAUSE)
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_PAUSE)
        result = recognizer.get_stable_gesture(GestureRecognizer.GESTURE_PAUSE)
        assert result == GestureRecognizer.GESTURE_PAUSE

    def test_returns_previous_when_no_majority(self, recognizer: GestureRecognizer):
        for g in [GestureRecognizer.GESTURE_DRAWING,
                  GestureRecognizer.GESTURE_PAUSE,
                  GestureRecognizer.GESTURE_DRAWING,
                  GestureRecognizer.GESTURE_PAUSE,
                  GestureRecognizer.GESTURE_DRAWING,
                  GestureRecognizer.GESTURE_PAUSE]:
            recognizer.get_stable_gesture(g)
        result = recognizer.get_stable_gesture(GestureRecognizer.GESTURE_NONE)
        assert result in (GestureRecognizer.GESTURE_DRAWING, GestureRecognizer.GESTURE_PAUSE)


class TestSmoothCoordinates:
    def test_single_point_returns_same(self, recognizer: GestureRecognizer):
        x, y = recognizer.smooth_coordinates(100, 200)
        assert (x, y) == (100, 200)

    def test_averages_two_points(self, recognizer: GestureRecognizer):
        recognizer.smooth_coordinates(10, 20)
        x, y = recognizer.smooth_coordinates(20, 40)
        assert (x, y) == (15, 30)

    def test_averages_three_points(self, recognizer: GestureRecognizer):
        recognizer.smooth_coordinates(10, 10)
        recognizer.smooth_coordinates(20, 20)
        x, y = recognizer.smooth_coordinates(30, 30)
        assert (x, y) == (20, 20)

    def test_buffer_clears_on_reset(self, recognizer: GestureRecognizer):
        recognizer.smooth_coordinates(100, 200)
        recognizer.reset_buffers()
        x, y = recognizer.smooth_coordinates(50, 60)
        assert (x, y) == (50, 60)


class TestGetFingerPositions:
    def test_returns_index_and_palm(self, recognizer: GestureRecognizer):
        lm = make_landmarks([False, True, False, False, False])
        # Fake landmarks all have (0.5, 0.5) except modified tips
        result = recognizer.get_finger_positions((480, 640), lm)
        assert "index" in result
        assert "palm" in result

    def test_coordinates_are_ints(self, recognizer: GestureRecognizer):
        lm = make_landmarks([False, True, False, False, False])
        result = recognizer.get_finger_positions((480, 640), lm)
        assert isinstance(result["index"][0], int)
        assert isinstance(result["index"][1], int)


class TestDrawHandLandmarks:
    def test_draws_without_error(self, recognizer: GestureRecognizer):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        lm = make_landmarks([False, True, False, False, False])
        recognizer.draw_hand_landmarks(frame, lm)
        # Some pixels should be drawn (green lines, blue dots)
        assert np.any(frame != 0)


class TestResetBuffers:
    def test_clears_both_gesture_and_smoothing(self, recognizer: GestureRecognizer):
        recognizer.get_stable_gesture(GestureRecognizer.GESTURE_DRAWING)
        recognizer.smooth_coordinates(10, 10)
        recognizer.reset_buffers()
        assert len(recognizer.gesture_buffer) == 0
        assert len(recognizer.smoothing_buffer) == 0
