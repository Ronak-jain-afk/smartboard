"""Tests for CanvasManager."""

import numpy as np
import pytest

from canvas_manager import CanvasManager


class TestCanvasInit:
    def test_creates_canvas_with_correct_dimensions(self):
        cm = CanvasManager(640, 480)
        assert cm.width == 640
        assert cm.height == 480
        assert cm.canvas.shape == (480, 640, 3)

    def test_initial_canvas_is_black(self):
        cm = CanvasManager(100, 100)
        assert np.all(cm.canvas == 0)

    def test_initial_history_has_one_state(self):
        cm = CanvasManager(100, 100)
        idx, total = cm.get_history_info()
        assert idx == 0
        assert total == 1

    def test_default_color_is_green(self):
        cm = CanvasManager(100, 100)
        assert cm.current_color == (0, 255, 0)

    def test_default_shape_is_line(self):
        cm = CanvasManager(100, 100)
        assert cm.current_shape == "line"


class TestDrawing:
    def test_draw_line_changes_pixels_in_expected_area(self, canvas: CanvasManager):
        canvas.draw_line(10, 10)
        canvas.draw_line(20, 20)
        # Pixels along the line should be non-zero (green)
        assert np.any(canvas.canvas[15, 15] != 0)

    def test_first_draw_does_not_error(self, canvas: CanvasManager):
        canvas.draw_line(50, 50)
        assert canvas._has_prev_position

    def test_reset_draw_position_clears_prev(self, canvas: CanvasManager):
        canvas.draw_line(10, 10)
        canvas.reset_draw_position()
        assert not canvas._has_prev_position

    def test_erase_at_creates_black_circle(self, canvas: CanvasManager):
        canvas.canvas[:] = (255, 255, 255)
        canvas.erase_at(50, 50)
        # eraser_color is (0,0,0), so pixel at center should be 0
        assert tuple(canvas.canvas[50, 50]) == (0, 0, 0)


class TestUndoRedo:
    def test_undo_returns_true_when_possible(self, canvas: CanvasManager):
        canvas.draw_line(10, 10)
        canvas.save_canvas_state()
        assert canvas.undo() is True

    def test_undo_returns_false_when_at_beginning(self, canvas: CanvasManager):
        assert canvas.undo() is False

    def test_redo_returns_true_when_possible(self, canvas: CanvasManager):
        canvas.draw_line(10, 10)
        canvas.save_canvas_state()
        canvas.undo()
        assert canvas.redo() is True

    def test_redo_returns_false_when_at_end(self, canvas: CanvasManager):
        assert canvas.redo() is False

    def test_undo_restores_previous_canvas_state(self, canvas: CanvasManager):
        # Save initial (blank) state, draw, save, undo
        canvas.save_canvas_state()
        canvas.draw_line(10, 10)
        canvas.draw_line(20, 20)
        canvas.save_canvas_state()
        original = canvas.canvas.copy()
        canvas.undo()
        assert not np.array_equal(canvas.canvas, original)

    def test_history_index_tracking(self, canvas: CanvasManager):
        idx1, total1 = canvas.get_history_info()
        assert idx1 == 0 and total1 == 1
        canvas.draw_line(10, 10)
        canvas.save_canvas_state()
        idx2, total2 = canvas.get_history_info()
        assert idx2 == 1 and total2 == 2
        canvas.undo()
        idx3, total3 = canvas.get_history_info()
        assert idx3 == 0 and total3 == 2

    def test_new_draw_clears_redo_history(self, canvas: CanvasManager):
        canvas.save_canvas_state()
        canvas.draw_line(10, 10)
        canvas.save_canvas_state()
        canvas.undo()
        canvas.draw_line(99, 99)
        canvas.save_canvas_state()
        # Redo should be at end of history
        assert canvas.redo() is False


class TestShapes:
    def test_start_shape_sets_start_point(self, canvas: CanvasManager):
        assert canvas.shape_start_point is None
        canvas.start_shape(100, 200)
        assert canvas.shape_start_point == (100, 200)

    def test_complete_shape_returns_false_without_start(self, canvas: CanvasManager):
        assert canvas.complete_shape(50, 50) is False

    def test_complete_shape_draws_line(self, canvas: CanvasManager):
        canvas.current_shape = "line"
        canvas.start_shape(10, 10)
        assert canvas.complete_shape(100, 100) is True
        assert canvas.shape_start_point is None
        assert np.any(canvas.canvas != 0)

    def test_complete_shape_draws_rectangle(self, canvas: CanvasManager):
        canvas.current_shape = "rectangle"
        canvas.start_shape(10, 10)
        canvas.complete_shape(100, 100)
        assert np.any(canvas.canvas != 0)

    def test_cancel_shape_clears_start_point(self, canvas: CanvasManager):
        canvas.start_shape(10, 10)
        canvas.cancel_shape()
        assert canvas.shape_start_point is None

    def test_cycle_shape(self, canvas: CanvasManager):
        assert canvas.current_shape == "line"
        canvas.cycle_shape()
        assert canvas.current_shape == "rectangle"
        canvas.cycle_shape()
        assert canvas.current_shape == "circle"
        canvas.cycle_shape()
        assert canvas.current_shape == "arrow"
        canvas.cycle_shape()
        assert canvas.current_shape == "line"

    def test_get_shape_preview_returns_none_without_shape(self, canvas: CanvasManager):
        assert canvas.get_shape_preview((50, 50)) is None

    def test_get_shape_preview_returns_dict_with_shape(self, canvas: CanvasManager):
        canvas.start_shape(10, 10)
        preview = canvas.get_shape_preview((100, 100))
        assert preview is not None
        assert preview["type"] == "line"
        assert preview["start"] == (10, 10)
        assert preview["end"] == (100, 100)


class TestBrushAndColor:
    def test_set_color_valid(self, canvas: CanvasManager):
        assert canvas.set_color(0) is True
        assert canvas.current_color_index == 0

    def test_set_color_invalid(self, canvas: CanvasManager):
        assert canvas.set_color(99) is False

    def test_increase_brush_size(self, canvas: CanvasManager):
        initial = canvas.brush_thickness
        canvas.increase_brush_size()
        assert canvas.brush_thickness > initial

    def test_decrease_brush_size(self, canvas: CanvasManager):
        canvas.current_brush_index = 2
        canvas.brush_thickness = canvas.brush_sizes[2]
        initial = canvas.brush_thickness
        canvas.decrease_brush_size()
        assert canvas.brush_thickness < initial

    def test_brush_size_bounds(self, canvas: CanvasManager):
        canvas.current_brush_index = 0
        canvas.brush_thickness = canvas.brush_sizes[0]
        result = canvas.decrease_brush_size()
        assert result == canvas.brush_sizes[0]

        canvas.current_brush_index = len(canvas.brush_sizes) - 1
        canvas.brush_thickness = canvas.brush_sizes[-1]
        result = canvas.increase_brush_size()
        assert result == canvas.brush_sizes[-1]


class TestClearCanvas:
    def test_clear_makes_canvas_black(self, canvas: CanvasManager):
        canvas.canvas[:] = (255, 255, 255)
        canvas.clear_canvas()
        assert np.all(canvas.canvas == 0)

    def test_clear_saves_undo_state(self, canvas: CanvasManager):
        canvas.canvas[:] = (255, 255, 255)
        canvas.clear_canvas()
        idx, total = canvas.get_history_info()
        assert idx == 1 and total == 2


class TestTrail:
    def test_trail_starts_as_default(self, canvas: CanvasManager):
        assert canvas.show_trail is True

    def test_toggle_trail(self, canvas: CanvasManager):
        canvas.toggle_trail()
        assert canvas.show_trail is False
        canvas.toggle_trail()
        assert canvas.show_trail is True

    def test_toggle_clears_points_when_disabled(self, canvas: CanvasManager):
        canvas.draw_line(10, 10)
        canvas.draw_line(20, 20)
        assert len(canvas.trail_points) > 0
        canvas.toggle_trail()
        assert len(canvas.trail_points) == 0


class TestGetCurrentColorName:
    def test_returns_correct_name(self, canvas: CanvasManager):
        canvas.set_color(0)
        assert canvas.get_current_color_name() == "red"

    def test_updates_after_color_change(self, canvas: CanvasManager):
        canvas.set_color(2)
        assert canvas.get_current_color_name() == "blue"


def test_save_canvas_state_preserves_data(canvas: CanvasManager):
    canvas.canvas[100, 100] = (255, 0, 0)
    canvas.save_canvas_state()
    saved = canvas.canvas.copy()
    canvas.canvas[100, 100] = (0, 0, 0)
    canvas.save_canvas_state()
    canvas.undo()
    assert np.array_equal(canvas.canvas, saved)
