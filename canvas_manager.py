"""
Canvas Manager Module
=====================
Handles canvas operations including drawing, erasing, shapes,
and history management (undo/redo).
"""

import math
import logging
from collections import deque
from typing import Tuple, Optional, List
import cv2
import numpy as np

from config import (
    BRUSH_SIZES,
    DEFAULT_BRUSH_INDEX,
    ERASER_THICKNESS,
    ERASER_COLOR,
    COLORS,
    DEFAULT_COLOR_INDEX,
    AVAILABLE_SHAPES,
    DEFAULT_SHAPE,
    ARROW_HEAD_LENGTH,
    ARROW_HEAD_ANGLE,
    MAX_HISTORY_SIZE,
    TRAIL_LENGTH,
    SHOW_TRAIL_DEFAULT
)

logger = logging.getLogger(__name__)


class CanvasManager:
    """Manages the drawing canvas and related operations."""
    
    def __init__(self, width: int, height: int):
        """
        Initialize the canvas manager.
        
        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
        """
        self.width = width
        self.height = height
        
        # Initialize canvas
        self.canvas: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Drawing state
        self.prev_x: int = 0
        self.prev_y: int = 0
        
        # Color settings
        self.color_names: List[str] = list(COLORS.keys())
        self.current_color_index: int = DEFAULT_COLOR_INDEX
        self.current_color: Tuple[int, int, int] = COLORS[self.color_names[self.current_color_index]]
        self.eraser_color: Tuple[int, int, int] = ERASER_COLOR
        
        # Brush settings
        self.brush_sizes: Tuple[int, ...] = BRUSH_SIZES
        self.current_brush_index: int = DEFAULT_BRUSH_INDEX
        self.brush_thickness: int = self.brush_sizes[self.current_brush_index]
        self.eraser_thickness: int = ERASER_THICKNESS
        
        # Shape settings
        self.shapes: Tuple[str, ...] = AVAILABLE_SHAPES
        self.current_shape: str = DEFAULT_SHAPE
        self.shape_start_point: Optional[Tuple[int, int]] = None
        
        # Trail visualization
        self.show_trail: bool = SHOW_TRAIL_DEFAULT
        self.trail_points: deque = deque(maxlen=TRAIL_LENGTH)
        
        # History management using deque for O(1) operations
        self._history: deque = deque(maxlen=MAX_HISTORY_SIZE)
        self._history_index: int = -1
        
        # Save initial state
        self._save_state()
    
    def _save_state(self) -> None:
        """Save current canvas state to history, clearing any redo states."""
        # If we're not at the end of history, we need to clear redo states
        if self._history_index < len(self._history) - 1:
            # Remove all states after current index
            while len(self._history) > self._history_index + 1:
                self._history.pop()
        
        # Add current state
        self._history.append(self.canvas.copy())
        self._history_index = len(self._history) - 1
        
        logger.debug(f"Canvas state saved. History size: {len(self._history)}")
    
    def save_canvas_state(self) -> None:
        """Public method to save canvas state for undo."""
        self._save_state()
    
    def undo(self) -> bool:
        """
        Undo the last action.
        
        Returns:
            True if undo was performed, False if nothing to undo.
        """
        if self._history_index > 0:
            self._history_index -= 1
            self.canvas = self._history[self._history_index].copy()
            logger.debug(f"Undo performed. Index: {self._history_index}")
            return True
        return False
    
    def redo(self) -> bool:
        """
        Redo the last undone action.
        
        Returns:
            True if redo was performed, False if nothing to redo.
        """
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.canvas = self._history[self._history_index].copy()
            logger.debug(f"Redo performed. Index: {self._history_index}")
            return True
        return False
    
    def clear_canvas(self) -> None:
        """Clear the canvas and save state."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._save_state()
        logger.info("Canvas cleared")
    
    def draw_line(self, x: int, y: int) -> None:
        """
        Draw a line from the previous position to the current position.
        
        Args:
            x: Current x coordinate.
            y: Current y coordinate.
        """
        if self.show_trail:
            self.trail_points.append((x, y))
        
        if self.prev_x != 0 and self.prev_y != 0:
            cv2.line(
                self.canvas,
                (self.prev_x, self.prev_y),
                (x, y),
                self.current_color,
                self.brush_thickness
            )
        
        self.prev_x, self.prev_y = x, y
    
    def erase_at(self, x: int, y: int) -> None:
        """
        Erase at the specified position.
        
        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        cv2.circle(
            self.canvas,
            (x, y),
            self.eraser_thickness,
            self.eraser_color,
            -1
        )
    
    def start_shape(self, x: int, y: int) -> None:
        """
        Start drawing a shape at the specified position.
        
        Args:
            x: Start x coordinate.
            y: Start y coordinate.
        """
        if self.shape_start_point is None:
            self.shape_start_point = (x, y)
    
    def complete_shape(self, end_x: int, end_y: int) -> bool:
        """
        Complete the current shape.
        
        Args:
            end_x: End x coordinate.
            end_y: End y coordinate.
            
        Returns:
            True if shape was drawn, False otherwise.
        """
        if self.shape_start_point is None:
            return False
        
        # Save state BEFORE drawing (so we can undo the shape)
        self._save_state()
        
        self._draw_shape(
            self.shape_start_point,
            (end_x, end_y),
            self.current_shape
        )
        
        self.shape_start_point = None
        logger.info(f"Shape drawn: {self.current_shape}")
        return True
    
    def cancel_shape(self) -> None:
        """Cancel the current shape being drawn."""
        self.shape_start_point = None
    
    def _draw_shape(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        shape_type: str
    ) -> None:
        """
        Draw a shape on the canvas.
        
        Args:
            start: Start point (x, y).
            end: End point (x, y).
            shape_type: Type of shape to draw.
        """
        if shape_type == 'line':
            cv2.line(self.canvas, start, end, self.current_color, self.brush_thickness)
        
        elif shape_type == 'rectangle':
            cv2.rectangle(self.canvas, start, end, self.current_color, self.brush_thickness)
        
        elif shape_type == 'circle':
            radius = int(math.sqrt(
                (end[0] - start[0])**2 + (end[1] - start[1])**2
            ))
            cv2.circle(self.canvas, start, radius, self.current_color, self.brush_thickness)
        
        elif shape_type == 'arrow':
            self._draw_arrow(start, end)
    
    def _draw_arrow(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        """
        Draw an arrow from start to end.
        
        Args:
            start: Arrow start point.
            end: Arrow end point (with arrowhead).
        """
        # Draw main line
        cv2.line(self.canvas, start, end, self.current_color, self.brush_thickness)
        
        # Calculate arrow head
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        
        # Arrow head lines
        x1 = int(end[0] - ARROW_HEAD_LENGTH * math.cos(angle - ARROW_HEAD_ANGLE))
        y1 = int(end[1] - ARROW_HEAD_LENGTH * math.sin(angle - ARROW_HEAD_ANGLE))
        x2 = int(end[0] - ARROW_HEAD_LENGTH * math.cos(angle + ARROW_HEAD_ANGLE))
        y2 = int(end[1] - ARROW_HEAD_LENGTH * math.sin(angle + ARROW_HEAD_ANGLE))
        
        cv2.line(self.canvas, end, (x1, y1), self.current_color, self.brush_thickness)
        cv2.line(self.canvas, end, (x2, y2), self.current_color, self.brush_thickness)
    
    def get_shape_preview(
        self,
        end_point: Tuple[int, int],
        preview_color: Tuple[int, int, int] = (255, 255, 0)
    ) -> Optional[dict]:
        """
        Get shape preview parameters for drawing on the frame.
        
        Args:
            end_point: Current cursor position.
            preview_color: Color for the preview shape.
            
        Returns:
            Dictionary with shape preview info, or None if no shape in progress.
        """
        if self.shape_start_point is None:
            return None
        
        return {
            'type': self.current_shape,
            'start': self.shape_start_point,
            'end': end_point,
            'color': preview_color
        }
    
    def cycle_shape(self) -> str:
        """
        Cycle to the next shape type.
        
        Returns:
            Name of the new current shape.
        """
        current_index = self.shapes.index(self.current_shape)
        self.current_shape = self.shapes[(current_index + 1) % len(self.shapes)]
        return self.current_shape
    
    def set_color(self, color_index: int) -> bool:
        """
        Set the current drawing color by index.
        
        Args:
            color_index: Index into the color list.
            
        Returns:
            True if color was changed, False if index invalid.
        """
        if 0 <= color_index < len(self.color_names):
            self.current_color_index = color_index
            self.current_color = COLORS[self.color_names[color_index]]
            return True
        return False
    
    def increase_brush_size(self) -> int:
        """
        Increase brush size.
        
        Returns:
            New brush thickness.
        """
        if self.current_brush_index < len(self.brush_sizes) - 1:
            self.current_brush_index += 1
            self.brush_thickness = self.brush_sizes[self.current_brush_index]
        return self.brush_thickness
    
    def decrease_brush_size(self) -> int:
        """
        Decrease brush size.
        
        Returns:
            New brush thickness.
        """
        if self.current_brush_index > 0:
            self.current_brush_index -= 1
            self.brush_thickness = self.brush_sizes[self.current_brush_index]
        return self.brush_thickness
    
    def toggle_trail(self) -> bool:
        """
        Toggle trail visualization.
        
        Returns:
            New trail state.
        """
        self.show_trail = not self.show_trail
        if not self.show_trail:
            self.trail_points.clear()
        return self.show_trail
    
    def reset_draw_position(self) -> None:
        """Reset the previous drawing position."""
        self.prev_x, self.prev_y = 0, 0
    
    def get_history_info(self) -> Tuple[int, int]:
        """
        Get history information.
        
        Returns:
            Tuple of (current_index, total_states).
        """
        return len(self._history), MAX_HISTORY_SIZE
    
    def get_current_color_name(self) -> str:
        """Get the name of the current color."""
        return self.color_names[self.current_color_index]
