"""
UI Renderer Module
==================
Handles all UI drawing operations including status panels,
color palette, instructions, and visual feedback.
"""

import math
from typing import Tuple, Dict, Optional
import cv2
import numpy as np

from config import (
    COLORS,
    UI_PANEL_COLOR,
    UI_PANEL_POSITION,
    UI_TEXT_COLOR,
    UI_SECONDARY_TEXT_COLOR,
    GESTURE_DISPLAY_NAMES,
    GESTURE_COLORS,
    CROSSHAIR_SIZE,
    CROSSHAIR_THICKNESS,
    CURSOR_CIRCLE_RADIUS,
    MAX_HISTORY_SIZE,
    CANVAS_BLEND_ALPHA
)


class UIRenderer:
    """Handles rendering of UI elements on the frame."""
    
    def __init__(self):
        """Initialize the UI renderer."""
        self.color_names = list(COLORS.keys())
        self.colors = COLORS
    
    def blend_canvas_with_frame(
        self,
        frame: np.ndarray,
        canvas: np.ndarray,
        alpha: float = CANVAS_BLEND_ALPHA
    ) -> np.ndarray:
        """
        Blend the canvas with the camera frame.
        
        Args:
            frame: Camera frame.
            canvas: Drawing canvas.
            alpha: Blend ratio (0-1, higher = more frame).
            
        Returns:
            Blended frame.
        """
        return cv2.addWeighted(frame, alpha, canvas, 1 - alpha, 0)
    
    def draw_status_panel(
        self,
        frame: np.ndarray,
        gesture: str,
        current_color: Tuple[int, int, int],
        color_name: str,
        brush_thickness: int,
        current_shape: str,
        position: Tuple[int, int],
        fps: int,
        history_count: int
    ) -> None:
        """
        Draw the status information panel.
        
        Args:
            frame: Frame to draw on.
            gesture: Current gesture name.
            current_color: Current drawing color (BGR).
            color_name: Name of current color.
            brush_thickness: Current brush size.
            current_shape: Current shape type (if in shape mode).
            position: Current cursor (x, y) position.
            fps: Current FPS.
            history_count: Number of items in history.
        """
        x, y, w, h = UI_PANEL_POSITION
        
        # Draw panel background
        cv2.rectangle(frame, (x, y), (x + w, y + h), UI_PANEL_COLOR, -1)
        
        # Get gesture display text (pre-computed for efficiency)
        gesture_text = GESTURE_DISPLAY_NAMES.get(gesture, gesture.upper())
        mode_text = f"Mode: {gesture_text}"
        mode_color = GESTURE_COLORS.get(gesture, UI_TEXT_COLOR)
        
        # Draw status text
        cv2.putText(frame, mode_text, (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        cv2.putText(frame, f"Color: {color_name.upper()}", (x + 10, y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
        
        cv2.putText(frame, f"Brush Size: {brush_thickness}", (x + 10, y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_TEXT_COLOR, 1)
        
        # Shape info (only in shape mode)
        if gesture == "shape_mode":
            cv2.putText(frame, f"Shape: {current_shape.upper()}", (x + 10, y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Position and FPS
        pos_x, pos_y = position
        cv2.putText(frame, f"Position: ({pos_x}, {pos_y})", (x + 10, y + 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_SECONDARY_TEXT_COLOR, 1)
        
        cv2.putText(frame, f"FPS: {fps}", (x + 10, y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_SECONDARY_TEXT_COLOR, 1)
        
        # History info
        cv2.putText(frame, f"History: {history_count}/{MAX_HISTORY_SIZE}", (x + 10, y + 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_SECONDARY_TEXT_COLOR, 1)
    
    def draw_color_palette(
        self,
        frame: np.ndarray,
        current_color_index: int,
        y_position: int = 200
    ) -> None:
        """
        Draw the color palette.
        
        Args:
            frame: Frame to draw on.
            current_color_index: Index of currently selected color.
            y_position: Y position for the palette.
        """
        x_start = 20
        box_size = 30
        spacing = 35
        
        for i, (name, color) in enumerate(self.colors.items()):
            x_pos = x_start + i * spacing
            
            # Color box
            cv2.rectangle(frame, (x_pos, y_position), 
                         (x_pos + box_size, y_position + 20), color, -1)
            
            # Selection indicator
            if i == current_color_index:
                cv2.rectangle(frame, (x_pos - 2, y_position - 2),
                             (x_pos + box_size + 2, y_position + 22), 
                             UI_TEXT_COLOR, 2)
    
    def draw_instructions(self, frame: np.ndarray) -> None:
        """
        Draw keyboard/gesture instructions at the bottom of the frame.
        
        Args:
            frame: Frame to draw on.
        """
        height = frame.shape[0]
        
        instructions = [
            "GESTURES: Index=Draw | Open Palm=Erase | Peace=Shapes | Fist=Pause",
            "CONTROLS: 1-8=Colors | -/+=Brush | Space=Shapes | Z=Undo | X=Redo | C=Clear | S=Save"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, height - 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, UI_SECONDARY_TEXT_COLOR, 1)
    
    def draw_cursor(
        self,
        frame: np.ndarray,
        position: Tuple[int, int],
        gesture: str,
        trail_points: Optional[list] = None
    ) -> None:
        """
        Draw cursor and trail visualization.
        
        Args:
            frame: Frame to draw on.
            position: Cursor (x, y) position.
            gesture: Current gesture.
            trail_points: List of recent trail points.
        """
        x, y = position
        
        if x <= 0 or y <= 0:
            return
        
        if gesture in ["drawing", "shape_mode"]:
            color = GESTURE_COLORS.get(gesture, (0, 255, 0))
            
            # Draw crosshair
            cv2.line(frame, (x - CROSSHAIR_SIZE, y), (x + CROSSHAIR_SIZE, y), 
                    color, CROSSHAIR_THICKNESS)
            cv2.line(frame, (x, y - CROSSHAIR_SIZE), (x, y + CROSSHAIR_SIZE), 
                    color, CROSSHAIR_THICKNESS)
            cv2.circle(frame, (x, y), CURSOR_CIRCLE_RADIUS, color, CROSSHAIR_THICKNESS)
            
            # Draw trail
            if trail_points and len(trail_points) > 1:
                self._draw_trail(frame, trail_points, color)
    
    def _draw_trail(
        self,
        frame: np.ndarray,
        trail_points: list,
        color: Tuple[int, int, int]
    ) -> None:
        """
        Draw fading trail behind cursor.
        
        Args:
            frame: Frame to draw on.
            trail_points: List of trail points.
            color: Base color for the trail.
        """
        for i in range(1, len(trail_points)):
            alpha = i / len(trail_points)
            pt1 = trail_points[i - 1]
            pt2 = trail_points[i]
            trail_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, trail_color, 2)
    
    def draw_eraser_indicator(
        self,
        frame: np.ndarray,
        position: Tuple[int, int],
        eraser_size: int
    ) -> None:
        """
        Draw eraser position indicator.
        
        Args:
            frame: Frame to draw on.
            position: Palm (x, y) position.
            eraser_size: Size of the eraser.
        """
        x, y = position
        
        # Draw eraser circle
        cv2.circle(frame, (x, y), eraser_size, (0, 0, 255), 3)
        
        # Draw "ERASING" label
        cv2.putText(frame, "ERASING", (x - 40, y - eraser_size - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def draw_shape_preview(
        self,
        frame: np.ndarray,
        shape_info: dict
    ) -> None:
        """
        Draw shape preview on the frame.
        
        Args:
            frame: Frame to draw on.
            shape_info: Dictionary with shape type, start, end, and color.
        """
        if not shape_info:
            return
        
        shape_type = shape_info['type']
        start = shape_info['start']
        end = shape_info['end']
        color = shape_info['color']
        
        if shape_type == 'line':
            cv2.line(frame, start, end, color, 2)
        
        elif shape_type == 'rectangle':
            cv2.rectangle(frame, start, end, color, 2)
        
        elif shape_type == 'circle':
            radius = int(math.sqrt(
                (end[0] - start[0])**2 + (end[1] - start[1])**2
            ))
            cv2.circle(frame, start, radius, color, 2)
        
        elif shape_type == 'arrow':
            self._draw_arrow_preview(frame, start, end, color)
    
    def _draw_arrow_preview(
        self,
        frame: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int]
    ) -> None:
        """
        Draw arrow preview.
        
        Args:
            frame: Frame to draw on.
            start: Arrow start point.
            end: Arrow end point.
            color: Arrow color.
        """
        cv2.line(frame, start, end, color, 2)
        
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        arrow_length = 20
        arrow_angle = math.pi / 6
        
        x1 = int(end[0] - arrow_length * math.cos(angle - arrow_angle))
        y1 = int(end[1] - arrow_length * math.sin(angle - arrow_angle))
        x2 = int(end[0] - arrow_length * math.cos(angle + arrow_angle))
        y2 = int(end[1] - arrow_length * math.sin(angle + arrow_angle))
        
        cv2.line(frame, end, (x1, y1), color, 2)
        cv2.line(frame, end, (x2, y2), color, 2)
    
    def draw_auto_save_notification(
        self,
        frame: np.ndarray,
        filename: str,
        duration_frames: int = 60
    ) -> None:
        """
        Draw auto-save notification.
        
        Args:
            frame: Frame to draw on.
            filename: Name of saved file.
            duration_frames: How long to show notification.
        """
        height = frame.shape[0]
        
        text = f"Auto-saved: {filename}"
        cv2.putText(frame, text, (10, height - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
