"""
SmartBoard - AI-Powered Finger Writing System
==============================================
Main application entry point that coordinates all modules.

Features:
- Gesture-based drawing using index finger tracking
- Palm-based erasing
- Shape drawing mode (lines, rectangles, circles, arrows)
- Color palette with 8 colors
- Undo/redo functionality
- Auto-save with cleanup

Usage:
    python main.py

Keyboard Shortcuts:
    1-8: Change colors
    -/+: Brush size
    Space: Cycle shapes
    Z: Undo | X: Redo
    C: Clear (double-tap) | S: Save
    H: Toggle landmarks | T: Trail | Q: Quit
"""

import logging
import time
import sys
import os
from typing import Optional

import cv2
import numpy as np

# Local modules
from config import CANVAS_BLEND_ALPHA, TARGET_FPS, AUTO_SAVE_NOTIFICATION_FRAMES
from camera_manager import CameraManager
from gesture_recognizer import GestureRecognizer
from canvas_manager import CanvasManager
from ui_renderer import UIRenderer
from file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartBoard:
    """Main application class coordinating all components."""
    
    def __init__(self):
        """Initialize the SmartBoard application."""
        self.camera: Optional[CameraManager] = None
        self.gesture_recognizer: Optional[GestureRecognizer] = None
        self.canvas_manager: Optional[CanvasManager] = None
        self.ui_renderer: Optional[UIRenderer] = None
        self.file_manager: Optional[FileManager] = None
        
        # FPS tracking
        self._fps_counter: int = 0
        self._fps_start_time: float = time.time()
        self._current_fps: int = 0
        
        # State tracking
        self._is_running: bool = False
        self._prev_gesture: Optional[str] = None
        
        # Auto-save notification state
        self._notification_text: Optional[str] = None
        self._notification_frames: int = 0
        
        # Landmark overlay
        self._show_landmarks: bool = True
        
        # Clear canvas confirmation
        self._last_clear_press: float = 0.0
    
    def _initialize_components(self) -> bool:
        """
        Initialize all application components.
        
        Returns:
            True if all components initialized successfully.
        """
        logger.info("Initializing SmartBoard components...")
        
        # Initialize camera
        self.camera = CameraManager()
        if not self.camera.initialize():
            logger.error("Failed to initialize camera")
            return False
        
        # Get frame dimensions for canvas
        width, height = self.camera.get_frame_dimensions()
        
        # Initialize other components
        self.gesture_recognizer = GestureRecognizer()
        self.canvas_manager = CanvasManager(width, height)
        self.ui_renderer = UIRenderer()
        self.file_manager = FileManager()
        
        logger.info("All components initialized successfully")
        return True
    
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self._fps_counter += 1
        current_time = time.time()
        
        if current_time - self._fps_start_time >= 1.0:
            self._current_fps = self._fps_counter
            self._fps_counter = 0
            self._fps_start_time = current_time
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey().
            
        Returns:
            True if should quit, False otherwise.
        """
        if key == ord('q'):
            return True
        
        elif key == ord('c'):
            now = time.time()
            if now - self._last_clear_press < 1.0:
                self.canvas_manager.clear_canvas()
                logger.info("🗑️ Canvas cleared!")
                self._last_clear_press = 0.0
            else:
                self._last_clear_press = now
                logger.info("⌨️ Press C again within 1s to confirm clear")
        
        elif key == ord('s'):
            saved_path = self.file_manager.save_canvas(self.canvas_manager.canvas)
            if saved_path:
                logger.info(f"💾 Drawing saved as {saved_path}")
        
        elif key == ord('z'):
            if self.canvas_manager.undo():
                logger.info("↶ Undo")
        
        elif key == ord('x'):
            if self.canvas_manager.redo():
                logger.info("↷ Redo")
        
        elif key == ord('t'):
            new_state = self.canvas_manager.toggle_trail()
            logger.info(f"✨ Trail: {'ON' if new_state else 'OFF'}")
        
        elif key == ord('h'):
            self._show_landmarks = not self._show_landmarks
            logger.info(f"👁️ Landmarks: {'ON' if self._show_landmarks else 'OFF'}")
        
        elif key == ord(' '):
            new_shape = self.canvas_manager.cycle_shape()
            logger.info(f"🔷 Shape: {new_shape}")
        
        elif key == ord('-'):
            new_size = self.canvas_manager.decrease_brush_size()
            logger.info(f"🖌️ Brush size: {new_size}")
        
        elif key == ord('=') or key == ord('+'):
            new_size = self.canvas_manager.increase_brush_size()
            logger.info(f"🖌️ Brush size: {new_size}")
        
        elif ord('1') <= key <= ord('8'):
            color_index = key - ord('1')
            if self.canvas_manager.set_color(color_index):
                logger.info(f"🎨 Color: {self.canvas_manager.get_current_color_name()}")
        
        return False
    
    def _process_frame(
        self,
        positions: dict,
        gesture: str
    ) -> None:
        """
        Process a frame based on the detected gesture.
        
        Args:
            positions: Dictionary with 'index' and 'palm' positions.
            gesture: Current detected gesture.
        """
        x, y = positions['index']
        palm_x, palm_y = positions['palm']
        
        if gesture == GestureRecognizer.GESTURE_DRAWING:
            self.canvas_manager.draw_line(x, y)
        
        elif gesture == GestureRecognizer.GESTURE_PALM_ERASE:
            self.canvas_manager.erase_at(palm_x, palm_y)
        
        elif gesture == GestureRecognizer.GESTURE_SHAPE_MODE:
            self.canvas_manager.start_shape(x, y)
        
        else:
            # Reset drawing position when not drawing
            self.canvas_manager.reset_draw_position()
            if gesture != GestureRecognizer.GESTURE_SHAPE_MODE:
                self.canvas_manager.cancel_shape()
    
    def _render_frame(
        self,
        frame: np.ndarray,
        gesture: str,
        positions: dict
    ) -> np.ndarray:
        """
        Render all UI elements on the frame.
        
        Args:
            frame: Camera frame.
            gesture: Current gesture.
            positions: Cursor positions.
            
        Returns:
            Rendered frame.
        """
        # Blend canvas with frame
        combined = self.ui_renderer.blend_canvas_with_frame(
            frame,
            self.canvas_manager.canvas,
            CANVAS_BLEND_ALPHA
        )
        
        # Draw status panel
        history_index, history_total = self.canvas_manager.get_history_info()
        self.ui_renderer.draw_status_panel(
            combined,
            gesture=gesture,
            current_color=self.canvas_manager.current_color,
            color_name=self.canvas_manager.get_current_color_name(),
            brush_thickness=self.canvas_manager.brush_thickness,
            current_shape=self.canvas_manager.current_shape,
            position=positions['index'],
            fps=self._current_fps,
            history_index=history_index,
            history_total=history_total
        )
        
        # Draw color palette
        self.ui_renderer.draw_color_palette(
            combined,
            self.canvas_manager.current_color_index
        )
        
        # Draw instructions
        self.ui_renderer.draw_instructions(combined)
        
        # Draw cursor/eraser indicator based on gesture
        if gesture == GestureRecognizer.GESTURE_PALM_ERASE:
            self.ui_renderer.draw_eraser_indicator(
                combined,
                positions['palm'],
                self.canvas_manager.eraser_thickness
            )
        else:
            trail_points = list(self.canvas_manager.trail_points) if self.canvas_manager.show_trail else None
            self.ui_renderer.draw_cursor(
                combined,
                positions['index'],
                gesture,
                trail_points
            )
        
        # Draw shape preview
        if gesture == GestureRecognizer.GESTURE_SHAPE_MODE:
            shape_info = self.canvas_manager.get_shape_preview(positions['index'])
            if shape_info:
                self.ui_renderer.draw_shape_preview(combined, shape_info)
        
        # Draw auto-save notification if active
        if self._notification_frames > 0 and self._notification_text:
            self.ui_renderer.draw_auto_save_notification(
                combined, self._notification_text
            )
            self._notification_frames -= 1
        
        return combined
    
    def _print_startup_info(self) -> None:
        """Print startup information and controls."""
        print("\n" + "=" * 60)
        print("🎨 SmartBoard - AI-Powered Finger Writing System")
        print("=" * 60)
        print("\n🖐️  GESTURE CONTROLS:")
        print("   ☝️  Index finger: Drawing mode")
        print("   🖐️  Open palm: Eraser mode")
        print("   ✌️  Peace sign: Shape drawing mode")
        print("   ✊  Fist: Pause")
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("   1-8: Change colors")
        print("   -/+: Brush size")
        print("   Space: Cycle shapes")
        print("   Z: Undo | X: Redo")
        print("   C: Clear | S: Save")
        print("   H: Toggle landmarks | T: Trail | Q: Quit")
        print("=" * 60 + "\n")
    
    def run(self) -> None:
        """Run the main application loop."""
        # Initialize all components
        if not self._initialize_components():
            logger.error("Failed to initialize. Exiting.")
            return
        
        self._print_startup_info()
        self._is_running = True
        
        try:
            while self._is_running:
                loop_start = time.time()
                
                # Read frame
                ret, frame = self.camera.read_frame(flip=True)
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                height, width = frame.shape[:2]
                
                # Process hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.gesture_recognizer.process_frame(rgb_frame)
                
                # Default values
                current_gesture = GestureRecognizer.GESTURE_NONE
                positions = {'index': (0, 0), 'palm': (0, 0)}
                
                # Process hand landmarks if detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        if self._show_landmarks:
                            self.gesture_recognizer.draw_hand_landmarks(frame, hand_landmarks)
                        
                        # Get finger positions
                        positions = self.gesture_recognizer.get_finger_positions(
                            (height, width),
                            hand_landmarks
                        )
                        
                        # Smooth coordinates
                        smooth_x, smooth_y = self.gesture_recognizer.smooth_coordinates(
                            positions['index'][0],
                            positions['index'][1]
                        )
                        positions['index'] = (smooth_x, smooth_y)
                        
                        # Detect and stabilize gesture
                        raw_gesture = self.gesture_recognizer.detect_gesture(hand_landmarks)
                        current_gesture = self.gesture_recognizer.get_stable_gesture(raw_gesture)
                        
                        # Save canvas state at start of a new stroke
                        if (self._prev_gesture is not None
                                and current_gesture in (GestureRecognizer.GESTURE_DRAWING, GestureRecognizer.GESTURE_PALM_ERASE)
                                and self._prev_gesture not in (GestureRecognizer.GESTURE_DRAWING, GestureRecognizer.GESTURE_PALM_ERASE)):
                            self.canvas_manager.save_canvas_state()
                        
                        # Reset smoothing buffer when entering shape mode
                        if (self._prev_gesture is not None
                                and current_gesture == GestureRecognizer.GESTURE_SHAPE_MODE
                                and self._prev_gesture != GestureRecognizer.GESTURE_SHAPE_MODE):
                            self.gesture_recognizer.reset_buffers()
                        
                        # Process gesture actions
                        self._process_frame(positions, current_gesture)
                        
                        self._prev_gesture = current_gesture
                else:
                    self._prev_gesture = GestureRecognizer.GESTURE_NONE
                
                # Check for shape completion via Enter key
                key = cv2.waitKey(1) & 0xFF
                if key == 13 and current_gesture == GestureRecognizer.GESTURE_SHAPE_MODE:
                    if self.canvas_manager.complete_shape(positions['index'][0], positions['index'][1]):
                        logger.info(f"✅ {self.canvas_manager.current_shape} drawn!")
                
                # Render frame with UI
                combined_frame = self._render_frame(frame, current_gesture, positions)
                
                # Update FPS
                self._update_fps()
                
                # Auto-save check
                saved_file = self.file_manager.auto_save_canvas(self.canvas_manager.canvas)
                if saved_file:
                    logger.info(f"📁 Auto-saved: {saved_file}")
                    self._notification_text = os.path.basename(saved_file)
                    self._notification_frames = AUTO_SAVE_NOTIFICATION_FRAMES
                
                # Display frame
                cv2.imshow('SmartBoard - Finger Writing System', combined_frame)
                
                # Exit if window was closed (e.g., clicked X)
                if cv2.getWindowProperty('SmartBoard - Finger Writing System', cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # Cap frame rate to save CPU
                if TARGET_FPS > 0:
                    elapsed = time.time() - loop_start
                    sleep_time = 1.0 / TARGET_FPS - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Handle keyboard input
                if key != 255 and key != 13:  # Not waiting and not Enter
                    if self._handle_keyboard_input(key):
                        break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Shutting down...")
        
        if self.camera:
            self.camera.release()
        
        if self.gesture_recognizer:
            self.gesture_recognizer.close()
        
        cv2.destroyAllWindows()
        logger.info("🏁 SmartBoard shutdown complete")


def main():
    """Application entry point."""
    print("🚀 Starting SmartBoard...")
    
    try:
        app = SmartBoard()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
