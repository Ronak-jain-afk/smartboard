"""
SmartBoard Configuration
========================
Centralized configuration for the finger writing system.
All magic numbers and settings are defined here for easy modification.
"""

from typing import Dict, Tuple, List

# =============================================================================
# MediaPipe Hand Detection Settings
# =============================================================================
DETECTION_CONFIDENCE: float = 0.7
TRACKING_CONFIDENCE: float = 0.6
MAX_NUM_HANDS: int = 1
STATIC_IMAGE_MODE: bool = False

# =============================================================================
# Gesture Detection Settings
# =============================================================================
FINGER_EXTENSION_THRESHOLD: float = 0.04  # Threshold for detecting extended fingers
GESTURE_BUFFER_SIZE: int = 12             # Buffer size for gesture smoothing
GESTURE_CONFIRMATION_FRAMES: int = 6       # Recent frames to analyze
GESTURE_CONFIRMATION_COUNT: int = 4        # Min count to confirm gesture

# =============================================================================
# Camera Settings
# =============================================================================
CAMERA_INDEX: int = 0                      # Default camera index
CAMERA_WIDTH: int = 1280                   # Preferred camera width
CAMERA_HEIGHT: int = 720                   # Preferred camera height
CAMERA_FALLBACK_WIDTHS: List[int] = [1280, 1024, 800, 640]
CAMERA_FALLBACK_HEIGHTS: List[int] = [720, 768, 600, 480]

# =============================================================================
# Drawing Settings
# =============================================================================
BRUSH_SIZES: Tuple[int, ...] = (3, 5, 8, 12, 16)
DEFAULT_BRUSH_INDEX: int = 1
ERASER_THICKNESS: int = 50
SMOOTHING_BUFFER_SIZE: int = 4

# =============================================================================
# Color Palette (BGR format for OpenCV)
# =============================================================================
COLORS: Dict[str, Tuple[int, int, int]] = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'purple': (255, 0, 255),
    'cyan': (255, 255, 0),
    'white': (255, 255, 255),
    'orange': (0, 165, 255)
}
DEFAULT_COLOR_INDEX: int = 1  # Default to 'green'
ERASER_COLOR: Tuple[int, int, int] = (0, 0, 0)

# =============================================================================
# Shape Drawing Settings
# =============================================================================
AVAILABLE_SHAPES: Tuple[str, ...] = ('line', 'rectangle', 'circle', 'arrow')
DEFAULT_SHAPE: str = 'line'
ARROW_HEAD_LENGTH: int = 20
ARROW_HEAD_ANGLE: float = 0.5236  # pi/6 radians (30 degrees)

# =============================================================================
# Canvas & Display Settings
# =============================================================================
CANVAS_BLEND_ALPHA: float = 0.7           # Frame-canvas blend ratio
TRAIL_LENGTH: int = 8                      # Finger trail visualization length
SHOW_TRAIL_DEFAULT: bool = True

# =============================================================================
# History (Undo/Redo) Settings
# =============================================================================
MAX_HISTORY_SIZE: int = 10

# =============================================================================
# Auto-Save Settings
# =============================================================================
AUTO_SAVE_INTERVAL: int = 30              # Seconds between auto-saves
AUTO_SAVE_DIR: str = "auto_saves"
AUTO_SAVE_MAX_FILES: int = 10             # Maximum auto-save files to keep
AUTO_SAVE_FORMAT: str = "jpg"

# =============================================================================
# UI Settings
# =============================================================================
UI_PANEL_COLOR: Tuple[int, int, int] = (50, 50, 50)
UI_PANEL_POSITION: Tuple[int, int, int, int] = (10, 10, 350, 180)  # x, y, width, height
UI_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)
UI_SECONDARY_TEXT_COLOR: Tuple[int, int, int] = (200, 200, 200)

# Gesture display names (pre-computed to avoid string operations in hot path)
GESTURE_DISPLAY_NAMES: Dict[str, str] = {
    'drawing': 'DRAWING',
    'palm_erase': 'PALM ERASE',
    'shape_mode': 'SHAPE MODE',
    'menu': 'MENU',
    'pause': 'PAUSE',
    'none': 'NONE'
}

GESTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
    'drawing': (0, 255, 0),      # Green
    'palm_erase': (0, 0, 255),   # Red
    'shape_mode': (255, 255, 0), # Cyan
    'menu': (255, 0, 255),       # Magenta
    'pause': (255, 255, 255),    # White
    'none': (255, 255, 255)      # White
}

# =============================================================================
# Crosshair & Cursor Settings
# =============================================================================
CROSSHAIR_SIZE: int = 15
CROSSHAIR_THICKNESS: int = 2
CURSOR_CIRCLE_RADIUS: int = 8
