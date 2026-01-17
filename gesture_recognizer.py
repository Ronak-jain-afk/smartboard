"""
Gesture Recognizer Module
=========================
Handles hand detection and gesture recognition using MediaPipe.
Includes gesture buffering for smooth recognition.
"""

import logging
from collections import deque
from typing import Dict, Tuple, Optional, List
import mediapipe as mp

from config import (
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    MAX_NUM_HANDS,
    STATIC_IMAGE_MODE,
    FINGER_EXTENSION_THRESHOLD,
    GESTURE_BUFFER_SIZE,
    GESTURE_CONFIRMATION_FRAMES,
    GESTURE_CONFIRMATION_COUNT,
    SMOOTHING_BUFFER_SIZE
)

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """Handles hand gesture detection and recognition."""
    
    # Gesture type constants
    GESTURE_DRAWING = "drawing"
    GESTURE_PALM_ERASE = "palm_erase"
    GESTURE_SHAPE_MODE = "shape_mode"
    GESTURE_PAUSE = "pause"
    GESTURE_NONE = "none"
    
    def __init__(self):
        """Initialize the gesture recognizer with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=STATIC_IMAGE_MODE,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture smoothing buffer using deque for O(1) operations
        self.gesture_buffer: deque = deque(maxlen=GESTURE_BUFFER_SIZE)
        
        # Coordinate smoothing buffer
        self.smoothing_buffer: deque = deque(maxlen=SMOOTHING_BUFFER_SIZE)
        
        # Landmark indices for finger detection
        self._finger_tips = (8, 12, 16, 20)  # Index, Middle, Ring, Pinky tips
        self._finger_mcps = (5, 9, 13, 17)   # Corresponding MCP joints
    
    def process_frame(self, rgb_frame) -> Optional[object]:
        """
        Process a frame and return hand detection results.
        
        Args:
            rgb_frame: RGB image frame from camera.
            
        Returns:
            MediaPipe hand detection results, or None if no hands detected.
        """
        return self.hands.process(rgb_frame)
    
    def get_finger_positions(
        self, 
        frame_shape: Tuple[int, int], 
        hand_landmarks
    ) -> Dict[str, Tuple[int, int]]:
        """
        Extract finger and palm positions from hand landmarks.
        
        Args:
            frame_shape: Tuple of (height, width) of the frame.
            hand_landmarks: MediaPipe hand landmarks.
            
        Returns:
            Dictionary with 'index' and 'palm' positions as (x, y) tuples.
        """
        height, width = frame_shape
        landmarks = hand_landmarks.landmark
        
        # Index finger tip position
        index_tip = landmarks[8]
        index_pos = (int(index_tip.x * width), int(index_tip.y * height))
        
        # Palm center (average of wrist and middle MCP)
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        palm_x = int((wrist.x + middle_mcp.x) * width / 2)
        palm_y = int((wrist.y + middle_mcp.y) * height / 2)
        
        return {
            'index': index_pos,
            'palm': (palm_x, palm_y)
        }
    
    def _detect_extended_fingers(self, landmarks) -> List[bool]:
        """
        Detect which fingers are extended.
        
        Args:
            landmarks: MediaPipe hand landmarks.
            
        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky].
        """
        lm = landmarks.landmark
        fingers = []
        
        # Thumb detection (horizontal movement)
        thumb_extended = abs(lm[4].x - lm[2].x) > FINGER_EXTENSION_THRESHOLD
        fingers.append(thumb_extended)
        
        # Other fingers (vertical movement)
        for tip_id, mcp_id in zip(self._finger_tips, self._finger_mcps):
            finger_extended = (lm[mcp_id].y - lm[tip_id].y) > FINGER_EXTENSION_THRESHOLD
            fingers.append(finger_extended)
        
        return fingers
    
    def detect_gesture(self, hand_landmarks) -> str:
        """
        Detect the current hand gesture.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.
            
        Returns:
            Gesture string constant.
        """
        fingers = self._detect_extended_fingers(hand_landmarks)
        
        # Count extended fingers (excluding thumb)
        extended_count = sum(fingers[1:])
        
        index_up = fingers[1]
        middle_up = fingers[2]
        
        # Gesture classification
        if extended_count >= 4:  # Open palm
            return self.GESTURE_PALM_ERASE
        elif extended_count == 1 and index_up:  # Only index finger
            return self.GESTURE_DRAWING
        elif extended_count == 2 and index_up and middle_up:  # Peace sign
            return self.GESTURE_SHAPE_MODE
        elif extended_count == 0:  # Fist
            return self.GESTURE_PAUSE
        else:
            return self.GESTURE_NONE
    
    def get_stable_gesture(self, current_gesture: str) -> str:
        """
        Get a stabilized gesture using the gesture buffer.
        Helps reduce flickering between gesture states.
        
        Args:
            current_gesture: The currently detected gesture.
            
        Returns:
            Stabilized gesture string.
        """
        self.gesture_buffer.append(current_gesture)
        
        if len(self.gesture_buffer) < GESTURE_CONFIRMATION_FRAMES:
            return current_gesture
        
        # Count gestures in recent frames using Counter-like logic
        # Iterate directly over deque slice (more efficient than list conversion)
        gesture_counts: Dict[str, int] = {}
        for i, g in enumerate(self.gesture_buffer):
            if i >= len(self.gesture_buffer) - GESTURE_CONFIRMATION_FRAMES:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Find most common gesture
        most_common = max(gesture_counts, key=gesture_counts.get)
        
        if gesture_counts[most_common] >= GESTURE_CONFIRMATION_COUNT:
            return most_common
        
        # Fall back to previous gesture if no clear majority
        if len(self.gesture_buffer) > 1:
            # Get second to last item from deque
            return self.gesture_buffer[-2]
        
        return current_gesture
    
    def smooth_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """
        Apply smoothing to coordinates to reduce jitter.
        
        Args:
            x: Raw x coordinate.
            y: Raw y coordinate.
            
        Returns:
            Smoothed (x, y) tuple.
        """
        self.smoothing_buffer.append((x, y))
        
        if len(self.smoothing_buffer) < 2:
            return x, y
        
        # Use last 3 points for smoothing (or fewer if not available)
        points_to_use = min(3, len(self.smoothing_buffer))
        
        # Calculate average directly from deque tail
        total_x = 0
        total_y = 0
        for i in range(1, points_to_use + 1):
            idx = -i
            total_x += self.smoothing_buffer[idx][0]
            total_y += self.smoothing_buffer[idx][1]
        
        avg_x = total_x // points_to_use
        avg_y = total_y // points_to_use
        
        return avg_x, avg_y
    
    def draw_hand_landmarks(self, frame, hand_landmarks) -> None:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: BGR frame to draw on.
            hand_landmarks: MediaPipe hand landmarks.
        """
        self.mp_drawing.draw_landmarks(
            frame, 
            hand_landmarks, 
            self.mp_hands.HAND_CONNECTIONS
        )
    
    def reset_buffers(self) -> None:
        """Clear gesture and smoothing buffers."""
        self.gesture_buffer.clear()
        self.smoothing_buffer.clear()
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
