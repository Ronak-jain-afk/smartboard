"""
Camera Manager Module
=====================
Handles camera initialization, frame capture, and resolution management
with fallback support for different camera configurations.
"""

import logging
from typing import Tuple, Optional, List
import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FALLBACK_WIDTHS,
    CAMERA_FALLBACK_HEIGHTS
)

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages camera operations with fallback support."""
    
    def __init__(self, camera_index: int = CAMERA_INDEX):
        """
        Initialize the camera manager.
        
        Args:
            camera_index: Index of camera to use.
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.width: int = 0
        self.height: int = 0
        self._is_initialized: bool = False
    
    def initialize(self) -> bool:
        """
        Initialize the camera with resolution fallback.
        
        Returns:
            True if camera initialized successfully, False otherwise.
        """
        # Try to open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            # Try alternative camera indices
            for alt_index in range(4):
                if alt_index == self.camera_index:
                    continue
                logger.info(f"Trying camera index {alt_index}...")
                self.cap = cv2.VideoCapture(alt_index)
                if self.cap.isOpened():
                    self.camera_index = alt_index
                    break
        
        if not self.cap.isOpened():
            logger.error("Could not open any camera")
            return False
        
        # Try to set preferred resolution with fallback
        self._set_resolution_with_fallback()
        
        # Read actual resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera initialized: index={self.camera_index}, "
                   f"resolution={self.width}x{self.height}")
        
        self._is_initialized = True
        return True
    
    def _set_resolution_with_fallback(self) -> Tuple[int, int]:
        """
        Try to set camera resolution with fallback options.
        
        Returns:
            Tuple of (width, height) actually set.
        """
        resolutions = list(zip(CAMERA_FALLBACK_WIDTHS, CAMERA_FALLBACK_HEIGHTS))
        
        # Try preferred resolution first
        preferred = (CAMERA_WIDTH, CAMERA_HEIGHT)
        if preferred not in resolutions:
            resolutions.insert(0, preferred)
        
        for width, height in resolutions:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify resolution was actually set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                logger.debug(f"Resolution set to {width}x{height}")
                return (width, height)
        
        # Return whatever resolution the camera defaulted to
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.warning(f"Using camera default resolution: {actual_width}x{actual_height}")
        return (actual_width, actual_height)
    
    def read_frame(self, flip: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Args:
            flip: Whether to flip the frame horizontally (mirror).
            
        Returns:
            Tuple of (success, frame).
        """
        if not self._is_initialized or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return False, None
        
        if flip:
            frame = cv2.flip(frame, 1)
        
        return True, frame
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get the current frame dimensions.
        
        Returns:
            Tuple of (width, height).
        """
        return self.width, self.height
    
    def is_opened(self) -> bool:
        """Check if camera is opened and initialized."""
        return self._is_initialized and self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_initialized = False
        logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
