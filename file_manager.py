"""
File Manager Module
===================
Handles all file I/O operations including saving, auto-saving, and cleanup.
Includes proper error handling and path validation.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from config import (
    AUTO_SAVE_DIR,
    AUTO_SAVE_INTERVAL,
    AUTO_SAVE_MAX_FILES,
    AUTO_SAVE_FORMAT
)

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations for the SmartBoard application."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the FileManager.
        
        Args:
            base_dir: Base directory for file operations. Defaults to script directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.absolute()
        self.auto_save_dir = self.base_dir / AUTO_SAVE_DIR
        self.last_save_time = time.time()
        self.auto_save_interval = AUTO_SAVE_INTERVAL
        self.max_auto_saves = AUTO_SAVE_MAX_FILES
        
        self._ensure_auto_save_directory()
    
    def _ensure_auto_save_directory(self) -> bool:
        """
        Ensure the auto-save directory exists.
        
        Returns:
            True if directory exists or was created, False on error.
        """
        try:
            self.auto_save_dir.mkdir(parents=True, exist_ok=True)
            return True
        except OSError as e:
            logger.error(f"Failed to create auto-save directory: {e}")
            return False
    
    def _cleanup_old_auto_saves(self) -> int:
        """
        Remove old auto-save files, keeping only the most recent ones.
        
        Returns:
            Number of files deleted.
        """
        try:
            auto_saves = sorted(
                self.auto_save_dir.glob(f"auto_save_*.{AUTO_SAVE_FORMAT}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            files_to_delete = auto_saves[self.max_auto_saves:]
            deleted_count = 0
            
            for old_file in files_to_delete:
                try:
                    old_file.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted old auto-save: {old_file.name}")
                except OSError as e:
                    logger.warning(f"Failed to delete {old_file}: {e}")
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error during auto-save cleanup: {e}")
            return 0
    
    def auto_save_canvas(self, canvas: np.ndarray) -> Optional[str]:
        """
        Auto-save the canvas if the interval has passed.
        
        Args:
            canvas: The canvas numpy array to save.
            
        Returns:
            Filename if saved, None otherwise.
        """
        if canvas is None:
            return None
        
        current_time = time.time()
        if current_time - self.last_save_time < self.auto_save_interval:
            return None
        
        if not self._ensure_auto_save_directory():
            return None
        
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = self.auto_save_dir / f"auto_save_{timestamp}.{AUTO_SAVE_FORMAT}"
            
            success = cv2.imwrite(str(filename), canvas)
            
            if success:
                self.last_save_time = current_time
                self._cleanup_old_auto_saves()
                logger.info(f"Auto-saved: {filename.name}")
                return str(filename)
            else:
                logger.error(f"Failed to write auto-save file: {filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error during auto-save: {e}")
            return None
    
    def save_canvas(self, canvas: np.ndarray, filename: Optional[str] = None) -> Optional[str]:
        """
        Save the canvas to a file.
        
        Args:
            canvas: The canvas numpy array to save.
            filename: Optional custom filename. Auto-generates if not provided.
            
        Returns:
            Full path of saved file, or None on error.
        """
        if canvas is None:
            logger.warning("Cannot save: canvas is None")
            return None
        
        try:
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"finger_drawing_{timestamp}.{AUTO_SAVE_FORMAT}"
            
            filepath = self.base_dir / filename
            
            success = cv2.imwrite(str(filepath), canvas)
            
            if success:
                logger.info(f"Canvas saved: {filepath}")
                return str(filepath)
            else:
                logger.error(f"Failed to write canvas file: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving canvas: {e}")
            return None
    
    def get_auto_save_list(self) -> List[Path]:
        """
        Get list of auto-save files sorted by modification time (newest first).
        
        Returns:
            List of Path objects for auto-save files.
        """
        try:
            return sorted(
                self.auto_save_dir.glob(f"auto_save_*.{AUTO_SAVE_FORMAT}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error listing auto-saves: {e}")
            return []
    
    def load_canvas(self, filepath: str) -> Optional[np.ndarray]:
        """
        Load a canvas from a file.
        
        Args:
            filepath: Path to the image file.
            
        Returns:
            Loaded canvas as numpy array, or None on error.
        """
        try:
            path = Path(filepath)
            if not path.exists():
                logger.error(f"File not found: {filepath}")
                return None
            
            canvas = cv2.imread(str(path))
            if canvas is None:
                logger.error(f"Failed to read image: {filepath}")
                return None
            
            logger.info(f"Loaded canvas: {filepath}")
            return canvas
            
        except Exception as e:
            logger.error(f"Error loading canvas: {e}")
            return None
