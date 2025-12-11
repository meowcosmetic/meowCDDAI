"""
Video Processing với Context Manager - Resource Management
"""
import cv2
import logging
import os
from contextlib import contextmanager
from typing import Optional, Tuple, Iterator
import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def video_capture(path: str, use_gpu: bool = False) -> Iterator[cv2.VideoCapture]:
    """
    Context manager cho VideoCapture - đảm bảo cleanup
    """
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {path}")
        
        yield cap
    finally:
        if cap is not None:
            cap.release()
            logger.debug(f"[Video] Released video capture: {path}")


@contextmanager
def safe_file_cleanup(file_path: str, max_retries: int = 3, retry_delay: float = 0.1):
    """
    Context manager để cleanup file an toàn với retry mechanism
    """
    import time
    try:
        yield file_path
    finally:
        if file_path and os.path.exists(file_path):
            for attempt in range(max_retries):
                try:
                    os.remove(file_path)
                    logger.debug(f"[Video] Removed temp file: {file_path}")
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logger.warning(f"[Video] Không thể xóa file sau {max_retries} lần thử: {file_path}")


class VideoProcessor:
    """Class để xử lý video với các utilities"""
    
    def __init__(self, max_width: int = 1280, default_fps: int = 30):
        self.max_width = max_width
        self.default_fps = default_fps
    
    def get_video_properties(self, cap: cv2.VideoCapture) -> Tuple[int, int, float]:
        """
        Get video properties: width, height, fps
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else self.default_fps
        
        return width, height, fps
    
    def resize_frame_if_needed(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize frame nếu quá lớn, return (resized_frame, scale)
        """
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            new_width = self.max_width
            new_height = int(h * scale)
            resized = cv2.resize(frame, (new_width, new_height))
            return resized, scale
        return frame, 1.0
    
    def validate_video_file(self, file_path: str, max_size_mb: int = 500) -> bool:
        """
        Validate video file (size, format, etc.)
        """
        if not os.path.exists(file_path):
            return False
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            logger.warning(f"[Video] File quá lớn: {file_size_mb:.2f}MB (max: {max_size_mb}MB)")
            return False
        
        # Check if can open
        try:
            with video_capture(file_path) as cap:
                if not cap.isOpened():
                    return False
                # Try to read first frame
                ret, _ = cap.read()
                return ret
        except Exception as e:
            logger.error(f"[Video] Validation error: {str(e)}")
            return False

