"""
Configuration và constants cho Gaze Analysis
"""
import logging
import os
import cv2
import sys

# Import config để sử dụng GPU settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import Config

logger = logging.getLogger(__name__)

# Import draw_annotations riêng - luôn cần thiết cho visualization
try:
    from ..gaze_tracking.visualizer import draw_annotations
except ImportError as e:
    logger.error(f"[Gaze] Không thể import draw_annotations: {str(e)}")
    raise ImportError(f"draw_annotations là bắt buộc. Vui lòng kiểm tra gaze_tracking/visualizer.py: {str(e)}")

# Import gaze tracking modules
try:
    from ..gaze_tracking import (
        GazeConfig, GPUManager, GazeEstimator3D,
        GazeWanderingDetector, FocusTimeline,
        FatigueDetector, FocusLevelCalculator
    )
    from ..gaze_tracking.gaze_stability import (
        ImprovedGazeStabilityCalculator,
        calculate_interocular_distance
    )
    from ..gaze_tracking.object_detector import ObjectDetector
    from ..gaze_tracking.face_detector import create_face_detector
    GAZE_TRACKING_MODULES_AVAILABLE = True
except ImportError:
    GAZE_TRACKING_MODULES_AVAILABLE = False
    logger.warning("[Gaze] Gaze tracking modules không available, sử dụng logic cũ")

# Emotion detection đã bị tắt theo yêu cầu
EMOTION_DETECTION_AVAILABLE = False

# Lazy import MediaPipe - chỉ import khi cần
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("[Gaze] MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe")
    # Tạo dummy để tránh lỗi
    mp = None
    mp_face_mesh = None
    mp_drawing = None

# GPU detection
USE_GPU = Config.USE_GPU.lower() if hasattr(Config, 'USE_GPU') else "auto"
GPU_AVAILABLE = False
GPU_DEVICE_ID = 0

# Kiểm tra GPU cho OpenCV
try:
    # Kiểm tra OpenCV có build với CUDA không
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        GPU_DEVICE_ID = Config.GPU_DEVICE_ID if hasattr(Config, 'GPU_DEVICE_ID') else 0
        logger.info(f"[Gaze] ✅ OpenCV GPU detected: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
        logger.info(f"[Gaze] Using GPU device: {GPU_DEVICE_ID}")
    else:
        logger.info("[Gaze] OpenCV không có CUDA support, sử dụng CPU")
except Exception as e:
    logger.info(f"[Gaze] OpenCV GPU check failed: {str(e)}, sử dụng CPU")

# Kiểm tra GPU cho MediaPipe/PyTorch (nếu có)
if MEDIAPIPE_AVAILABLE:
    try:
        import torch
        if torch.cuda.is_available() and (USE_GPU == "auto" or USE_GPU == "true"):
            if not GPU_AVAILABLE:  # Chỉ set nếu OpenCV GPU chưa available
                GPU_AVAILABLE = True
            logger.info(f"[Gaze] ✅ PyTorch GPU available: {torch.cuda.get_device_name(0)}")
    except:
        pass

