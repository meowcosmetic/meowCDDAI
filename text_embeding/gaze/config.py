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

# Lazy import MediaPipe
try:
    import mediapipe as mp
    # Direct access to solutions to avoid AttributeError in some environments
    from mediapipe.python.solutions import face_mesh as mp_face_mesh_sol
    from mediapipe.python.solutions import drawing_utils as mp_drawing_sol
    
    mp_face_mesh = mp_face_mesh_sol
    mp_drawing = mp_drawing_sol
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(f"[Gaze] MediaPipe không available hoặc lỗi: {str(e)}")
    mp = None
    mp_face_mesh = None
    mp_drawing = None

# GPU detection
USE_GPU = Config.USE_GPU.lower() if hasattr(Config, 'USE_GPU') else "auto"
GPU_AVAILABLE = False
GPU_DEVICE_ID = 0

# Kiểm tra GPU cho OpenCV
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        GPU_DEVICE_ID = Config.GPU_DEVICE_ID if hasattr(Config, 'GPU_DEVICE_ID') else 0
        logger.info(f"[Gaze] ✅ OpenCV GPU detected: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
    else:
        logger.info("[Gaze] OpenCV không có CUDA support, sử dụng CPU")
except Exception as e:
    logger.info(f"[Gaze] OpenCV GPU check failed: {str(e)}, sử dụng CPU")

# PyTorch GPU check removed as we consolidated AI libs

