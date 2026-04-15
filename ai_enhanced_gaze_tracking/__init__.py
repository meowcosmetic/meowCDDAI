"""
AI-Enhanced Gaze Tracking System

A modular, extensible gaze tracking system that combines computer vision,
artificial intelligence, and 3D geometry for robust gaze estimation.

Key Features:
- 3D head pose compensation
- Camera angle bias correction  
- AI-powered gaze prediction
- Multi-modal sensor fusion
- Automatic calibration
- Improved focus detection
"""

from .core.interfaces import (
    FaceDetector,
    GazeEstimator, 
    HeadPoseEstimator,
    CameraCalibrator,
    SensorFusion,
    FocusDetector,
    QualityAssessor
)

from .core.data_models import (
    GazeEstimate,
    HeadPose,
    CameraParameters,
    FocusEvent,
    QualityMetrics,
    FaceDetection
)

from .core.dependency_injection import DIContainer
from .core.logging_config import setup_logging

__version__ = "2.0.0"

__all__ = [
    # Core interfaces
    'FaceDetector',
    'GazeEstimator',
    'HeadPoseEstimator', 
    'CameraCalibrator',
    'SensorFusion',
    'FocusDetector',
    'QualityAssessor',
    
    # Data models
    'GazeEstimate',
    'HeadPose',
    'CameraParameters',
    'FocusEvent',
    'QualityMetrics',
    'FaceDetection',
    
    # Core utilities
    'DIContainer',
    'setup_logging'
]