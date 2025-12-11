"""
Gaze Tracking Module - Refactored version
"""
from .config import GazeConfig
from .gpu_utils import GPUManager
from .models import GazeAnalysisResponse
from .focus_timeline import FocusTimeline, FocusPeriod
from .gaze_estimation_3d import GazeEstimator3D
from .gaze_wandering import GazeWanderingDetector, WanderingPeriod
from .fatigue_detector import FatigueDetector
from .focus_level import FocusLevelCalculator
from .oid_detector import OIDDetector, create_oid_detector

__all__ = [
    'GazeConfig', 
    'GPUManager', 
    'GazeAnalysisResponse', 
    'FocusTimeline', 
    'FocusPeriod',
    'GazeEstimator3D',
    'GazeWanderingDetector',
    'WanderingPeriod',
    'FatigueDetector',
    'FocusLevelCalculator',
    'OIDDetector',
    'create_oid_detector'
]

