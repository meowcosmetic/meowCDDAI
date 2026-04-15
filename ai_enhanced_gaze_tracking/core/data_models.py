"""
Enhanced data models for the AI-Enhanced Gaze Tracking System.

This module defines all data structures used throughout the system,
providing type safety and clear data contracts.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum


class AttentionType(Enum):
    """Types of attention behavior."""
    OBJECT = "object"
    PERSON = "person" 
    WANDERING = "wandering"
    UNKNOWN = "unknown"


class FatigueLevel(Enum):
    """Levels of fatigue detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class FaceDetection:
    """Face detection result with landmarks and metadata."""
    bbox: Tuple[float, float, float, float]  # (x, y, width, height)
    landmarks: np.ndarray  # 2D facial landmarks
    confidence: float  # Detection confidence (0-1)
    face_id: Optional[int] = None  # Tracking ID
    is_child: bool = False  # Child vs adult classification
    quality_score: float = 1.0  # Face quality assessment


@dataclass
class HeadPose:
    """3D head pose information with full transformation data."""
    yaw: float  # Head rotation left/right (radians)
    pitch: float  # Head rotation up/down (radians)
    roll: float  # Head rotation tilt (radians)
    translation: np.ndarray  # 3D position vector
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    confidence: float  # Pose estimation confidence (0-1)
    euler_angles: Optional[Tuple[float, float, float]] = None  # Alternative representation


@dataclass
class GazeEstimate:
    """Enhanced gaze estimate with comprehensive 3D information."""
    gaze_vector_3d: np.ndarray  # 3D gaze direction in world coordinates
    gaze_point_2d: Tuple[float, float]  # 2D gaze point on screen
    confidence: float  # Overall confidence (0-1)
    head_pose: HeadPose  # Associated head pose
    timestamp: float  # Frame timestamp
    source_confidences: Dict[str, float]  # Individual source confidences
    quality_metrics: 'QualityMetrics'  # Data quality assessment
    method: str = "unknown"  # Estimation method used
    raw_features: Optional[Dict[str, Any]] = None  # Raw feature data


@dataclass
class CameraParameters:
    """Camera calibration and correction parameters."""
    intrinsic_matrix: np.ndarray  # 3x3 camera matrix
    distortion_coeffs: np.ndarray  # Distortion coefficients
    camera_angle: float  # Camera angle from straight-on (degrees)
    correction_matrix: np.ndarray  # Angle correction transformation
    reference_frame: np.ndarray  # Normalized coordinate system
    calibration_quality: float = 1.0  # Calibration quality score
    last_calibrated: Optional[float] = None  # Timestamp of last calibration


@dataclass
class FocusEvent:
    """Focus detection event with comprehensive metadata."""
    target_object_id: Optional[str]  # Object being focused on
    focus_type: AttentionType  # Type of attention
    start_time: float  # Focus start timestamp
    duration: float  # Focus duration (seconds)
    stability_score: float  # Gaze stability during focus
    confidence: float  # Focus detection confidence
    target_bbox: Optional[Tuple[float, float, float, float]] = None  # Target bounding box
    gaze_trajectory: Optional[List[Tuple[float, float]]] = None  # Gaze path during focus
    interruptions: int = 0  # Number of brief interruptions


@dataclass
class QualityMetrics:
    """Comprehensive data quality assessment."""
    overall_quality: float  # Overall quality score (0-1)
    head_pose_quality: float  # Head pose estimation quality
    lighting_quality: float  # Lighting condition assessment
    occlusion_level: float  # Face occlusion level (0-1)
    motion_blur: float  # Motion blur assessment (0-1)
    tracking_stability: float  # Temporal tracking stability
    eye_visibility: float = 1.0  # Eye region visibility (0-1)
    landmark_quality: float = 1.0  # Facial landmark quality
    temporal_consistency: float = 1.0  # Consistency with previous frames


@dataclass
class DetectedObject:
    """Enhanced object detection result."""
    class_name: str  # Object class name
    bbox: Tuple[float, float, float, float]  # Bounding box (x, y, w, h)
    confidence: float  # Detection confidence
    center: Tuple[float, float]  # Center point (x, y)
    track_id: Optional[int] = None  # Tracking ID
    gaze_confidence: Optional[float] = None  # Gaze intersection confidence
    depth_estimate: Optional[float] = None  # Estimated depth
    interaction_history: Optional[List[float]] = None  # Previous interaction timestamps


@dataclass
class WanderingPeriod:
    """Period of wandering gaze behavior."""
    start_time: float  # Start timestamp
    end_time: float  # End timestamp
    duration: float  # Duration in seconds
    stability_score: float  # Gaze stability during period
    average_position: Tuple[float, float]  # Average gaze position
    variance: float  # Gaze position variance


@dataclass
class FatigueIndicators:
    """Fatigue detection indicators."""
    blink_rate: float  # Blinks per minute
    eye_closure_duration: float  # Average eye closure time
    head_movement_frequency: float  # Head movement frequency
    gaze_stability_decline: float  # Decline in gaze stability
    attention_span_reduction: float  # Reduction in attention span


@dataclass
class SystemConfiguration:
    """System-wide configuration parameters."""
    face_detection_confidence: float = 0.5
    gaze_estimation_method: str = "multi_modal"
    head_pose_compensation: bool = True
    camera_angle_correction: bool = True
    ai_model_ensemble: bool = True
    real_time_processing: bool = True
    quality_assessment: bool = True
    focus_detection_enabled: bool = True
    wandering_detection_enabled: bool = True
    fatigue_detection_enabled: bool = True


@dataclass
class ProcessingMetrics:
    """Performance and processing metrics."""
    fps: float  # Processing frames per second
    latency_ms: float  # Processing latency in milliseconds
    memory_usage_mb: float  # Memory usage in MB
    gpu_utilization: float = 0.0  # GPU utilization percentage
    cpu_utilization: float = 0.0  # CPU utilization percentage
    dropped_frames: int = 0  # Number of dropped frames
    processing_time_breakdown: Optional[Dict[str, float]] = None  # Time per component


@dataclass
class SessionSummary:
    """Summary of a complete gaze tracking session."""
    session_id: str  # Unique session identifier
    start_time: float  # Session start timestamp
    end_time: float  # Session end timestamp
    total_frames: int  # Total frames processed
    valid_frames: int  # Frames with valid gaze data
    focus_events: List[FocusEvent]  # All focus events
    wandering_periods: List[WanderingPeriod]  # Wandering behavior periods
    overall_quality: float  # Session-wide quality score
    attention_statistics: Dict[str, float]  # Attention behavior statistics
    fatigue_assessment: Optional[Dict[str, Any]] = None  # Fatigue analysis
    processing_metrics: Optional[ProcessingMetrics] = None  # Performance metrics