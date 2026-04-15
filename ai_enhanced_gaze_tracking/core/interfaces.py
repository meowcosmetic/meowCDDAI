"""
Abstract base classes defining interfaces for all major components.

This module establishes clear contracts between components, enabling
flexible component swapping and dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from .data_models import (
    GazeEstimate, HeadPose, CameraParameters, FocusEvent, 
    QualityMetrics, FaceDetection
)


class FaceDetector(ABC):
    """Abstract interface for face detection components."""
    
    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the given frame.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            List of face detections with bounding boxes and landmarks
        """
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold for face detection."""
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for face detection."""
        pass


class GazeEstimator(ABC):
    """Abstract interface for gaze estimation components."""
    
    @abstractmethod
    def estimate_gaze(
        self, 
        face_detection: FaceDetection,
        frame: np.ndarray
    ) -> GazeEstimate:
        """
        Estimate gaze direction from face detection.
        
        Args:
            face_detection: Face detection with landmarks
            frame: Input video frame
            
        Returns:
            Gaze estimate with direction vector and confidence
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the gaze estimation model."""
        pass


class HeadPoseEstimator(ABC):
    """Abstract interface for 3D head pose estimation."""
    
    @abstractmethod
    def estimate_head_pose(
        self,
        landmarks: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray
    ) -> HeadPose:
        """
        Estimate 3D head pose from facial landmarks.
        
        Args:
            landmarks: 2D facial landmarks
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            
        Returns:
            Head pose with rotation angles and transformation matrix
        """
        pass
    
    @abstractmethod
    def get_3d_face_model(self) -> np.ndarray:
        """Get the 3D face model points used for pose estimation."""
        pass


class CameraCalibrator(ABC):
    """Abstract interface for camera calibration and angle correction."""
    
    @abstractmethod
    def calibrate_camera(
        self,
        reference_points: List[Tuple[np.ndarray, np.ndarray]]
    ) -> CameraParameters:
        """
        Calibrate camera parameters from reference points.
        
        Args:
            reference_points: List of (2D image points, 3D world points) pairs
            
        Returns:
            Camera parameters including intrinsic matrix and correction
        """
        pass
    
    @abstractmethod
    def detect_camera_angle(self, face_detections: List[FaceDetection]) -> float:
        """
        Detect camera angle bias from face geometry.
        
        Args:
            face_detections: List of face detections
            
        Returns:
            Camera angle in degrees from straight-on
        """
        pass
    
    @abstractmethod
    def correct_coordinates(
        self,
        points: np.ndarray,
        camera_params: CameraParameters
    ) -> np.ndarray:
        """
        Transform coordinates to normalized reference frame.
        
        Args:
            points: Input coordinates to transform
            camera_params: Camera calibration parameters
            
        Returns:
            Transformed coordinates in normalized frame
        """
        pass


class SensorFusion(ABC):
    """Abstract interface for multi-modal sensor fusion."""
    
    @abstractmethod
    def fuse_estimates(
        self,
        estimates: List[GazeEstimate],
        confidences: List[float]
    ) -> GazeEstimate:
        """
        Fuse multiple gaze estimates into single result.
        
        Args:
            estimates: List of gaze estimates from different sources
            confidences: Confidence scores for each estimate
            
        Returns:
            Fused gaze estimate with combined confidence
        """
        pass
    
    @abstractmethod
    def update_weights(self, source_reliabilities: Dict[str, float]) -> None:
        """
        Update fusion weights based on source reliability.
        
        Args:
            source_reliabilities: Reliability scores for each source
        """
        pass
    
    @abstractmethod
    def resolve_conflicts(
        self,
        estimates: List[GazeEstimate],
        temporal_history: List[GazeEstimate]
    ) -> GazeEstimate:
        """
        Resolve conflicting predictions using temporal consistency.
        
        Args:
            estimates: Conflicting gaze estimates
            temporal_history: Previous gaze estimates for context
            
        Returns:
            Resolved gaze estimate
        """
        pass


class FocusDetector(ABC):
    """Abstract interface for focus detection and attention tracking."""
    
    @abstractmethod
    def detect_focus(
        self,
        gaze_vector: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        stability_metrics: Dict[str, float]
    ) -> Optional[FocusEvent]:
        """
        Detect focus events from gaze data.
        
        Args:
            gaze_vector: 3D gaze direction vector
            tracked_objects: List of tracked objects in scene
            stability_metrics: Gaze stability measurements
            
        Returns:
            Focus event if detected, None otherwise
        """
        pass
    
    @abstractmethod
    def classify_attention_type(
        self,
        gaze_pattern: List[GazeEstimate],
        scene_objects: List[Dict[str, Any]]
    ) -> str:
        """
        Classify type of attention behavior.
        
        Args:
            gaze_pattern: Sequence of gaze estimates
            scene_objects: Objects present in scene
            
        Returns:
            Attention type: "object", "person", "wandering", "unknown"
        """
        pass
    
    @abstractmethod
    def track_attention_shifts(
        self,
        focus_events: List[FocusEvent]
    ) -> Dict[str, Any]:
        """
        Track sequences and patterns of attention shifts.
        
        Args:
            focus_events: Sequence of focus events
            
        Returns:
            Analysis of attention shift patterns
        """
        pass


class QualityAssessor(ABC):
    """Abstract interface for data quality assessment."""
    
    @abstractmethod
    def assess_quality(
        self,
        gaze_estimate: GazeEstimate,
        frame: np.ndarray,
        face_detection: FaceDetection
    ) -> QualityMetrics:
        """
        Assess quality of gaze tracking data.
        
        Args:
            gaze_estimate: Gaze estimate to assess
            frame: Video frame
            face_detection: Associated face detection
            
        Returns:
            Quality metrics and assessment
        """
        pass
    
    @abstractmethod
    def validate_data_segment(
        self,
        gaze_sequence: List[GazeEstimate],
        quality_threshold: float
    ) -> List[bool]:
        """
        Validate reliability of data segments.
        
        Args:
            gaze_sequence: Sequence of gaze estimates
            quality_threshold: Minimum quality threshold
            
        Returns:
            Boolean mask indicating reliable segments
        """
        pass
    
    @abstractmethod
    def flag_unreliable_data(
        self,
        quality_metrics: List[QualityMetrics]
    ) -> List[int]:
        """
        Flag potentially unreliable data segments.
        
        Args:
            quality_metrics: Quality assessments for data sequence
            
        Returns:
            Indices of unreliable segments
        """
        pass


class ObjectDetector(ABC):
    """Abstract interface for object detection components."""
    
    @abstractmethod
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the given frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detected objects with bounding boxes and classes
        """
        pass
    
    @abstractmethod
    def get_supported_classes(self) -> List[str]:
        """Get list of object classes this detector can identify."""
        pass


class CalibrationSystem(ABC):
    """Abstract interface for automatic calibration system."""
    
    @abstractmethod
    def auto_calibrate(
        self,
        face_characteristics: Dict[str, Any],
        reference_observations: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Automatically calibrate system parameters.
        
        Args:
            face_characteristics: Detected face characteristics
            reference_observations: Known gaze target observations
            
        Returns:
            Calibrated parameters
        """
        pass
    
    @abstractmethod
    def adapt_to_environment(
        self,
        environmental_conditions: Dict[str, Any]
    ) -> None:
        """
        Adapt parameters to environmental conditions.
        
        Args:
            environmental_conditions: Current environment state
        """
        pass