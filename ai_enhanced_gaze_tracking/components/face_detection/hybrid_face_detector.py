"""
Hybrid face detector combining MediaPipe and custom detection with temporal consistency.

This module provides the main face detection interface that intelligently combines
MediaPipe Face Mesh with custom fallback detection and adds temporal tracking.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
from collections import deque
import time

from ...core.interfaces import FaceDetector
from ...core.data_models import FaceDetection, QualityMetrics
from .mediapipe_face_detector import MediaPipeFaceDetector
from .custom_face_detector import CustomFaceDetector


class HybridFaceDetector(FaceDetector):
    """
    Hybrid face detector with MediaPipe primary and custom fallback.
    
    Provides robust face detection by combining MediaPipe Face Mesh with
    custom detection fallback, plus temporal consistency tracking.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_num_faces: int = 1,
        temporal_window: int = 5,
        fallback_threshold: float = 0.3,
        stability_threshold: float = 0.8
    ):
        """
        Initialize hybrid face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            max_num_faces: Maximum number of faces to detect
            temporal_window: Number of frames for temporal consistency
            fallback_threshold: Threshold to trigger fallback detector
            stability_threshold: Minimum stability score for temporal tracking
        """
        self.confidence_threshold = confidence_threshold
        self.max_num_faces = max_num_faces
        self.temporal_window = temporal_window
        self.fallback_threshold = fallback_threshold
        self.stability_threshold = stability_threshold
        
        # Initialize detectors
        self.mediapipe_detector = MediaPipeFaceDetector(
            confidence_threshold=confidence_threshold,
            max_num_faces=max_num_faces
        )
        
        self.custom_detector = CustomFaceDetector(
            confidence_threshold=fallback_threshold
        )
        
        # Temporal tracking state
        self.detection_history = deque(maxlen=temporal_window)
        self.face_tracks = {}  # face_id -> track_info
        self.next_face_id = 0
        
        # Performance monitoring
        self.performance_stats = {
            'mediapipe_success': 0,
            'fallback_used': 0,
            'temporal_predictions': 0,
            'total_frames': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using hybrid approach with temporal consistency.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            List of face detections with enhanced temporal tracking
        """
        if frame is None or frame.size == 0:
            return []
        
        self.performance_stats['total_frames'] += 1
        current_time = time.time()
        
        # Try MediaPipe detection first
        detections = self._try_mediapipe_detection(frame)
        
        # Use fallback if MediaPipe fails or produces low-quality results
        if not detections or self._should_use_fallback(detections):
            fallback_detections = self._try_fallback_detection(frame)
            
            if fallback_detections:
                detections = fallback_detections
                self.performance_stats['fallback_used'] += 1
            else:
                # Use temporal prediction if both detectors fail
                detections = self._predict_from_temporal_history(frame, current_time)
                if detections:
                    self.performance_stats['temporal_predictions'] += 1
        else:
            self.performance_stats['mediapipe_success'] += 1
        
        # Apply temporal consistency and tracking
        detections = self._apply_temporal_consistency(detections, frame, current_time)
        
        # Update detection history
        self.detection_history.append({
            'timestamp': current_time,
            'detections': detections,
            'frame_shape': frame.shape
        })
        
        return detections
    
    def _try_mediapipe_detection(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Try MediaPipe face detection.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face detections or empty list if failed
        """
        try:
            detections = self.mediapipe_detector.detect_faces(frame)
            
            # Filter by confidence threshold
            valid_detections = [
                det for det in detections 
                if det.confidence >= self.confidence_threshold
            ]
            
            return valid_detections
            
        except Exception as e:
            self.logger.warning(f"MediaPipe detection failed: {e}")
            return []
    
    def _try_fallback_detection(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Try custom fallback detection.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face detections or empty list if failed
        """
        try:
            detections = self.custom_detector.detect_faces(frame)
            
            # Mark detections as fallback
            for detection in detections:
                detection.quality_score *= 0.8  # Reduce quality score for fallback
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"Fallback detection failed: {e}")
            return []
    
    def _should_use_fallback(self, detections: List[FaceDetection]) -> bool:
        """
        Determine if fallback detector should be used.
        
        Args:
            detections: Current detections from primary detector
            
        Returns:
            True if fallback should be used
        """
        if not detections:
            return True
        
        # Check if any detection meets quality threshold
        has_good_detection = any(
            det.confidence >= self.confidence_threshold and 
            det.quality_score >= self.stability_threshold
            for det in detections
        )
        
        return not has_good_detection
    
    def _predict_from_temporal_history(
        self, 
        frame: np.ndarray, 
        current_time: float
    ) -> List[FaceDetection]:
        """
        Predict face locations from temporal history when detection fails.
        
        Args:
            frame: Current frame
            current_time: Current timestamp
            
        Returns:
            Predicted face detections
        """
        if len(self.detection_history) < 2:
            return []
        
        predictions = []
        
        # Get recent detections
        recent_detections = list(self.detection_history)[-3:]  # Last 3 frames
        
        # For each tracked face, predict current location
        for face_id, track_info in self.face_tracks.items():
            if current_time - track_info['last_seen'] > 0.5:  # Skip old tracks
                continue
            
            # Simple linear prediction based on velocity
            predicted_detection = self._predict_face_location(
                track_info, current_time, frame.shape
            )
            
            if predicted_detection:
                predicted_detection.confidence *= 0.6  # Reduce confidence for predictions
                predictions.append(predicted_detection)
        
        return predictions
    
    def _predict_face_location(
        self, 
        track_info: Dict[str, Any], 
        current_time: float,
        frame_shape: Tuple[int, int, int]
    ) -> Optional[FaceDetection]:
        """
        Predict face location based on tracking history.
        
        Args:
            track_info: Face tracking information
            current_time: Current timestamp
            frame_shape: Current frame shape
            
        Returns:
            Predicted face detection or None
        """
        if len(track_info['positions']) < 2:
            return None
        
        # Calculate velocity from recent positions
        recent_positions = list(track_info['positions'])[-2:]
        recent_times = list(track_info['timestamps'])[-2:]
        
        if len(recent_times) < 2:
            return None
        
        dt = recent_times[-1] - recent_times[-2]
        if dt <= 0:
            return None
        
        # Calculate velocity
        pos_change = np.array(recent_positions[-1]) - np.array(recent_positions[-2])
        velocity = pos_change / dt
        
        # Predict current position
        time_since_last = current_time - recent_times[-1]
        predicted_pos = np.array(recent_positions[-1]) + velocity * time_since_last
        
        # Check if prediction is within frame bounds
        h, w = frame_shape[:2]
        if not (0 <= predicted_pos[0] < w and 0 <= predicted_pos[1] < h):
            return None
        
        # Create predicted detection
        last_detection = track_info['last_detection']
        
        # Adjust bounding box based on predicted position
        bbox_center = np.array([
            last_detection.bbox[0] + last_detection.bbox[2] / 2,
            last_detection.bbox[1] + last_detection.bbox[3] / 2
        ])
        
        position_shift = predicted_pos - bbox_center
        
        predicted_bbox = (
            last_detection.bbox[0] + position_shift[0],
            last_detection.bbox[1] + position_shift[1],
            last_detection.bbox[2],
            last_detection.bbox[3]
        )
        
        # Adjust landmarks based on position shift
        predicted_landmarks = last_detection.landmarks + position_shift
        
        predicted_detection = FaceDetection(
            bbox=predicted_bbox,
            landmarks=predicted_landmarks,
            confidence=last_detection.confidence * 0.5,  # Reduce confidence
            face_id=last_detection.face_id,
            is_child=last_detection.is_child,
            quality_score=last_detection.quality_score * 0.5
        )
        
        return predicted_detection
    
    def _apply_temporal_consistency(
        self, 
        detections: List[FaceDetection], 
        frame: np.ndarray,
        current_time: float
    ) -> List[FaceDetection]:
        """
        Apply temporal consistency and update face tracking.
        
        Args:
            detections: Current frame detections
            frame: Current frame
            current_time: Current timestamp
            
        Returns:
            Temporally consistent detections
        """
        # Update face tracks
        self._update_face_tracks(detections, current_time)
        
        # Apply temporal smoothing
        smoothed_detections = self._apply_temporal_smoothing(detections)
        
        # Validate temporal consistency
        consistent_detections = self._validate_temporal_consistency(smoothed_detections)
        
        return consistent_detections
    
    def _update_face_tracks(self, detections: List[FaceDetection], current_time: float):
        """
        Update face tracking information.
        
        Args:
            detections: Current detections
            current_time: Current timestamp
        """
        # Match detections to existing tracks
        for detection in detections:
            if detection.face_id is None:
                # Assign new face ID
                detection.face_id = self._assign_new_face_id(detection)
            
            face_id = detection.face_id
            
            # Update or create track
            if face_id not in self.face_tracks:
                self.face_tracks[face_id] = {
                    'positions': deque(maxlen=self.temporal_window),
                    'timestamps': deque(maxlen=self.temporal_window),
                    'detections': deque(maxlen=self.temporal_window),
                    'created': current_time,
                    'last_seen': current_time,
                    'last_detection': detection
                }
            
            track = self.face_tracks[face_id]
            
            # Calculate face center
            face_center = [
                detection.bbox[0] + detection.bbox[2] / 2,
                detection.bbox[1] + detection.bbox[3] / 2
            ]
            
            # Update track
            track['positions'].append(face_center)
            track['timestamps'].append(current_time)
            track['detections'].append(detection)
            track['last_seen'] = current_time
            track['last_detection'] = detection
        
        # Clean up old tracks
        self._cleanup_old_tracks(current_time)
    
    def _assign_new_face_id(self, detection: FaceDetection) -> int:
        """
        Assign new face ID or match to existing track.
        
        Args:
            detection: Face detection
            
        Returns:
            Face ID
        """
        face_center = np.array([
            detection.bbox[0] + detection.bbox[2] / 2,
            detection.bbox[1] + detection.bbox[3] / 2
        ])
        
        # Try to match to existing track
        min_distance = float('inf')
        best_match_id = None
        
        for face_id, track in self.face_tracks.items():
            if len(track['positions']) > 0:
                last_position = np.array(track['positions'][-1])
                distance = np.linalg.norm(face_center - last_position)
                
                if distance < min_distance and distance < 100:  # 100 pixel threshold
                    min_distance = distance
                    best_match_id = face_id
        
        if best_match_id is not None:
            return best_match_id
        else:
            # Create new ID
            new_id = self.next_face_id
            self.next_face_id += 1
            return new_id
    
    def _cleanup_old_tracks(self, current_time: float):
        """
        Remove old face tracks.
        
        Args:
            current_time: Current timestamp
        """
        max_age = 2.0  # 2 seconds
        
        old_tracks = [
            face_id for face_id, track in self.face_tracks.items()
            if current_time - track['last_seen'] > max_age
        ]
        
        for face_id in old_tracks:
            del self.face_tracks[face_id]
    
    def _apply_temporal_smoothing(
        self, 
        detections: List[FaceDetection]
    ) -> List[FaceDetection]:
        """
        Apply temporal smoothing to detections.
        
        Args:
            detections: Current detections
            
        Returns:
            Smoothed detections
        """
        smoothed_detections = []
        
        for detection in detections:
            if detection.face_id in self.face_tracks:
                track = self.face_tracks[detection.face_id]
                
                if len(track['detections']) >= 2:
                    # Apply smoothing
                    smoothed_detection = self._smooth_detection(detection, track)
                    smoothed_detections.append(smoothed_detection)
                else:
                    smoothed_detections.append(detection)
            else:
                smoothed_detections.append(detection)
        
        return smoothed_detections
    
    def _smooth_detection(
        self, 
        current_detection: FaceDetection, 
        track: Dict[str, Any]
    ) -> FaceDetection:
        """
        Smooth detection using temporal history.
        
        Args:
            current_detection: Current detection
            track: Face track information
            
        Returns:
            Smoothed detection
        """
        # Simple exponential smoothing
        alpha = 0.7  # Smoothing factor
        
        if len(track['detections']) == 0:
            return current_detection
        
        prev_detection = track['detections'][-1]
        
        # Smooth bounding box
        smoothed_bbox = (
            alpha * current_detection.bbox[0] + (1 - alpha) * prev_detection.bbox[0],
            alpha * current_detection.bbox[1] + (1 - alpha) * prev_detection.bbox[1],
            alpha * current_detection.bbox[2] + (1 - alpha) * prev_detection.bbox[2],
            alpha * current_detection.bbox[3] + (1 - alpha) * prev_detection.bbox[3]
        )
        
        # Smooth landmarks
        smoothed_landmarks = (
            alpha * current_detection.landmarks + 
            (1 - alpha) * prev_detection.landmarks
        )
        
        # Smooth confidence
        smoothed_confidence = (
            alpha * current_detection.confidence + 
            (1 - alpha) * prev_detection.confidence
        )
        
        # Create smoothed detection
        smoothed_detection = FaceDetection(
            bbox=smoothed_bbox,
            landmarks=smoothed_landmarks,
            confidence=smoothed_confidence,
            face_id=current_detection.face_id,
            is_child=current_detection.is_child,
            quality_score=current_detection.quality_score
        )
        
        return smoothed_detection
    
    def _validate_temporal_consistency(
        self, 
        detections: List[FaceDetection]
    ) -> List[FaceDetection]:
        """
        Validate temporal consistency of detections.
        
        Args:
            detections: Detections to validate
            
        Returns:
            Validated detections
        """
        validated_detections = []
        
        for detection in detections:
            if self._is_temporally_consistent(detection):
                validated_detections.append(detection)
            else:
                # Reduce confidence for inconsistent detections
                detection.confidence *= 0.5
                detection.quality_score *= 0.5
                validated_detections.append(detection)
        
        return validated_detections
    
    def _is_temporally_consistent(self, detection: FaceDetection) -> bool:
        """
        Check if detection is temporally consistent.
        
        Args:
            detection: Detection to check
            
        Returns:
            True if consistent
        """
        if detection.face_id not in self.face_tracks:
            return True  # New detection, assume consistent
        
        track = self.face_tracks[detection.face_id]
        
        if len(track['positions']) < 2:
            return True
        
        # Check position consistency
        current_center = np.array([
            detection.bbox[0] + detection.bbox[2] / 2,
            detection.bbox[1] + detection.bbox[3] / 2
        ])
        
        last_center = np.array(track['positions'][-1])
        distance = np.linalg.norm(current_center - last_center)
        
        # Allow reasonable movement (up to 50 pixels per frame)
        max_movement = 50
        
        return distance <= max_movement
    
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold."""
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
        self.mediapipe_detector.set_confidence_threshold(threshold)
        self.custom_detector.set_confidence_threshold(max(0.1, threshold - 0.2))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.performance_stats['total_frames'] > 0:
            return {
                'mediapipe_success_rate': self.performance_stats['mediapipe_success'] / self.performance_stats['total_frames'],
                'fallback_usage_rate': self.performance_stats['fallback_used'] / self.performance_stats['total_frames'],
                'temporal_prediction_rate': self.performance_stats['temporal_predictions'] / self.performance_stats['total_frames'],
                'total_frames': self.performance_stats['total_frames'],
                'active_tracks': len(self.face_tracks)
            }
        return self.performance_stats
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.mediapipe_detector.cleanup()
        except:
            pass
        
        if hasattr(self, 'face_tracks'):
            self.face_tracks.clear()
        if hasattr(self, 'detection_history'):
            self.detection_history.clear()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            # Ignore cleanup errors during destruction
            pass