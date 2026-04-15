"""
MediaPipe-based face detection with enhanced landmark extraction.

This module provides the primary face detection capability using MediaPipe Face Mesh,
with comprehensive landmark extraction and quality assessment.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any
import logging

# Optional MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

from ...core.interfaces import FaceDetector
from ...core.data_models import FaceDetection, QualityMetrics


class MediaPipeFaceDetector(FaceDetector):
    """
    MediaPipe-based face detector with enhanced landmark extraction.
    
    Provides high-quality face detection and 468-point landmark extraction
    using Google's MediaPipe Face Mesh solution.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum detection confidence for MediaPipe
            min_tracking_confidence: Minimum tracking confidence for MediaPipe
        """
        self.confidence_threshold = confidence_threshold
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        
        # Initialize MediaPipe Face Mesh if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            self.mp_face_mesh = None
            self.face_mesh = None
        
        # Face tracking state
        self.face_tracker = {}
        self.next_face_id = 0
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            'min_face_size': 50,  # Minimum face size in pixels
            'max_blur_threshold': 100,  # Maximum blur metric
            'min_brightness': 80,  # Minimum average brightness
            'max_brightness': 190,  # Maximum average brightness
            'max_occlusion_ratio': 0.3  # Maximum occlusion ratio
        }
        
        self.logger = logging.getLogger(__name__)
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the given frame using MediaPipe Face Mesh.
        
        Args:
            frame: Input video frame as numpy array (BGR format)
            
        Returns:
            List of face detections with landmarks and quality metrics
        """
        if frame is None or frame.size == 0:
            return []
        
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            self.logger.warning("MediaPipe not available, cannot detect faces")
            return []
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        detections = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(face_landmarks, frame.shape)
                
                # Calculate bounding box
                bbox = self._calculate_bbox(landmarks)
                
                # Assess face quality
                quality_score = self._assess_face_quality(frame, landmarks, bbox)
                
                # Skip faces below quality threshold
                if quality_score < self.confidence_threshold:
                    continue
                
                # Assign or update face ID for tracking
                face_id = self._assign_face_id(landmarks)
                
                # Create face detection
                detection = FaceDetection(
                    bbox=bbox,
                    landmarks=landmarks,
                    confidence=quality_score,
                    face_id=face_id,
                    is_child=self._classify_age(landmarks, bbox),
                    quality_score=quality_score
                )
                
                detections.append(detection)
        
        return detections
    
    def _extract_landmarks(self, face_landmarks, frame_shape) -> np.ndarray:
        """
        Extract 2D landmark coordinates from MediaPipe results.
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            Array of 2D landmark coordinates (468 x 2)
        """
        height, width = frame_shape[:2]
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        for i, landmark in enumerate(face_landmarks.landmark):
            landmarks[i] = [
                landmark.x * width,
                landmark.y * height
            ]
        
        return landmarks
    
    def _calculate_bbox(self, landmarks: np.ndarray) -> tuple:
        """
        Calculate bounding box from facial landmarks.
        
        Args:
            landmarks: Array of 2D landmark coordinates
            
        Returns:
            Bounding box as (x, y, width, height)
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding
        padding = 0.1
        width = x_max - x_min
        height = y_max - y_min
        
        x_min -= width * padding
        y_min -= height * padding
        width *= (1 + 2 * padding)
        height *= (1 + 2 * padding)
        
        return (x_min, y_min, width, height)
    
    def _assess_face_quality(
        self, 
        frame: np.ndarray, 
        landmarks: np.ndarray, 
        bbox: tuple
    ) -> float:
        """
        Assess the quality of detected face for gaze tracking.
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            bbox: Face bounding box
            
        Returns:
            Quality score between 0 and 1
        """
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Ensure bbox is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return 0.0
        
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return 0.0
        
        quality_factors = {}
        
        # 1. Face size quality
        face_size = min(w, h)
        size_quality = min(1.0, face_size / self.quality_thresholds['min_face_size'])
        quality_factors['size'] = size_quality
        
        # 2. Blur assessment using Laplacian variance
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur_metric = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        blur_quality = min(1.0, blur_metric / self.quality_thresholds['max_blur_threshold'])
        quality_factors['blur'] = blur_quality
        
        # 3. Lighting quality
        brightness = np.mean(gray_face)
        if brightness < self.quality_thresholds['min_brightness']:
            lighting_quality = brightness / self.quality_thresholds['min_brightness']
        elif brightness > self.quality_thresholds['max_brightness']:
            lighting_quality = (255 - brightness) / (255 - self.quality_thresholds['max_brightness'])
        else:
            lighting_quality = 1.0
        quality_factors['lighting'] = lighting_quality
        
        # 4. Eye visibility assessment
        eye_quality = self._assess_eye_visibility(landmarks, frame)
        quality_factors['eyes'] = eye_quality
        
        # 5. Occlusion assessment
        occlusion_quality = self._assess_occlusion(landmarks, frame)
        quality_factors['occlusion'] = occlusion_quality
        
        # Combine quality factors with weights
        weights = {
            'size': 0.2,
            'blur': 0.25,
            'lighting': 0.2,
            'eyes': 0.25,
            'occlusion': 0.1
        }
        
        overall_quality = sum(
            quality_factors[factor] * weights[factor] 
            for factor in quality_factors
        )
        
        return np.clip(overall_quality, 0.0, 1.0)
    
    def _assess_eye_visibility(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """
        Assess visibility and quality of eye regions.
        
        Args:
            landmarks: Facial landmarks
            frame: Input frame
            
        Returns:
            Eye visibility quality score (0-1)
        """
        # Key eye landmark indices for MediaPipe Face Mesh
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        eye_qualities = []
        
        for eye_indices in [left_eye_indices, right_eye_indices]:
            # Get eye landmarks
            eye_landmarks = landmarks[eye_indices]
            
            # Calculate eye bounding box
            x_min, x_max = np.min(eye_landmarks[:, 0]), np.max(eye_landmarks[:, 0])
            y_min, y_max = np.min(eye_landmarks[:, 1]), np.max(eye_landmarks[:, 1])
            
            # Ensure coordinates are within frame
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(frame.shape[1], int(x_max))
            y_max = min(frame.shape[0], int(y_max))
            
            if x_max <= x_min or y_max <= y_min:
                eye_qualities.append(0.0)
                continue
            
            # Extract eye region
            eye_region = frame[y_min:y_max, x_min:x_max]
            
            if eye_region.size == 0:
                eye_qualities.append(0.0)
                continue
            
            # Assess eye region quality
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Check for sufficient contrast (indicates open eye)
            contrast = np.std(gray_eye)
            contrast_quality = min(1.0, contrast / 30.0)  # Normalize by expected contrast
            
            # Check eye size
            eye_width = x_max - x_min
            eye_height = y_max - y_min
            size_quality = min(1.0, min(eye_width, eye_height) / 10.0)  # Minimum 10 pixels
            
            eye_quality = (contrast_quality + size_quality) / 2.0
            eye_qualities.append(eye_quality)
        
        return np.mean(eye_qualities)
    
    def _assess_occlusion(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """
        Assess face occlusion level.
        
        Args:
            landmarks: Facial landmarks
            frame: Input frame
            
        Returns:
            Occlusion quality score (1 = no occlusion, 0 = fully occluded)
        """
        # Simple occlusion assessment based on landmark visibility
        # In a more sophisticated implementation, this would use additional computer vision techniques
        
        # Check if key landmarks are within reasonable bounds
        frame_height, frame_width = frame.shape[:2]
        
        # Key landmarks for face structure
        key_indices = [1, 33, 263, 61, 291, 152]  # Nose tip, eye corners, mouth corners, chin
        
        visible_landmarks = 0
        for idx in key_indices:
            if idx < len(landmarks):
                x, y = landmarks[idx]
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    visible_landmarks += 1
        
        visibility_ratio = visible_landmarks / len(key_indices)
        
        # Simple occlusion score based on landmark visibility
        occlusion_quality = visibility_ratio
        
        return occlusion_quality
    
    def _classify_age(self, landmarks: np.ndarray, bbox: tuple) -> bool:
        """
        Simple age classification to identify children vs adults.
        
        Args:
            landmarks: Facial landmarks
            bbox: Face bounding box
            
        Returns:
            True if classified as child, False otherwise
        """
        # Simple heuristic based on face proportions
        # Children typically have larger eyes relative to face size
        
        x, y, w, h = bbox
        face_area = w * h
        
        # Calculate eye region size
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        eye_areas = []
        for eye_indices in [left_eye_indices, right_eye_indices]:
            if all(idx < len(landmarks) for idx in eye_indices):
                eye_landmarks = landmarks[eye_indices]
                eye_x_min, eye_x_max = np.min(eye_landmarks[:, 0]), np.max(eye_landmarks[:, 0])
                eye_y_min, eye_y_max = np.min(eye_landmarks[:, 1]), np.max(eye_landmarks[:, 1])
                eye_area = (eye_x_max - eye_x_min) * (eye_y_max - eye_y_min)
                eye_areas.append(eye_area)
        
        if not eye_areas:
            return False  # Default to adult if can't assess
        
        avg_eye_area = np.mean(eye_areas)
        eye_to_face_ratio = avg_eye_area / face_area
        
        # Children typically have eye-to-face ratio > 0.02
        return eye_to_face_ratio > 0.02
    
    def _assign_face_id(self, landmarks: np.ndarray) -> int:
        """
        Assign or update face ID for temporal tracking.
        
        Args:
            landmarks: Current face landmarks
            
        Returns:
            Face ID for tracking
        """
        # Simple tracking based on landmark similarity
        # In production, this would use more sophisticated tracking algorithms
        
        current_center = np.mean(landmarks, axis=0)
        
        # Find closest existing face
        min_distance = float('inf')
        closest_id = None
        
        for face_id, prev_center in self.face_tracker.items():
            distance = np.linalg.norm(current_center - prev_center)
            if distance < min_distance and distance < 50:  # 50 pixel threshold
                min_distance = distance
                closest_id = face_id
        
        if closest_id is not None:
            # Update existing face
            self.face_tracker[closest_id] = current_center
            return closest_id
        else:
            # Create new face ID
            new_id = self.next_face_id
            self.next_face_id += 1
            self.face_tracker[new_id] = current_center
            return new_id
    
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold for face detection."""
        return self.confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for face detection."""
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except (ValueError, AttributeError):
                # Already closed or not properly initialized
                pass
            self.face_mesh = None
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            # Ignore cleanup errors during destruction
            pass