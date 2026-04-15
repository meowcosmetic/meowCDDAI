"""
Custom face detector as fallback when MediaPipe fails.

This module provides a robust fallback face detection system using OpenCV's
DNN-based face detection with custom landmark estimation.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import os

# Optional dlib import
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None

from ...core.interfaces import FaceDetector
from ...core.data_models import FaceDetection


class CustomFaceDetector(FaceDetector):
    """
    Custom face detector using OpenCV DNN and dlib for landmark detection.
    
    Serves as a fallback when MediaPipe face detection fails, providing
    robust face detection with basic landmark estimation.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple = (300, 300),
        use_dlib_landmarks: bool = True
    ):
        """
        Initialize custom face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            nms_threshold: Non-maximum suppression threshold
            input_size: Input size for DNN model
            use_dlib_landmarks: Whether to use dlib for landmark detection
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_dlib_landmarks = use_dlib_landmarks
        
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenCV DNN face detector
        self.net = None
        self._load_face_detection_model()
        
        # Initialize dlib landmark predictor if available
        self.landmark_predictor = None
        if use_dlib_landmarks:
            self._load_landmark_predictor()
        
        # Face tracking state
        self.face_tracker = {}
        self.next_face_id = 0
    
    def _load_face_detection_model(self):
        """Load OpenCV DNN face detection model."""
        try:
            # Try to load pre-trained face detection model
            # In production, these would be downloaded or provided
            model_path = "models/opencv_face_detector_uint8.pb"
            config_path = "models/opencv_face_detector.pbtxt"
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                self.logger.info("Loaded OpenCV DNN face detection model")
            else:
                # Fallback to Haar cascades
                self.haar_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.logger.info("Using Haar cascade face detection as fallback")
                
        except Exception as e:
            self.logger.warning(f"Failed to load face detection model: {e}")
            # Use Haar cascade as ultimate fallback
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def _load_landmark_predictor(self):
        """Load dlib landmark predictor if available."""
        if not DLIB_AVAILABLE:
            self.logger.info("Dlib not available, using geometric landmark estimation")
            return
            
        try:
            # Try to load dlib's 68-point landmark predictor
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            
            if os.path.exists(predictor_path):
                self.landmark_predictor = dlib.shape_predictor(predictor_path)
                self.logger.info("Loaded dlib landmark predictor")
            else:
                self.logger.info("Dlib landmark predictor not available, using geometric estimation")
                
        except Exception as e:
            self.logger.warning(f"Failed to load dlib landmark predictor: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using custom detection pipeline.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            List of face detections with landmarks
        """
        if frame is None or frame.size == 0:
            return []
        
        detections = []
        
        # Try DNN-based detection first
        if self.net is not None:
            detections = self._detect_faces_dnn(frame)
        
        # Fallback to Haar cascade if DNN fails or no faces found
        if not detections and hasattr(self, 'haar_cascade'):
            detections = self._detect_faces_haar(frame)
        
        return detections
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using OpenCV DNN.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face detections
        """
        detections = []
        
        try:
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, self.input_size, [104, 117, 123]
            )
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run forward pass
            detection_results = self.net.forward()
            
            h, w = frame.shape[:2]
            
            # Process detections
            for i in range(detection_results.shape[2]):
                confidence = detection_results[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    # Get bounding box coordinates
                    x1 = int(detection_results[0, 0, i, 3] * w)
                    y1 = int(detection_results[0, 0, i, 4] * h)
                    x2 = int(detection_results[0, 0, i, 5] * w)
                    y2 = int(detection_results[0, 0, i, 6] * h)
                    
                    # Ensure valid bounding box
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        
                        # Extract landmarks
                        landmarks = self._extract_landmarks(frame, bbox)
                        
                        # Assign face ID
                        face_id = self._assign_face_id(landmarks)
                        
                        # Assess quality
                        quality_score = self._assess_face_quality(frame, bbox)
                        
                        detection = FaceDetection(
                            bbox=bbox,
                            landmarks=landmarks,
                            confidence=confidence,
                            face_id=face_id,
                            is_child=self._classify_age(bbox),
                            quality_score=quality_score
                        )
                        
                        detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"DNN face detection failed: {e}")
        
        return detections
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using Haar cascade classifier.
        
        Args:
            frame: Input frame
            
        Returns:
            List of face detections
        """
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                bbox = (x, y, w, h)
                
                # Extract landmarks
                landmarks = self._extract_landmarks(frame, bbox)
                
                # Assign face ID
                face_id = self._assign_face_id(landmarks)
                
                # Assess quality
                quality_score = self._assess_face_quality(frame, bbox)
                
                # Haar cascade doesn't provide confidence, use quality as proxy
                confidence = quality_score
                
                if confidence > self.confidence_threshold:
                    detection = FaceDetection(
                        bbox=bbox,
                        landmarks=landmarks,
                        confidence=confidence,
                        face_id=face_id,
                        is_child=self._classify_age(bbox),
                        quality_score=quality_score
                    )
                    
                    detections.append(detection)
            
        except Exception as e:
            self.logger.error(f"Haar cascade face detection failed: {e}")
        
        return detections
    
    def _extract_landmarks(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract facial landmarks from detected face.
        
        Args:
            frame: Input frame
            bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Array of 2D landmark coordinates (468 x 2 to match MediaPipe format)
        """
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Initialize landmarks array (468 points to match MediaPipe)
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        if self.landmark_predictor is not None:
            # Use dlib for 68-point landmarks
            landmarks = self._extract_dlib_landmarks(frame, bbox)
        else:
            # Use geometric estimation for basic landmarks
            landmarks = self._estimate_geometric_landmarks(bbox)
        
        return landmarks
    
    def _extract_dlib_landmarks(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract landmarks using dlib's 68-point predictor.
        
        Args:
            frame: Input frame
            bbox: Face bounding box
            
        Returns:
            Array of landmark coordinates
        """
        if not DLIB_AVAILABLE or self.landmark_predictor is None:
            # Fallback to geometric estimation
            return self._estimate_geometric_landmarks(bbox)
            
        x, y, w, h = [int(coord) for coord in bbox]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create dlib rectangle
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Predict landmarks
        shape = self.landmark_predictor(gray, rect)
        
        # Convert to numpy array
        dlib_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        
        # Map 68 dlib landmarks to 468 MediaPipe format
        landmarks = self._map_dlib_to_mediapipe(dlib_landmarks, bbox)
        
        return landmarks
    
    def _map_dlib_to_mediapipe(self, dlib_landmarks: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Map 68 dlib landmarks to 468 MediaPipe landmark format.
        
        Args:
            dlib_landmarks: 68 dlib landmarks
            bbox: Face bounding box
            
        Returns:
            468 landmarks in MediaPipe format
        """
        # Initialize MediaPipe-style landmarks
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Map key dlib landmarks to MediaPipe indices
        # This is a simplified mapping - in production, a more comprehensive mapping would be used
        
        # Key landmark mappings (dlib index -> MediaPipe index)
        landmark_mapping = {
            30: 1,    # Nose tip
            36: 33,   # Left eye left corner
            39: 133,  # Left eye right corner
            42: 362,  # Right eye left corner
            45: 263,  # Right eye right corner
            48: 61,   # Mouth left corner
            54: 291,  # Mouth right corner
            8: 152,   # Chin
        }
        
        # Map known landmarks
        for dlib_idx, mp_idx in landmark_mapping.items():
            if dlib_idx < len(dlib_landmarks):
                landmarks[mp_idx] = dlib_landmarks[dlib_idx]
        
        # Fill in remaining landmarks with interpolated values
        self._interpolate_missing_landmarks(landmarks, bbox)
        
        return landmarks
    
    def _estimate_geometric_landmarks(self, bbox: tuple) -> np.ndarray:
        """
        Estimate basic landmarks using geometric assumptions.
        
        Args:
            bbox: Face bounding box
            
        Returns:
            Estimated landmark coordinates
        """
        x, y, w, h = bbox
        
        # Initialize landmarks
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Estimate key landmarks based on typical face proportions
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Key landmarks with typical proportions
        landmarks[1] = [center_x, y + h * 0.6]  # Nose tip
        landmarks[33] = [x + w * 0.3, y + h * 0.4]  # Left eye
        landmarks[263] = [x + w * 0.7, y + h * 0.4]  # Right eye
        landmarks[61] = [x + w * 0.35, y + h * 0.75]  # Left mouth corner
        landmarks[291] = [x + w * 0.65, y + h * 0.75]  # Right mouth corner
        landmarks[152] = [center_x, y + h * 0.9]  # Chin
        
        # Fill remaining landmarks with interpolated values
        self._interpolate_missing_landmarks(landmarks, bbox)
        
        return landmarks
    
    def _interpolate_missing_landmarks(self, landmarks: np.ndarray, bbox: tuple):
        """
        Fill in missing landmarks with interpolated values.
        
        Args:
            landmarks: Landmark array to fill
            bbox: Face bounding box for reference
        """
        x, y, w, h = bbox
        
        # Get indices of non-zero landmarks
        non_zero_indices = np.where(np.any(landmarks != 0, axis=1))[0]
        
        if len(non_zero_indices) == 0:
            return
        
        # Fill missing landmarks with interpolated values
        for i in range(len(landmarks)):
            if i not in non_zero_indices:
                # Simple interpolation based on face region
                if i < 100:  # Upper face region
                    landmarks[i] = [x + w * 0.5, y + h * 0.3] + np.random.normal(0, 5, 2)
                elif i < 200:  # Middle face region
                    landmarks[i] = [x + w * 0.5, y + h * 0.5] + np.random.normal(0, 5, 2)
                elif i < 300:  # Lower face region
                    landmarks[i] = [x + w * 0.5, y + h * 0.7] + np.random.normal(0, 5, 2)
                else:  # Mouth and chin region
                    landmarks[i] = [x + w * 0.5, y + h * 0.8] + np.random.normal(0, 3, 2)
    
    def _assess_face_quality(self, frame: np.ndarray, bbox: tuple) -> float:
        """
        Assess face quality for gaze tracking suitability.
        
        Args:
            frame: Input frame
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
        
        # Basic quality assessment
        quality_factors = []
        
        # 1. Face size
        face_size = min(w, h)
        size_quality = min(1.0, face_size / 50.0)  # Minimum 50 pixels
        quality_factors.append(size_quality)
        
        # 2. Blur assessment
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur_metric = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        blur_quality = min(1.0, blur_metric / 100.0)
        quality_factors.append(blur_quality)
        
        # 3. Brightness assessment
        brightness = np.mean(gray_face)
        if 50 <= brightness <= 200:
            brightness_quality = 1.0
        else:
            brightness_quality = max(0.0, 1.0 - abs(brightness - 125) / 125.0)
        quality_factors.append(brightness_quality)
        
        return np.mean(quality_factors)
    
    def _classify_age(self, bbox: tuple) -> bool:
        """
        Simple age classification based on face size.
        
        Args:
            bbox: Face bounding box
            
        Returns:
            True if classified as child
        """
        _, _, w, h = bbox
        face_size = min(w, h)
        
        # Simple heuristic: smaller faces are more likely to be children
        return face_size < 120
    
    def _assign_face_id(self, landmarks: np.ndarray) -> int:
        """
        Assign face ID for tracking.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            Face ID
        """
        # Simple tracking based on landmark center
        current_center = np.mean(landmarks[landmarks.any(axis=1)], axis=0)
        
        # Find closest existing face
        min_distance = float('inf')
        closest_id = None
        
        for face_id, prev_center in self.face_tracker.items():
            distance = np.linalg.norm(current_center - prev_center)
            if distance < min_distance and distance < 50:
                min_distance = distance
                closest_id = face_id
        
        if closest_id is not None:
            self.face_tracker[closest_id] = current_center
            return closest_id
        else:
            new_id = self.next_face_id
            self.next_face_id += 1
            self.face_tracker[new_id] = current_center
            return new_id
    
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold."""
        self.confidence_threshold = np.clip(threshold, 0.0, 1.0)