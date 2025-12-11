"""
Face Detection với Strategy Pattern - MediaPipe vs OpenCV
"""
import logging
import os
import cv2
import numpy as np
from typing import Protocol, List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FaceDetector(Protocol):
    """Protocol cho Face Detector - type safety"""
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in frame, return list of face info"""
        ...


class MediaPipeFaceDetector:
    """MediaPipe Face Detector Implementation"""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.face_mesh = None
        self._initialize()
    
    def _initialize(self):
        """Initialize MediaPipe Face Mesh"""
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("[FaceDetector] MediaPipe Face Mesh initialized")
        except ImportError:
            logger.warning("[FaceDetector] MediaPipe không có, sẽ dùng OpenCV fallback")
            self.face_mesh = None
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe"""
        if self.face_mesh is None:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            faces_info = []
            if results.multi_face_landmarks:
                h, w = frame.shape[:2]
                
                for face_landmarks in results.multi_face_landmarks:
                    # Extract face bounding box
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    # Extract key landmarks (eyes, nose, mouth)
                    landmarks = self._extract_key_landmarks(face_landmarks, w, h)
                    
                    # Estimate gaze direction
                    gaze_vector = self._estimate_gaze(face_landmarks, w, h)
                    
                    faces_info.append({
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'gaze_vector': gaze_vector,
                        'confidence': 1.0,  # MediaPipe không return confidence
                        'all_landmarks': face_landmarks
                    })
            
            return faces_info
        except Exception as e:
            logger.error(f"[FaceDetector] MediaPipe detection error: {str(e)}")
            return []
    
    def _extract_key_landmarks(self, face_landmarks, w: int, h: int) -> List[List[float]]:
        """Extract key facial landmarks"""
        # MediaPipe Face Mesh landmark indices
        LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        NOSE_TIP = 4
        MOUTH_CENTER = 13
        
        key_points = []
        landmark = face_landmarks.landmark
        
        # Left eye center
        left_eye_x = sum(landmark[i].x for i in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES)
        left_eye_y = sum(landmark[i].y for i in LEFT_EYE_INDICES) / len(LEFT_EYE_INDICES)
        key_points.append([left_eye_x * w, left_eye_y * h])
        
        # Right eye center
        right_eye_x = sum(landmark[i].x for i in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES)
        right_eye_y = sum(landmark[i].y for i in RIGHT_EYE_INDICES) / len(RIGHT_EYE_INDICES)
        key_points.append([right_eye_x * w, right_eye_y * h])
        
        # Nose tip
        key_points.append([landmark[NOSE_TIP].x * w, landmark[NOSE_TIP].y * h])
        
        # Mouth center
        key_points.append([landmark[MOUTH_CENTER].x * w, landmark[MOUTH_CENTER].y * h])
        
        return key_points
    
    def _estimate_gaze(self, face_landmarks, w: int, h: int) -> Optional[List[float]]:
        """Estimate gaze direction from face landmarks"""
        try:
            # Use iris landmarks for more accurate gaze
            LEFT_IRIS = [474, 475, 476, 477]
            RIGHT_IRIS = [469, 470, 471, 472]
            
            landmark = face_landmarks.landmark
            
            # Left iris center
            left_iris_x = sum(landmark[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)
            left_iris_y = sum(landmark[i].y for i in LEFT_IRIS) / len(LEFT_IRIS)
            
            # Right iris center
            right_iris_x = sum(landmark[i].x for i in RIGHT_IRIS) / len(RIGHT_IRIS)
            right_iris_y = sum(landmark[i].y for i in RIGHT_IRIS) / len(RIGHT_IRIS)
            
            # Eye centers
            LEFT_EYE_CENTER = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE_CENTER = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            left_eye_center_x = sum(landmark[i].x for i in LEFT_EYE_CENTER) / len(LEFT_EYE_CENTER)
            left_eye_center_y = sum(landmark[i].y for i in LEFT_EYE_CENTER) / len(LEFT_EYE_CENTER)
            
            right_eye_center_x = sum(landmark[i].x for i in RIGHT_EYE_CENTER) / len(RIGHT_EYE_CENTER)
            right_eye_center_y = sum(landmark[i].y for i in RIGHT_EYE_CENTER) / len(RIGHT_EYE_CENTER)
            
            # Calculate gaze offset
            left_gaze_x = (left_iris_x - left_eye_center_x) * w
            left_gaze_y = (left_iris_y - left_eye_center_y) * h
            
            right_gaze_x = (right_iris_x - right_eye_center_x) * w
            right_gaze_y = (right_iris_y - right_eye_center_y) * h
            
            # Average gaze
            gaze_x = (left_gaze_x + right_gaze_x) / 2
            gaze_y = (left_gaze_y + right_gaze_y) / 2
            
            # Normalize
            gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
            if gaze_magnitude > 0:
                gaze_x /= gaze_magnitude
                gaze_y /= gaze_magnitude
            
            return [gaze_x, gaze_y]
        except Exception as e:
            logger.warning(f"[FaceDetector] Gaze estimation error: {str(e)}")
            return None


class OpenCVFaceDetector:
    """OpenCV Haar Cascade Face Detector Implementation"""
    
    def __init__(self, cascade_path: Optional[str] = None, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.face_cascade = None
        self._initialize(cascade_path)
    
    def _initialize(self, cascade_path: Optional[str] = None):
        """Initialize OpenCV Haar Cascade"""
        try:
            if cascade_path and os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                # Try default path
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise ValueError("Không thể load Haar Cascade")
            
            logger.info("[FaceDetector] OpenCV Haar Cascade initialized")
        except Exception as e:
            logger.error(f"[FaceDetector] OpenCV initialization error: {str(e)}")
            self.face_cascade = None
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV"""
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            faces_info = []
            for (x, y, w, h) in faces:
                bbox = [float(x), float(y), float(w), float(h)]
                
                # Simple gaze estimation (center of face)
                face_center_x = x + w / 2
                face_center_y = y + h / 2
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                
                gaze_x = (face_center_x - frame_center_x) / frame.shape[1]
                gaze_y = (face_center_y - frame_center_y) / frame.shape[0]
                
                # Normalize
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                if gaze_magnitude > 0:
                    gaze_x /= gaze_magnitude
                    gaze_y /= gaze_magnitude
                
                faces_info.append({
                    'bbox': bbox,
                    'landmarks': None,  # OpenCV không có landmarks
                    'gaze_vector': [gaze_x, gaze_y],
                    'confidence': 1.0
                })
            
            return faces_info
        except Exception as e:
            logger.error(f"[FaceDetector] OpenCV detection error: {str(e)}")
            return []


def create_face_detector(use_mediapipe: bool = True, use_gpu: bool = False) -> FaceDetector:
    """
    Factory function để tạo Face Detector - Strategy Pattern
    """
    if use_mediapipe:
        try:
            detector = MediaPipeFaceDetector(use_gpu=use_gpu)
            if detector.face_mesh is not None:
                return detector
        except Exception as e:
            logger.warning(f"[FaceDetector] MediaPipe không available: {str(e)}, dùng OpenCV")
    
    # Fallback to OpenCV
    return OpenCVFaceDetector(use_gpu=use_gpu)

