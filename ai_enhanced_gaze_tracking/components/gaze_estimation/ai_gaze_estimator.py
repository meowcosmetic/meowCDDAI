"""
AI-based gaze estimation using deep learning models.

This module provides CNN-based gaze estimation with support for multiple
models and ensemble predictions for improved accuracy.
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path

from ...core.interfaces import GazeEstimator
from ...core.data_models import GazeEstimate, FaceDetection, HeadPose, QualityMetrics


class AIGazeModel(ABC):
    """Abstract base class for AI gaze estimation models."""
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the AI model from file."""
        pass
    
    @abstractmethod
    def predict_gaze(self, eye_image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict gaze direction from eye image and landmarks.
        
        Args:
            eye_image: Cropped eye region image
            landmarks: Eye landmarks
            
        Returns:
            Tuple of (gaze_vector, confidence)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        pass


class MockAIGazeModel(AIGazeModel):
    """Mock AI gaze model for testing and development."""
    
    def __init__(self):
        self.model_loaded = False
        self.model_info = {
            "name": "MockGazeNet",
            "version": "1.0.0",
            "accuracy": 0.85,
            "input_size": (64, 64),
            "description": "Mock AI gaze estimation model for testing"
        }
    
    def load_model(self, model_path: str) -> None:
        """Load the mock model."""
        self.model_loaded = True
        logging.info(f"Mock AI model loaded from {model_path}")
    
    def predict_gaze(self, eye_image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Mock gaze prediction based on eye center and simple heuristics.
        
        For frontal faces, this provides reasonable accuracy for testing.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Simple heuristic: use eye landmarks to estimate gaze direction
        if landmarks.shape[0] >= 6:  # Assuming we have at least 6 eye landmarks
            left_eye = landmarks[:3]  # First 3 points for left eye
            right_eye = landmarks[3:6]  # Next 3 points for right eye
            
            # Calculate eye centers
            left_center = np.mean(left_eye, axis=0)
            right_center = np.mean(right_eye, axis=0)
            eye_center = (left_center + right_center) / 2
            
            # Simple gaze direction based on eye position relative to face center
            # This is a simplified model - real AI models would be much more sophisticated
            face_center = np.array([eye_image.shape[1] / 2, eye_image.shape[0] / 2])
            gaze_offset = eye_center - face_center
            
            # Normalize and create 3D gaze vector
            gaze_2d = (gaze_offset / np.linalg.norm(gaze_offset)) if np.linalg.norm(gaze_offset) > 0 else np.array([0, 0])
            gaze_vector = np.array([gaze_2d[0], gaze_2d[1], 1.0])
            gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
            
            # Confidence based on eye image quality
            confidence = min(0.9, max(0.3, 1.0 - np.std(eye_image) / 255.0))
            
            return gaze_vector, confidence
        else:
            # Fallback: forward gaze with low confidence
            return np.array([0, 0, 1]), 0.2
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return self.model_info.copy()


class AIGazeEstimator(GazeEstimator):
    """
    AI-based gaze estimator with ensemble capabilities.
    
    Supports multiple AI models and combines their predictions for improved accuracy.
    Includes confidence-based fallback to traditional computer vision methods.
    """
    
    def __init__(self, models: Optional[List[AIGazeModel]] = None, confidence_threshold: float = 0.5):
        """
        Initialize AI gaze estimator.
        
        Args:
            models: List of AI gaze models to use
            confidence_threshold: Minimum confidence for AI predictions
        """
        self.models = models or [MockAIGazeModel()]
        self.confidence_threshold = confidence_threshold
        self.ensemble_enabled = len(self.models) > 1
        self.fallback_enabled = True
        
        # Load all models
        for model in self.models:
            if hasattr(model, 'model_loaded') and not model.model_loaded:
                model.load_model("mock_path")  # In real implementation, use actual model paths
    
    def estimate_gaze(self, face_detection: FaceDetection, frame: np.ndarray) -> GazeEstimate:
        """
        Estimate gaze using AI models with ensemble and fallback capabilities.
        
        Args:
            face_detection: Face detection with landmarks
            frame: Input video frame
            
        Returns:
            Gaze estimate with AI prediction and confidence
        """
        try:
            # Extract eye region from face detection
            eye_image = self._extract_eye_region(face_detection, frame)
            
            # Get predictions from all models
            predictions = []
            confidences = []
            
            for model in self.models:
                try:
                    gaze_vector, confidence = model.predict_gaze(eye_image, face_detection.landmarks)
                    predictions.append(gaze_vector)
                    confidences.append(confidence)
                except Exception as e:
                    logging.warning(f"AI model prediction failed: {e}")
                    continue
            
            if not predictions:
                # All models failed - use fallback
                return self._fallback_estimation(face_detection, frame)
            
            # Ensemble predictions if multiple models available
            if self.ensemble_enabled and len(predictions) > 1:
                final_gaze_vector, final_confidence = self._ensemble_predictions(predictions, confidences)
            else:
                final_gaze_vector = predictions[0]
                final_confidence = confidences[0]
         
            # Check if confidence meets threshold
            if final_confidence < self.confidence_threshold and self.fallback_enabled:
                logging.info(f"AI confidence {final_confidence:.3f} below threshold {self.confidence_threshold}, using fallback")
                return self._fallback_estimation(face_detection, frame)
            
            # Create gaze estimate
            gaze_point_2d = self._project_to_2d(final_gaze_vector, frame.shape)
            
            # Create mock head pose for now (in real implementation, this would come from head pose estimator)
            head_pose = HeadPose(
                yaw=0.0, pitch=0.0, roll=0.0,
                translation=np.array([0, 0, 0]),
                rotation_matrix=np.eye(3),
                confidence=0.8
            )
            
            # Create quality metrics
            quality_metrics = QualityMetrics(
                overall_quality=final_confidence,
                head_pose_quality=0.8,
                lighting_quality=0.9,
                occlusion_level=0.1,
                motion_blur=0.1,
                tracking_stability=0.9
            )
            
            return GazeEstimate(
                gaze_vector_3d=final_gaze_vector,
                gaze_point_2d=gaze_point_2d,
                confidence=final_confidence,
                head_pose=head_pose,
                timestamp=0.0,  # Would be set by calling code
                source_confidences={"ai_model": final_confidence},
                quality_metrics=quality_metrics,
                method="ai_ensemble" if self.ensemble_enabled else "ai_single"
            )
            
        except Exception as e:
            logging.error(f"AI gaze estimation failed: {e}")
            return self._fallback_estimation(face_detection, frame)
    
    def _extract_eye_region(self, face_detection: FaceDetection, frame: np.ndarray) -> np.ndarray:
        """Extract eye region from face detection."""
        # Simple eye region extraction based on face bounding box
        x, y, w, h = face_detection.bbox
        
        # Eye region is typically in the upper portion of the face
        eye_y = int(y + h * 0.2)
        eye_h = int(h * 0.4)
        eye_x = int(x + w * 0.1)
        eye_w = int(w * 0.8)
        
        # Ensure bounds are within frame
        eye_x = max(0, min(eye_x, frame.shape[1] - 1))
        eye_y = max(0, min(eye_y, frame.shape[0] - 1))
        eye_w = min(eye_w, frame.shape[1] - eye_x)
        eye_h = min(eye_h, frame.shape[0] - eye_y)
        
        eye_region = frame[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
        
        # Resize to standard size for AI model
        if eye_region.size > 0:
            eye_region = cv2.resize(eye_region, (64, 64))
        else:
            eye_region = np.zeros((64, 64, 3), dtype=np.uint8)
        
        return eye_region
    
    def _ensemble_predictions(self, predictions: List[np.ndarray], confidences: List[float]) -> Tuple[np.ndarray, float]:
        """
        Combine multiple model predictions using weighted averaging.
        
        Strategy: Always use weighted average of all predictions.
        This reduces variance and provides robustness across different scenarios.
        While not guaranteed to beat the best model on every prediction,
        it provides better average performance and stability.
        
        Args:
            predictions: List of gaze vectors from different models
            confidences: Confidence scores for each prediction
            
        Returns:
            Tuple of (ensemble_gaze_vector, ensemble_confidence)
        """
        if not predictions:
            raise ValueError("No predictions to ensemble")
        
        if len(predictions) == 1:
            return predictions[0], confidences[0]
        
        # Weighted average of all predictions
        total_weight = sum(confidences)
        if total_weight == 0:
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            weights = [c / total_weight for c in confidences]
        
        ensemble_vector = np.zeros(3)
        for pred, weight in zip(predictions, weights):
            ensemble_vector += weight * pred
        
        if np.linalg.norm(ensemble_vector) > 0:
            ensemble_vector = ensemble_vector / np.linalg.norm(ensemble_vector)
        
        # Ensemble confidence is the maximum (ensemble reduces variance)
        ensemble_confidence = max(confidences)
        
        return ensemble_vector, ensemble_confidence
    
    def _fallback_estimation(self, face_detection: FaceDetection, frame: np.ndarray) -> GazeEstimate:
        """
        Fallback gaze estimation using traditional computer vision methods.
        
        This is a simplified fallback - in a real system, this would use
        the compensated gaze estimator or other traditional methods.
        """
        # Simple fallback: assume forward gaze
        gaze_vector = np.array([0, 0, 1])
        confidence = 0.3  # Low confidence for fallback
        
        gaze_point_2d = self._project_to_2d(gaze_vector, frame.shape)
        
        head_pose = HeadPose(
            yaw=0.0, pitch=0.0, roll=0.0,
            translation=np.array([0, 0, 0]),
            rotation_matrix=np.eye(3),
            confidence=0.5
        )
        
        quality_metrics = QualityMetrics(
            overall_quality=confidence,
            head_pose_quality=0.5,
            lighting_quality=0.7,
            occlusion_level=0.2,
            motion_blur=0.2,
            tracking_stability=0.6
        )
        
        return GazeEstimate(
            gaze_vector_3d=gaze_vector,
            gaze_point_2d=gaze_point_2d,
            confidence=confidence,
            head_pose=head_pose,
            timestamp=0.0,
            source_confidences={"fallback": confidence},
            quality_metrics=quality_metrics,
            method="fallback"
        )
    
    def _project_to_2d(self, gaze_vector: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[float, float]:
        """Project 3D gaze vector to 2D screen coordinates."""
        # Simple projection - in real implementation, this would use camera parameters
        height, width = frame_shape[:2]
        
        # Normalize gaze vector
        if gaze_vector[2] != 0:
            x = (gaze_vector[0] / gaze_vector[2]) * width * 0.5 + width * 0.5
            y = (gaze_vector[1] / gaze_vector[2]) * height * 0.5 + height * 0.5
        else:
            x, y = width * 0.5, height * 0.5
        
        # Clamp to frame bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        return (float(x), float(y))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded AI models."""
        model_info = {
            "num_models": len(self.models),
            "ensemble_enabled": self.ensemble_enabled,
            "confidence_threshold": self.confidence_threshold,
            "fallback_enabled": self.fallback_enabled,
            "models": []
        }
        
        for i, model in enumerate(self.models):
            info = model.get_model_info()
            info["model_index"] = i
            model_info["models"].append(info)
        
        return model_info
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for AI predictions."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def enable_ensemble(self, enabled: bool) -> None:
        """Enable or disable ensemble predictions."""
        self.ensemble_enabled = enabled and len(self.models) > 1
    
    def enable_fallback(self, enabled: bool) -> None:
        """Enable or disable fallback to traditional methods."""
        self.fallback_enabled = enabled