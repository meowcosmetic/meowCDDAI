"""
Enhanced PnP-based head pose estimator implementation.

This module provides robust 3D head pose estimation using the Perspective-n-Point (PnP)
algorithm with an enhanced 3D face model, validation, filtering, and temporal smoothing.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Deque
from collections import deque
from ...core.interfaces import HeadPoseEstimator
from ...core.data_models import HeadPose
from ...core.logging_config import get_logger, log_performance


class PnPHeadPoseEstimator(HeadPoseEstimator):
    """
    Enhanced head pose estimator using PnP solver with robust 3D face model.
    
    Features:
    - Robust 3D face model with multiple landmark sets
    - Head pose validation and filtering
    - Temporal smoothing for stability
    - Transformation matrix calculations
    - Confidence scoring based on reprojection error
    """
    
    def __init__(self, temporal_window_size: int = 5, smoothing_factor: float = 0.3):
        self.logger = get_logger('head_pose_estimation')
        
        # Enhanced 3D face model points (in mm, centered at nose tip)
        # Using a more comprehensive set of landmarks for robustness
        self._face_model_3d = np.array([
            [0.0, 0.0, 0.0],          # Nose tip (landmark 1)
            [0.0, -330.0, -65.0],     # Chin (landmark 152)
            [-225.0, 170.0, -135.0],  # Left eye left corner (landmark 33)
            [225.0, 170.0, -135.0],   # Right eye right corner (landmark 263)
            [-150.0, -150.0, -125.0], # Left mouth corner (landmark 61)
            [150.0, -150.0, -125.0],  # Right mouth corner (landmark 291)
            [-75.0, 130.0, -110.0],   # Left eye center (landmark 159)
            [75.0, 130.0, -110.0],    # Right eye center (landmark 386)
            [0.0, -100.0, -120.0],    # Upper lip center (landmark 13)
            [0.0, -180.0, -100.0],    # Lower lip center (landmark 14)
        ], dtype=np.float32)
        
        # Corresponding 2D landmark indices (MediaPipe face mesh)
        self._landmark_indices = [1, 152, 33, 263, 61, 291, 159, 386, 13, 14]
        
        # Temporal smoothing parameters
        self.temporal_window_size = temporal_window_size
        self.smoothing_factor = smoothing_factor
        self._pose_history: Deque[HeadPose] = deque(maxlen=temporal_window_size)
        
        # Validation thresholds
        self.max_reprojection_error = 15.0  # pixels
        self.min_confidence_threshold = 0.3
        self.max_angle_change_per_frame = np.radians(30)  # 30 degrees per frame
    
    @log_performance("head_pose_estimation")
    def estimate_head_pose(
        self,
        landmarks: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray
    ) -> HeadPose:
        """
        Estimate 3D head pose from facial landmarks with validation and filtering.
        
        Args:
            landmarks: 2D facial landmarks (N x 2)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            distortion_coeffs: Camera distortion coefficients
            
        Returns:
            Head pose with rotation angles and transformation matrix
        """
        try:
            # Extract key landmarks for pose estimation
            if landmarks.shape[0] < max(self._landmark_indices):
                raise ValueError(f"Insufficient landmarks: got {landmarks.shape[0]}, need at least {max(self._landmark_indices) + 1}")
            
            image_points = landmarks[self._landmark_indices].astype(np.float32)
            
            # Validate landmark quality
            if not self._validate_landmarks(image_points):
                self.logger.warning("Landmark validation failed, using fallback estimation")
                return self._fallback_estimation()
            
            # Solve PnP with multiple methods for robustness
            pose_candidates = []
            
            # Method 1: ITERATIVE (most accurate)
            success1, rvec1, tvec1 = cv2.solvePnP(
                self._face_model_3d,
                image_points,
                camera_matrix,
                distortion_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success1:
                pose_candidates.append((rvec1, tvec1, "ITERATIVE"))
            
            # Method 2: EPNP (faster, good for initialization)
            success2, rvec2, tvec2 = cv2.solvePnP(
                self._face_model_3d,
                image_points,
                camera_matrix,
                distortion_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
            if success2:
                pose_candidates.append((rvec2, tvec2, "EPNP"))
            
            if not pose_candidates:
                self.logger.warning("All PnP methods failed, returning fallback pose")
                return self._fallback_estimation()
            
            # Select best pose candidate based on reprojection error
            best_pose = self._select_best_pose(
                pose_candidates, image_points, camera_matrix, distortion_coeffs
            )
            
            if best_pose is None:
                return self._fallback_estimation()
            
            rvec, tvec, method = best_pose
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Extract Euler angles (yaw, pitch, roll)
            yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Calculate confidence based on reprojection error and validation
            confidence = self._calculate_confidence(
                image_points, rvec, tvec, camera_matrix, distortion_coeffs
            )
            
            # Create head pose
            head_pose = HeadPose(
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                translation=tvec.flatten(),
                rotation_matrix=rotation_matrix,
                confidence=confidence,
                euler_angles=(yaw, pitch, roll)
            )
            
            # Apply temporal smoothing
            smoothed_pose = self._apply_temporal_smoothing(head_pose)
            
            # Validate pose against physical constraints
            if self._validate_head_pose(smoothed_pose):
                self._pose_history.append(smoothed_pose)
                return smoothed_pose
            else:
                self.logger.warning("Head pose validation failed, using previous pose")
                return self._get_last_valid_pose()
            
        except Exception as e:
            self.logger.error(f"Head pose estimation failed: {e}")
            return self._fallback_estimation()
    
    def get_3d_face_model(self) -> np.ndarray:
        """Get the enhanced 3D face model points used for pose estimation."""
        return self._face_model_3d.copy()
    
    def get_transformation_matrix(self, head_pose: HeadPose) -> np.ndarray:
        """
        Get 4x4 transformation matrix for coordinate conversion.
        
        Args:
            head_pose: Head pose to convert
            
        Returns:
            4x4 transformation matrix
        """
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = head_pose.rotation_matrix
        transformation_matrix[:3, 3] = head_pose.translation
        return transformation_matrix
    
    def transform_points_to_head_frame(
        self, 
        points_3d: np.ndarray, 
        head_pose: HeadPose
    ) -> np.ndarray:
        """
        Transform 3D points from world coordinates to head-centered frame.
        
        Args:
            points_3d: 3D points in world coordinates (N x 3)
            head_pose: Head pose for transformation
            
        Returns:
            Transformed points in head-centered frame
        """
        # Get inverse transformation matrix
        transformation_matrix = self.get_transformation_matrix(head_pose)
        inverse_transform = np.linalg.inv(transformation_matrix)
        
        # Convert points to homogeneous coordinates
        if points_3d.shape[1] == 3:
            homogeneous_points = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        else:
            homogeneous_points = points_3d
        
        # Apply transformation
        transformed_points = (inverse_transform @ homogeneous_points.T).T
        
        return transformed_points[:, :3]
    
    def _validate_landmarks(self, landmarks: np.ndarray) -> bool:
        """
        Validate landmark quality for pose estimation.
        
        Args:
            landmarks: 2D landmarks to validate
            
        Returns:
            True if landmarks are valid for pose estimation
        """
        # Check for NaN or infinite values
        if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
            return False
        
        # Check if landmarks form a reasonable face shape
        # Calculate inter-ocular distance
        left_eye = landmarks[2]  # Left eye corner
        right_eye = landmarks[3]  # Right eye corner
        inter_ocular_distance = np.linalg.norm(left_eye - right_eye)
        
        # Reasonable inter-ocular distance (15-300 pixels) - more lenient
        if inter_ocular_distance < 15 or inter_ocular_distance > 300:
            return False
        
        # Check face aspect ratio
        nose = landmarks[0]
        chin = landmarks[1]
        face_height = np.linalg.norm(nose - chin)
        
        # Face height should be reasonable relative to eye distance - more lenient
        if face_height < inter_ocular_distance * 0.3 or face_height > inter_ocular_distance * 4.0:
            return False
        
        # Check that landmarks are within reasonable image bounds
        if np.any(landmarks < -100) or np.any(landmarks > 1000):  # Reasonable image size bounds
            return False
        
        return True
    
    def _select_best_pose(
        self,
        pose_candidates: List[Tuple],
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray
    ) -> Optional[Tuple]:
        """
        Select the best pose candidate based on reprojection error.
        
        Args:
            pose_candidates: List of (rvec, tvec, method) tuples
            image_points: Original 2D points
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            
        Returns:
            Best pose candidate or None if all fail validation
        """
        best_pose = None
        best_error = float('inf')
        
        for rvec, tvec, method in pose_candidates:
            try:
                # Calculate reprojection error
                projected_points, _ = cv2.projectPoints(
                    self._face_model_3d, rvec, tvec, camera_matrix, distortion_coeffs
                )
                projected_points = projected_points.reshape(-1, 2)
                
                errors = np.linalg.norm(image_points - projected_points, axis=1)
                mean_error = np.mean(errors)
                
                # Check if error is acceptable
                if mean_error < self.max_reprojection_error and mean_error < best_error:
                    best_error = mean_error
                    best_pose = (rvec, tvec, method)
                    
            except Exception as e:
                self.logger.debug(f"Error evaluating pose candidate {method}: {e}")
                continue
        
        return best_pose
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (yaw, pitch, roll).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tuple of (yaw, pitch, roll) in radians
        """
        # Extract Euler angles from rotation matrix
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        return yaw, pitch, roll
    
    def _calculate_confidence(
        self,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        distortion_coeffs: np.ndarray
    ) -> float:
        """
        Calculate confidence based on reprojection error.
        
        Args:
            image_points: Original 2D points
            rvec: Rotation vector from PnP
            tvec: Translation vector from PnP
            camera_matrix: Camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Project 3D points back to 2D
            projected_points, _ = cv2.projectPoints(
                self._face_model_3d, rvec, tvec, camera_matrix, distortion_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            
            # Calculate reprojection error
            errors = np.linalg.norm(image_points - projected_points, axis=1)
            mean_error = np.mean(errors)
            
            # Convert error to confidence (lower error = higher confidence)
            # Assume error of 10 pixels = 0.5 confidence, error of 0 = 1.0 confidence
            confidence = max(0.0, min(1.0, 1.0 - mean_error / 20.0))
            
            return confidence
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    def _apply_temporal_smoothing(self, current_pose: HeadPose) -> HeadPose:
        """
        Apply temporal smoothing to reduce jitter in head pose estimation.
        
        Args:
            current_pose: Current frame head pose
            
        Returns:
            Temporally smoothed head pose
        """
        if not self._pose_history:
            return current_pose
        
        # Get the most recent pose for comparison
        previous_pose = self._pose_history[-1]
        
        # Calculate angle differences
        yaw_diff = abs(current_pose.yaw - previous_pose.yaw)
        pitch_diff = abs(current_pose.pitch - previous_pose.pitch)
        roll_diff = abs(current_pose.roll - previous_pose.roll)
        
        # If change is too large, it might be noise - apply stronger smoothing
        if (yaw_diff > self.max_angle_change_per_frame or 
            pitch_diff > self.max_angle_change_per_frame or 
            roll_diff > self.max_angle_change_per_frame):
            smoothing_factor = self.smoothing_factor * 0.5  # Stronger smoothing
        else:
            smoothing_factor = self.smoothing_factor
        
        # Apply exponential smoothing
        smoothed_yaw = (1 - smoothing_factor) * previous_pose.yaw + smoothing_factor * current_pose.yaw
        smoothed_pitch = (1 - smoothing_factor) * previous_pose.pitch + smoothing_factor * current_pose.pitch
        smoothed_roll = (1 - smoothing_factor) * previous_pose.roll + smoothing_factor * current_pose.roll
        
        # Smooth translation
        smoothed_translation = (
            (1 - smoothing_factor) * previous_pose.translation + 
            smoothing_factor * current_pose.translation
        )
        
        # Reconstruct rotation matrix from smoothed angles
        smoothed_rotation_matrix = self._euler_to_rotation_matrix(
            smoothed_yaw, smoothed_pitch, smoothed_roll
        )
        
        # Smooth confidence
        smoothed_confidence = (
            (1 - smoothing_factor) * previous_pose.confidence + 
            smoothing_factor * current_pose.confidence
        )
        
        return HeadPose(
            yaw=smoothed_yaw,
            pitch=smoothed_pitch,
            roll=smoothed_roll,
            translation=smoothed_translation,
            rotation_matrix=smoothed_rotation_matrix,
            confidence=smoothed_confidence,
            euler_angles=(smoothed_yaw, smoothed_pitch, smoothed_roll)
        )
    
    def _validate_head_pose(self, head_pose: HeadPose) -> bool:
        """
        Validate head pose against physical constraints.
        
        Args:
            head_pose: Head pose to validate
            
        Returns:
            True if pose is physically plausible
        """
        # Check angle limits (in radians)
        max_yaw = np.radians(90)    # ±90 degrees
        max_pitch = np.radians(60)  # ±60 degrees  
        max_roll = np.radians(45)   # ±45 degrees
        
        if (abs(head_pose.yaw) > max_yaw or 
            abs(head_pose.pitch) > max_pitch or 
            abs(head_pose.roll) > max_roll):
            return False
        
        # Check translation limits (reasonable distance from camera)
        distance = np.linalg.norm(head_pose.translation)
        if distance < 200 or distance > 2000:  # 20cm to 2m
            return False
        
        # Check confidence threshold
        if head_pose.confidence < self.min_confidence_threshold:
            return False
        
        # Check rotation matrix validity
        if not self._is_valid_rotation_matrix(head_pose.rotation_matrix):
            return False
        
        return True
    
    def _is_valid_rotation_matrix(self, R: np.ndarray) -> bool:
        """
        Check if matrix is a valid rotation matrix.
        
        Args:
            R: 3x3 matrix to check
            
        Returns:
            True if valid rotation matrix
        """
        # Check if matrix is 3x3
        if R.shape != (3, 3):
            return False
        
        # Check if determinant is 1 (proper rotation)
        det = np.linalg.det(R)
        if abs(det - 1.0) > 1e-3:
            return False
        
        # Check if R * R^T = I (orthogonal matrix)
        should_be_identity = np.dot(R, R.T)
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=1e-3):
            return False
        
        return True
    
    def _euler_to_rotation_matrix(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix.
        
        Args:
            yaw: Yaw angle in radians
            pitch: Pitch angle in radians
            roll: Roll angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        
        rotation_matrix = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])
        
        return rotation_matrix
    
    def _get_last_valid_pose(self) -> HeadPose:
        """
        Get the last valid pose from history.
        
        Returns:
            Last valid head pose or fallback pose
        """
        if self._pose_history:
            return self._pose_history[-1]
        else:
            return self._fallback_estimation()
    
    def _fallback_estimation(self) -> HeadPose:
        """
        Return fallback head pose when estimation fails.
        
        Returns:
            Default head pose with low confidence
        """
        return HeadPose(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            translation=np.array([0.0, 0.0, 500.0]),  # Default distance
            rotation_matrix=np.eye(3),
            confidence=0.1,  # Low confidence for fallback
            euler_angles=(0.0, 0.0, 0.0)
        )