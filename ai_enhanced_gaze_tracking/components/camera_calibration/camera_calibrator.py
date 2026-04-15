"""
Camera calibration and angle correction system.

This module implements automatic camera angle detection, parameter estimation,
and coordinate system normalization for robust gaze tracking.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

from ...core.interfaces import CameraCalibrator
from ...core.data_models import CameraParameters, FaceDetection


logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """A calibration reference point with 2D and 3D coordinates."""
    image_point: np.ndarray  # 2D point in image coordinates
    world_point: np.ndarray  # 3D point in world coordinates
    confidence: float = 1.0  # Confidence in this calibration point


class AutomaticCameraCalibrator(CameraCalibrator):
    """
    Automatic camera calibration system with angle correction.
    
    This implementation provides:
    - Automatic camera angle detection using face geometry
    - Camera parameter estimation and validation
    - Coordinate system normalization and correction matrices
    - Dynamic recalibration for camera position changes
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (640, 480),
                 max_angle_correction: float = 30.0):
        """
        Initialize the camera calibrator.
        
        Args:
            image_size: Expected image dimensions (width, height)
            max_angle_correction: Maximum camera angle that can be corrected (degrees)
        """
        self.image_size = image_size
        self.max_angle_correction = max_angle_correction
        self.calibration_history: List[CameraParameters] = []
        self.reference_face_model = self._create_reference_face_model()
        
        # Default camera parameters (will be refined during calibration)
        self.default_camera_matrix = self._estimate_default_camera_matrix()
        self.default_distortion = np.zeros(5)
        
    def _create_reference_face_model(self) -> np.ndarray:
        """Create a reference 3D face model for angle detection."""
        # Standard 3D face model points (in mm, centered at origin)
        # These correspond to key facial landmarks
        face_model = np.array([
            [0.0, 0.0, 0.0],      # Nose tip
            [0.0, -30.0, -10.0],  # Chin
            [-30.0, 20.0, -15.0], # Left eye corner
            [30.0, 20.0, -15.0],  # Right eye corner
            [-20.0, -10.0, -5.0], # Left mouth corner
            [20.0, -10.0, -5.0],  # Right mouth corner
        ], dtype=np.float32)
        return face_model
        
    def _estimate_default_camera_matrix(self) -> np.ndarray:
        """Estimate default camera intrinsic matrix based on image size."""
        width, height = self.image_size
        
        # Typical focal length is about 0.8 * image_width for webcams
        focal_length = 0.8 * width
        
        # Principal point is typically at image center
        cx = width / 2.0
        cy = height / 2.0
        
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return camera_matrix
        
    def calibrate_camera(self, 
                        reference_points: List[Tuple[np.ndarray, np.ndarray]]) -> CameraParameters:
        """
        Calibrate camera parameters from reference points.
        
        Args:
            reference_points: List of (2D image points, 3D world points) pairs
            
        Returns:
            Camera parameters including intrinsic matrix and correction
        """
        if len(reference_points) < 4:
            logger.warning("Insufficient reference points for calibration, using defaults")
            return self._create_default_parameters()
            
        try:
            # Separate 2D and 3D points
            image_points = []
            world_points = []
            
            for img_pts, world_pts in reference_points:
                if len(img_pts) >= 4 and len(world_pts) >= 4:
                    image_points.append(img_pts.astype(np.float32))
                    world_points.append(world_pts.astype(np.float32))
            
            if len(image_points) == 0:
                logger.warning("No valid point pairs for calibration")
                return self._create_default_parameters()
                
            # Perform camera calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                world_points, image_points, self.image_size, None, None
            )
            
            if not ret:
                logger.warning("Camera calibration failed, using defaults")
                return self._create_default_parameters()
                
            # Calculate calibration quality
            total_error = 0
            total_points = 0
            
            for i in range(len(world_points)):
                projected_points, _ = cv2.projectPoints(
                    world_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)
                total_error += error
                total_points += len(world_points[i])
                
            mean_error = total_error / total_points if total_points > 0 else float('inf')
            calibration_quality = max(0.0, 1.0 - mean_error / 10.0)  # Normalize error
            
            # Create camera parameters
            camera_params = CameraParameters(
                intrinsic_matrix=camera_matrix,
                distortion_coeffs=dist_coeffs,
                camera_angle=0.0,  # Will be updated by angle detection
                correction_matrix=np.eye(3),  # Will be updated
                reference_frame=np.eye(4),  # Will be updated
                calibration_quality=calibration_quality,
                last_calibrated=cv2.getTickCount() / cv2.getTickFrequency()
            )
            
            self.calibration_history.append(camera_params)
            logger.info(f"Camera calibration completed with quality: {calibration_quality:.3f}")
            
            return camera_params
            
        except Exception as e:
            logger.error(f"Camera calibration failed: {e}")
            return self._create_default_parameters()
            
    def detect_camera_angle(self, face_detections: List[FaceDetection]) -> float:
        """
        Detect camera angle bias from face geometry.
        
        Args:
            face_detections: List of face detections
            
        Returns:
            Camera angle in degrees from straight-on
        """
        if not face_detections:
            return 0.0
            
        angles = []
        
        for face in face_detections:
            if face.confidence < 0.5 or len(face.landmarks) < 6:
                continue
                
            try:
                # Extract key landmark points (assuming MediaPipe format)
                landmarks_2d = face.landmarks[:6]  # Use first 6 landmarks
                
                # Use PnP to estimate head pose
                camera_matrix = (self.calibration_history[-1].intrinsic_matrix 
                               if self.calibration_history 
                               else self.default_camera_matrix)
                
                dist_coeffs = (self.calibration_history[-1].distortion_coeffs 
                             if self.calibration_history 
                             else self.default_distortion)
                
                success, rvec, tvec = cv2.solvePnP(
                    self.reference_face_model,
                    landmarks_2d,
                    camera_matrix,
                    dist_coeffs
                )
                
                if success:
                    # Convert rotation vector to angles
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    
                    # Extract camera angle from rotation
                    # Camera angle is primarily the yaw component
                    yaw = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
                    angle_degrees = np.degrees(yaw)
                    
                    # Limit to reasonable range
                    if abs(angle_degrees) <= self.max_angle_correction:
                        angles.append(angle_degrees)
                        
            except Exception as e:
                logger.debug(f"Failed to estimate angle for face: {e}")
                continue
                
        if angles:
            # Return median angle to reduce outlier influence
            return float(np.median(angles))
        else:
            return 0.0
            
    def correct_coordinates(self, 
                          points: np.ndarray, 
                          camera_params: CameraParameters) -> np.ndarray:
        """
        Transform coordinates to normalized reference frame.
        
        Args:
            points: Input coordinates to transform (Nx2 or Nx3)
            camera_params: Camera calibration parameters
            
        Returns:
            Transformed coordinates in normalized frame
        """
        if points.size == 0:
            return points
            
        try:
            # Ensure points are in the right format
            points = np.array(points, dtype=np.float32)
            
            if points.ndim == 1:
                points = points.reshape(1, -1)
                
            # Check if points are already in normalized coordinates
            # Normalized coordinates are typically in range [-1, 1] or similar small range
            if points.shape[1] == 2:
                max_coord = np.max(np.abs(points))
                if max_coord < 2.0:  # Already normalized
                    return points.copy()
                
            # Apply camera angle correction first
            if abs(camera_params.camera_angle) > 0.1:  # Only correct if significant angle
                angle_corrected_points = self._apply_angle_correction(points, camera_params)
            else:
                angle_corrected_points = points.copy()
                
            # Apply coordinate normalization only for 2D points
            if points.shape[1] == 2:
                # Only apply undistortion if we have meaningful distortion coefficients
                if np.any(np.abs(camera_params.distortion_coeffs) > 1e-6):
                    normalized_points = cv2.undistortPoints(
                        angle_corrected_points.reshape(-1, 1, 2),
                        camera_params.intrinsic_matrix,
                        camera_params.distortion_coeffs
                    ).reshape(-1, 2)
                else:
                    # Just normalize using camera matrix without distortion correction
                    normalized_points = self._normalize_without_distortion(
                        angle_corrected_points, camera_params.intrinsic_matrix
                    )
            else:
                # For 3D points, apply reference frame transformation
                normalized_points = self._normalize_3d_coordinates(
                    angle_corrected_points, camera_params
                )
            
            return normalized_points
            
        except Exception as e:
            logger.error(f"Coordinate correction failed: {e}")
            return points
            
    def _normalize_without_distortion(self, points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Normalize 2D points using camera matrix without distortion correction."""
        # Convert to normalized coordinates manually
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        normalized = np.zeros_like(points)
        normalized[:, 0] = (points[:, 0] - cx) / fx
        normalized[:, 1] = (points[:, 1] - cy) / fy
        
        return normalized
        
    def _normalize_3d_coordinates(self, points: np.ndarray, camera_params: CameraParameters) -> np.ndarray:
        """Normalize 3D coordinates using reference frame transformation."""
        if camera_params.reference_frame.shape == (4, 4):
            # Convert to homogeneous coordinates
            homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
            # Apply transformation
            transformed = homogeneous @ camera_params.reference_frame.T
            return transformed[:, :3]  # Convert back to 3D
        else:
            return points
            
    def _apply_angle_correction(self, 
                              points: np.ndarray, 
                              camera_params: CameraParameters) -> np.ndarray:
        """Apply camera angle correction to points."""
        angle_rad = np.radians(camera_params.camera_angle)
        
        # Create rotation matrix for angle correction
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        if points.shape[1] == 2:
            # 2D rotation
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            corrected = points @ rotation_matrix.T
        else:
            # 3D rotation (around Y-axis for camera angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            corrected = points @ rotation_matrix.T
            
        return corrected
        
    def _normalize_coordinates(self, 
                             points: np.ndarray, 
                             camera_params: CameraParameters) -> np.ndarray:
        """Normalize coordinates to reference frame (legacy method)."""
        # This method is kept for backward compatibility but not used in the main flow
        return self._normalize_without_distortion(points, camera_params.intrinsic_matrix)
        
    def _create_default_parameters(self) -> CameraParameters:
        """Create default camera parameters when calibration fails."""
        return CameraParameters(
            intrinsic_matrix=self.default_camera_matrix,
            distortion_coeffs=self.default_distortion,
            camera_angle=0.0,
            correction_matrix=np.eye(3),
            reference_frame=np.eye(4),
            calibration_quality=0.5,  # Medium quality for defaults
            last_calibrated=cv2.getTickCount() / cv2.getTickFrequency()
        )
        
    def update_camera_parameters(self, 
                               camera_params: CameraParameters, 
                               detected_angle: float) -> CameraParameters:
        """
        Update camera parameters with detected angle and correction matrices.
        
        Args:
            camera_params: Current camera parameters
            detected_angle: Detected camera angle in degrees
            
        Returns:
            Updated camera parameters with angle correction
        """
        # Update camera angle
        camera_params.camera_angle = detected_angle
        
        # Create angle correction matrix
        angle_rad = np.radians(detected_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        camera_params.correction_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Update reference frame (4x4 transformation matrix)
        camera_params.reference_frame = np.array([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        return camera_params
        
    def needs_recalibration(self, 
                          current_detections: List[FaceDetection],
                          threshold_angle: float = 5.0) -> bool:
        """
        Check if recalibration is needed based on angle changes.
        
        Args:
            current_detections: Current face detections
            threshold_angle: Angle change threshold for recalibration
            
        Returns:
            True if recalibration is recommended
        """
        if not self.calibration_history:
            return True
            
        current_angle = self.detect_camera_angle(current_detections)
        last_params = self.calibration_history[-1]
        
        angle_change = abs(current_angle - last_params.camera_angle)
        
        return angle_change > threshold_angle
        
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status and quality metrics."""
        if not self.calibration_history:
            return {
                "calibrated": False,
                "quality": 0.0,
                "angle": 0.0,
                "last_calibrated": None
            }
            
        last_params = self.calibration_history[-1]
        
        return {
            "calibrated": True,
            "quality": last_params.calibration_quality,
            "angle": last_params.camera_angle,
            "last_calibrated": last_params.last_calibrated,
            "correction_available": abs(last_params.camera_angle) > 0.1
        }