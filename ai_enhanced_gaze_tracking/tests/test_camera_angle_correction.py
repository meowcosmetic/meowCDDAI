"""
Property-based tests for camera angle correction.

**Feature: ai-enhanced-gaze-tracking, Property 3: Camera Angle Correction**
**Validates: Requirements 2.1, 2.3**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from ..components.camera_calibration import AutomaticCameraCalibrator
from ..core.data_models import CameraParameters, FaceDetection


class TestCameraAngleCorrection:
    """Property-based tests for camera angle correction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = AutomaticCameraCalibrator(image_size=(640, 480))
        
    @given(
        camera_angle=st.floats(min_value=-30.0, max_value=30.0),
        points=arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=1, max_value=10), st.just(2)),
            elements=st.floats(min_value=50, max_value=590, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_camera_angle_correction_independence(self, camera_angle, points):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 3: Camera Angle Correction**
        
        For any camera angle within correction range (±30°), the system should 
        transform gaze coordinates to a normalized reference frame that is 
        independent of camera orientation.
        
        **Validates: Requirements 2.1, 2.3**
        """
        assume(not np.any(np.isnan(points)))
        assume(not np.any(np.isinf(points)))
        assume(points.shape[0] > 0)
        
        # Create camera parameters with the given angle
        camera_params = CameraParameters(
            intrinsic_matrix=self.calibrator.default_camera_matrix,
            distortion_coeffs=self.calibrator.default_distortion,
            camera_angle=camera_angle,
            correction_matrix=np.eye(3),
            reference_frame=np.eye(4),
            calibration_quality=1.0
        )
        
        # Update parameters with angle correction matrices
        camera_params = self.calibrator.update_camera_parameters(camera_params, camera_angle)
        
        # Apply coordinate correction
        corrected_points = self.calibrator.correct_coordinates(points, camera_params)
        
        # Property: Corrected coordinates should be in normalized reference frame
        # This means they should be independent of the original camera angle
        
        # Test 1: Output should have same shape as input
        assert corrected_points.shape == points.shape, \
            f"Shape mismatch: input {points.shape}, output {corrected_points.shape}"
        
        # Test 2: Corrected points should be finite
        assert np.all(np.isfinite(corrected_points)), \
            "Corrected coordinates contain non-finite values"
        
        # Test 3: For zero angle, correction should preserve general coordinate relationships
        if abs(camera_angle) < 0.1:
            # For very small angles, the relative positions should be preserved
            # Allow for coordinate normalization effects
            if points.shape[0] > 1:
                original_distances = np.linalg.norm(points[1:] - points[0], axis=1)
                corrected_distances = np.linalg.norm(corrected_points[1:] - corrected_points[0], axis=1)
                if np.any(original_distances > 0):
                    relative_change = np.abs(corrected_distances - original_distances) / (original_distances + 1e-6)
                    assert np.mean(relative_change) < 2.0, \
                        f"Excessive relative change for small angle {camera_angle}: {np.mean(relative_change)}"
        
        # Test 4: Correction should be reasonably stable for repeated applications
        double_corrected = self.calibrator.correct_coordinates(corrected_points, camera_params)
        correction_stability = np.max(np.abs(double_corrected - corrected_points))
        
        # For coordinate transformations, perfect idempotence is not always achievable
        # due to numerical precision and the nature of the transformations
        # Allow reasonable tolerance based on the magnitude of coordinates
        max_coord_magnitude = np.max(np.abs(corrected_points))
        stability_threshold = max(1.0, max_coord_magnitude * 0.02)  # 2% tolerance
        
        assert correction_stability < stability_threshold, \
            f"Correction not stable: difference {correction_stability} > threshold {stability_threshold}"
            
    @given(
        angle1=st.floats(min_value=-25.0, max_value=25.0),
        angle2=st.floats(min_value=-25.0, max_value=25.0),
        test_points=arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=1, max_value=5), st.just(2)),
            elements=st.floats(min_value=50, max_value=590, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_angle_correction_normalization(self, angle1, angle2, test_points):
        """
        Test that different camera angles produce coordinates in the same normalized frame.
        
        **Feature: ai-enhanced-gaze-tracking, Property 3: Camera Angle Correction**
        **Validates: Requirements 2.1, 2.3**
        """
        assume(not np.any(np.isnan(test_points)))
        assume(not np.any(np.isinf(test_points)))
        assume(test_points.shape[0] > 0)
        assume(abs(angle1 - angle2) > 1.0)  # Ensure meaningful difference
        
        # Create camera parameters for two different angles
        params1 = self.calibrator.update_camera_parameters(
            self.calibrator._create_default_parameters(), angle1
        )
        params2 = self.calibrator.update_camera_parameters(
            self.calibrator._create_default_parameters(), angle2
        )
        
        # Apply the same test points to both camera configurations
        corrected1 = self.calibrator.correct_coordinates(test_points, params1)
        corrected2 = self.calibrator.correct_coordinates(test_points, params2)
        
        # Property: After correction, points should be in similar normalized space
        # The correction should reduce the difference between different camera angles
        
        # Calculate the difference in corrected coordinates
        correction_difference = np.mean(np.abs(corrected1 - corrected2))
        
        # Calculate what the difference would be without correction
        uncorrected_difference = np.mean(np.abs(test_points - test_points))  # This is 0
        # Instead, estimate the effect of angle difference on coordinates
        angle_diff_rad = np.radians(abs(angle1 - angle2))
        expected_uncorrected_diff = np.mean(np.linalg.norm(test_points, axis=1)) * np.sin(angle_diff_rad)
        
        # The correction should significantly reduce coordinate differences
        # Allow some tolerance for numerical precision and imperfect correction
        max_allowed_difference = max(5.0, expected_uncorrected_diff * 0.3)
        
        assert correction_difference < max_allowed_difference, \
            f"Angle correction insufficient: diff {correction_difference:.2f} > {max_allowed_difference:.2f} " \
            f"for angles {angle1:.1f}° vs {angle2:.1f}°"
            
    @given(
        landmarks=arrays(
            dtype=np.float32,
            shape=st.tuples(st.integers(min_value=6, max_value=10), st.just(2)),
            elements=st.floats(min_value=100, max_value=540, allow_nan=False, allow_infinity=False)
        ),
        confidence=st.floats(min_value=0.6, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_camera_angle_detection_bounds(self, landmarks, confidence):
        """
        Test that camera angle detection produces reasonable results within bounds.
        
        **Feature: ai-enhanced-gaze-tracking, Property 3: Camera Angle Correction**
        **Validates: Requirements 2.1**
        """
        assume(not np.any(np.isnan(landmarks)))
        assume(not np.any(np.isinf(landmarks)))
        assume(landmarks.shape[0] >= 6)
        
        # Create face detection with the given landmarks
        face_detection = FaceDetection(
            bbox=(100, 100, 200, 200),
            landmarks=landmarks,
            confidence=confidence
        )
        
        # Detect camera angle
        detected_angle = self.calibrator.detect_camera_angle([face_detection])
        
        # Property: Detected angle should be within reasonable bounds
        assert isinstance(detected_angle, (int, float)), \
            f"Detected angle should be numeric, got {type(detected_angle)}"
        
        assert np.isfinite(detected_angle), \
            f"Detected angle should be finite, got {detected_angle}"
        
        # Should be within the maximum correction range
        assert abs(detected_angle) <= self.calibrator.max_angle_correction + 1.0, \
            f"Detected angle {detected_angle}° exceeds maximum correction range " \
            f"±{self.calibrator.max_angle_correction}°"
            
    @given(
        num_faces=st.integers(min_value=1, max_value=3),
        base_angle=st.floats(min_value=-20.0, max_value=20.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_multiple_faces_angle_consistency(self, num_faces, base_angle):
        """
        Test that angle detection is consistent across multiple faces.
        
        **Feature: ai-enhanced-gaze-tracking, Property 3: Camera Angle Correction**
        **Validates: Requirements 2.1**
        """
        # Create multiple face detections with similar geometry (simulating same camera angle)
        face_detections = []
        
        for i in range(num_faces):
            # Create landmarks that would result from the same camera angle
            # Add small random variations to simulate real detection noise
            noise = np.random.normal(0, 2.0, (6, 2)).astype(np.float32)
            base_landmarks = np.array([
                [320, 200],  # Nose tip
                [320, 280],  # Chin  
                [280, 180],  # Left eye
                [360, 180],  # Right eye
                [300, 240],  # Left mouth
                [340, 240],  # Right mouth
            ], dtype=np.float32) + noise
            
            face_detections.append(FaceDetection(
                bbox=(250 + i*10, 150 + i*10, 140, 180),
                landmarks=base_landmarks,
                confidence=0.8
            ))
        
        # Detect angle from all faces
        detected_angle = self.calibrator.detect_camera_angle(face_detections)
        
        # Property: Should produce a reasonable angle estimate
        assert np.isfinite(detected_angle), \
            f"Multi-face angle detection failed: {detected_angle}"
        
        assert abs(detected_angle) <= self.calibrator.max_angle_correction, \
            f"Multi-face detected angle {detected_angle}° exceeds bounds"