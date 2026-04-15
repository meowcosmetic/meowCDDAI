"""
Property-based tests for calibration coordinate system.

**Feature: ai-enhanced-gaze-tracking, Property 4: Calibration Coordinate System**
**Validates: Requirements 2.2**
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from ..components.camera_calibration import AutomaticCameraCalibrator
from ..core.data_models import CameraParameters


class TestCalibrationCoordinateSystem:
    """Property-based tests for calibration coordinate system functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = AutomaticCameraCalibrator(image_size=(640, 480))
        
    @given(
        num_point_pairs=st.integers(min_value=4, max_value=8),
        noise_level=st.floats(min_value=0.0, max_value=2.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_calibration_coordinate_system_properties(self, num_point_pairs, noise_level):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 4: Calibration Coordinate System**
        
        For any successful camera calibration, the established reference coordinate 
        system should satisfy standard geometric properties and account for camera orientation.
        
        **Validates: Requirements 2.2**
        """
        # Generate synthetic calibration data
        reference_points = self._generate_calibration_points(num_point_pairs, noise_level)
        
        # Perform calibration
        camera_params = self.calibrator.calibrate_camera(reference_points)
        
        # Property 1: Camera parameters should be valid
        assert camera_params is not None, "Calibration should produce valid parameters"
        
        # Property 2: Intrinsic matrix should have proper structure
        intrinsic = camera_params.intrinsic_matrix
        assert intrinsic.shape == (3, 3), f"Intrinsic matrix should be 3x3, got {intrinsic.shape}"
        
        # Check that focal lengths are positive and reasonable
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        assert fx > 0 and fy > 0, f"Focal lengths should be positive: fx={fx}, fy={fy}"
        assert fx < 2000 and fy < 2000, f"Focal lengths should be reasonable: fx={fx}, fy={fy}"
        
        # Principal point should be within image bounds
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        assert 0 <= cx <= 640, f"Principal point x should be in image: cx={cx}"
        assert 0 <= cy <= 480, f"Principal point y should be in image: cy={cy}"
        
        # Bottom row should be [0, 0, 1]
        expected_bottom_row = np.array([0, 0, 1])
        np.testing.assert_allclose(intrinsic[2, :], expected_bottom_row, atol=1e-6,
                                 err_msg="Bottom row of intrinsic matrix should be [0, 0, 1]")
        
        # Property 3: Distortion coefficients should be reasonable
        dist_coeffs = camera_params.distortion_coeffs
        assert len(dist_coeffs) >= 4, f"Should have at least 4 distortion coefficients, got {len(dist_coeffs)}"
        assert np.all(np.abs(dist_coeffs) < 2.0), f"Distortion coefficients too large: {dist_coeffs}"
        
        # Property 4: Calibration quality should be meaningful
        quality = camera_params.calibration_quality
        assert 0.0 <= quality <= 1.0, f"Quality should be in [0,1], got {quality}"
        
        # Property 5: Reference frame should be a valid transformation matrix
        ref_frame = camera_params.reference_frame
        if ref_frame.shape == (4, 4):
            # Should be a valid homogeneous transformation matrix
            # Bottom row should be [0, 0, 0, 1]
            expected_bottom = np.array([0, 0, 0, 1])
            np.testing.assert_allclose(ref_frame[3, :], expected_bottom, atol=1e-6,
                                     err_msg="Reference frame bottom row should be [0, 0, 0, 1]")
            
            # Upper-left 3x3 should be orthogonal (rotation matrix)
            rotation_part = ref_frame[:3, :3]
            should_be_identity = rotation_part @ rotation_part.T
            np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-3,
                                     err_msg="Rotation part should be orthogonal")
            
            # Determinant should be 1 (proper rotation, not reflection)
            det = np.linalg.det(rotation_part)
            assert abs(det - 1.0) < 1e-3, f"Rotation determinant should be 1, got {det}"
            
    @given(
        image_width=st.integers(min_value=320, max_value=1920),
        image_height=st.integers(min_value=240, max_value=1080)
    )
    @settings(max_examples=30, deadline=None)
    def test_default_parameters_validity(self, image_width, image_height):
        """
        Test that default camera parameters satisfy geometric properties.
        
        **Feature: ai-enhanced-gaze-tracking, Property 4: Calibration Coordinate System**
        **Validates: Requirements 2.2**
        """
        calibrator = AutomaticCameraCalibrator(image_size=(image_width, image_height))
        
        # Get default parameters
        default_params = calibrator._create_default_parameters()
        
        # Property: Default intrinsic matrix should be valid
        intrinsic = default_params.intrinsic_matrix
        
        # Focal length should be reasonable for the image size
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        expected_focal = 0.8 * image_width  # Based on implementation
        assert abs(fx - expected_focal) < 1.0, f"Default focal length incorrect: {fx} vs {expected_focal}"
        assert abs(fy - expected_focal) < 1.0, f"Default focal length incorrect: {fy} vs {expected_focal}"
        
        # Principal point should be at image center
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        expected_cx, expected_cy = image_width / 2.0, image_height / 2.0
        assert abs(cx - expected_cx) < 1.0, f"Default principal point incorrect: {cx} vs {expected_cx}"
        assert abs(cy - expected_cy) < 1.0, f"Default principal point incorrect: {cy} vs {expected_cy}"
        
        # Property: Default reference frame should be identity
        ref_frame = default_params.reference_frame
        np.testing.assert_allclose(ref_frame, np.eye(4), atol=1e-6,
                                 err_msg="Default reference frame should be identity")
                                 
    @given(
        angle=st.floats(min_value=-30.0, max_value=30.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_angle_correction_matrix_properties(self, angle):
        """
        Test that angle correction matrices satisfy geometric properties.
        
        **Feature: ai-enhanced-gaze-tracking, Property 4: Calibration Coordinate System**
        **Validates: Requirements 2.2**
        """
        # Create default parameters and update with angle
        params = self.calibrator._create_default_parameters()
        updated_params = self.calibrator.update_camera_parameters(params, angle)
        
        # Property 1: Correction matrix should be orthogonal
        correction_matrix = updated_params.correction_matrix
        assert correction_matrix.shape == (3, 3), f"Correction matrix should be 3x3"
        
        # Should be orthogonal
        should_be_identity = correction_matrix @ correction_matrix.T
        np.testing.assert_allclose(should_be_identity, np.eye(3), atol=1e-6,
                                 err_msg="Correction matrix should be orthogonal")
        
        # Determinant should be 1
        det = np.linalg.det(correction_matrix)
        assert abs(det - 1.0) < 1e-6, f"Correction matrix determinant should be 1, got {det}"
        
        # Property 2: Reference frame should be consistent with angle
        ref_frame = updated_params.reference_frame
        angle_rad = np.radians(angle)
        
        # Extract rotation part and check it matches expected angle
        rotation_2d = ref_frame[:2, :2]
        expected_rotation = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        np.testing.assert_allclose(rotation_2d, expected_rotation, atol=1e-6,
                                 err_msg=f"Reference frame rotation doesn't match angle {angle}")
        
        # Property 3: Camera angle should be stored correctly
        assert abs(updated_params.camera_angle - angle) < 1e-6, \
            f"Camera angle not stored correctly: {updated_params.camera_angle} vs {angle}"
            
    def _generate_calibration_points(self, num_pairs: int, noise_level: float):
        """Generate synthetic calibration point pairs."""
        reference_points = []
        
        for i in range(num_pairs):
            # Generate 3D world points (e.g., corners of a calibration pattern)
            world_points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0.5, 0.5, 0]
            ], dtype=np.float32) * 10  # Scale up
            
            # Add some variation for different views
            world_points[:, 2] += i * 5  # Different depths
            
            # Generate corresponding 2D image points with projection + noise
            # Simple perspective projection
            focal_length = 500
            cx, cy = 320, 240
            
            image_points = np.zeros((len(world_points), 2), dtype=np.float32)
            for j, (x, y, z) in enumerate(world_points):
                # Simple perspective projection
                z_proj = max(z + 50, 1)  # Avoid division by zero
                u = focal_length * x / z_proj + cx
                v = focal_length * y / z_proj + cy
                
                # Add noise
                u += np.random.normal(0, noise_level)
                v += np.random.normal(0, noise_level)
                
                image_points[j] = [u, v]
            
            reference_points.append((image_points, world_points))
            
        return reference_points
        
    @given(
        num_calibrations=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=20, deadline=None)
    def test_calibration_consistency(self, num_calibrations):
        """
        Test that multiple calibrations with similar data produce consistent results.
        
        **Feature: ai-enhanced-gaze-tracking, Property 4: Calibration Coordinate System**
        **Validates: Requirements 2.2**
        """
        calibration_results = []
        
        for i in range(num_calibrations):
            # Generate similar calibration data with small variations
            reference_points = self._generate_calibration_points(6, noise_level=0.5)
            camera_params = self.calibrator.calibrate_camera(reference_points)
            calibration_results.append(camera_params)
        
        # Property: Results should be reasonably consistent
        if len(calibration_results) > 1:
            # Compare focal lengths
            focal_lengths = [params.intrinsic_matrix[0, 0] for params in calibration_results]
            focal_std = np.std(focal_lengths)
            focal_mean = np.mean(focal_lengths)
            
            # Standard deviation should be small relative to mean
            if focal_mean > 0:
                relative_std = focal_std / focal_mean
                assert relative_std < 0.2, \
                    f"Focal length too inconsistent: std/mean = {relative_std}"
            
            # Compare principal points
            principal_points = [(params.intrinsic_matrix[0, 2], params.intrinsic_matrix[1, 2]) 
                              for params in calibration_results]
            cx_values = [p[0] for p in principal_points]
            cy_values = [p[1] for p in principal_points]
            
            cx_std = np.std(cx_values)
            cy_std = np.std(cy_values)
            
            assert cx_std < 20, f"Principal point x too inconsistent: std = {cx_std}"
            assert cy_std < 20, f"Principal point y too inconsistent: std = {cy_std}"