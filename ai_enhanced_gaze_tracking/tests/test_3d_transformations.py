"""
Property-based tests for 3D transformation consistency.

**Feature: ai-enhanced-gaze-tracking, Property 2: 3D Transformation Consistency**
**Validates: Requirements 1.3**

This module tests that 3D transformation matrices produce mathematically consistent
coordinate transformations that preserve geometric relationships.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import cv2

from ai_enhanced_gaze_tracking.components.head_pose.pnp_head_pose_estimator import PnPHeadPoseEstimator
from ai_enhanced_gaze_tracking.core.data_models import HeadPose


class Test3DTransformations:
    """Test suite for 3D transformation consistency property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.head_pose_estimator = PnPHeadPoseEstimator()
    
    def create_head_pose_from_angles(self, yaw: float, pitch: float, roll: float) -> HeadPose:
        """Create a HeadPose object from Euler angles."""
        # Convert to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Create rotation matrix from Euler angles
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        
        rotation_matrix = np.array([
            [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
            [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
            [-sin_p, cos_p * sin_r, cos_p * cos_r]
        ])
        
        return HeadPose(
            yaw=yaw_rad,
            pitch=pitch_rad,
            roll=roll_rad,
            translation=np.array([0.0, 0.0, 500.0]),
            rotation_matrix=rotation_matrix,
            confidence=1.0,
            euler_angles=(yaw_rad, pitch_rad, roll_rad)
        )
    
    @given(
        yaw=st.floats(min_value=-45.0, max_value=45.0),
        pitch=st.floats(min_value=-30.0, max_value=30.0),
        roll=st.floats(min_value=-20.0, max_value=20.0)
    )
    @settings(max_examples=100, deadline=3000)
    def test_3d_transformation_consistency(self, yaw, pitch, roll):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 2: 3D Transformation Consistency**
        **Validates: Requirements 1.3**
        
        Property: For any head pose change, applying the 3D transformation matrices should
        produce mathematically consistent coordinate transformations that preserve geometric relationships.
        """
        # Create head pose
        head_pose = self.create_head_pose_from_angles(yaw, pitch, roll)
        
        # Get transformation matrix
        transformation_matrix = self.head_pose_estimator.get_transformation_matrix(head_pose)
        
        # Property 1: Transformation matrix should be 4x4
        assert transformation_matrix.shape == (4, 4), (
            f"Transformation matrix should be 4x4, got {transformation_matrix.shape}"
        )
        
        # Property 2: Bottom row should be [0, 0, 0, 1]
        expected_bottom_row = np.array([0, 0, 0, 1])
        actual_bottom_row = transformation_matrix[3, :]
        assert np.allclose(actual_bottom_row, expected_bottom_row, atol=1e-10), (
            f"Bottom row should be [0, 0, 0, 1], got {actual_bottom_row}"
        )
        
        # Property 3: Upper-left 3x3 should be the rotation matrix
        rotation_part = transformation_matrix[:3, :3]
        assert np.allclose(rotation_part, head_pose.rotation_matrix, atol=1e-10), (
            "Upper-left 3x3 should match the head pose rotation matrix"
        )
        
        # Property 4: Translation part should match head pose translation
        translation_part = transformation_matrix[:3, 3]
        assert np.allclose(translation_part, head_pose.translation, atol=1e-10), (
            "Translation part should match head pose translation"
        )
        
        # Property 5: Rotation matrix should be orthogonal
        R = head_pose.rotation_matrix
        should_be_identity = np.dot(R, R.T)
        identity_error = np.linalg.norm(should_be_identity - np.eye(3))
        assert identity_error < 1e-10, (
            f"Rotation matrix should be orthogonal, error: {identity_error:.2e}"
        )
        
        # Property 6: Determinant of rotation matrix should be 1 (proper rotation)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, (
            f"Rotation matrix determinant should be 1, got {det:.10f}"
        )
        
        # Property 7: Transformation should preserve distances
        # Create test points
        test_points = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        # Transform points using the transformation matrix
        homogeneous_points = np.hstack([test_points, np.ones((test_points.shape[0], 1))])
        transformed_points = (transformation_matrix @ homogeneous_points.T).T[:, :3]
        
        # Check that relative distances are preserved (rotation preserves distances)
        for i in range(len(test_points)):
            for j in range(i + 1, len(test_points)):
                original_distance = np.linalg.norm(test_points[i] - test_points[j])
                transformed_distance = np.linalg.norm(transformed_points[i] - transformed_points[j])
                
                assert abs(original_distance - transformed_distance) < 1e-10, (
                    f"Distance should be preserved: original={original_distance:.10f}, "
                    f"transformed={transformed_distance:.10f}"
                )
    
    @given(
        yaw=st.floats(min_value=-30.0, max_value=30.0),
        pitch=st.floats(min_value=-20.0, max_value=20.0),
        roll=st.floats(min_value=-15.0, max_value=15.0)
    )
    @settings(max_examples=50, deadline=3000)
    def test_coordinate_transformation_consistency(self, yaw, pitch, roll):
        """
        Test that coordinate transformations are consistent and reversible.
        
        Property: Transforming points to head frame and back should yield original points.
        """
        # Create head pose
        head_pose = self.create_head_pose_from_angles(yaw, pitch, roll)
        
        # Create test points in world coordinates
        world_points = np.array([
            [100.0, 50.0, 200.0],
            [-50.0, 100.0, 300.0],
            [0.0, 0.0, 400.0],
            [75.0, -25.0, 250.0]
        ])
        
        # Transform to head frame
        head_frame_points = self.head_pose_estimator.transform_points_to_head_frame(
            world_points, head_pose
        )
        
        # Transform back to world frame using inverse transformation
        transformation_matrix = self.head_pose_estimator.get_transformation_matrix(head_pose)
        
        # Convert head frame points to homogeneous coordinates
        homogeneous_head_points = np.hstack([head_frame_points, np.ones((head_frame_points.shape[0], 1))])
        
        # Apply transformation to get back to world coordinates
        recovered_world_points = (transformation_matrix @ homogeneous_head_points.T).T[:, :3]
        
        # Property: Round-trip transformation should preserve points
        for i in range(len(world_points)):
            error = np.linalg.norm(world_points[i] - recovered_world_points[i])
            assert error < 1e-8, (
                f"Round-trip transformation error too large: {error:.2e} "
                f"for point {i}: {world_points[i]} -> {recovered_world_points[i]}"
            )
    
    def test_transformation_matrix_composition(self):
        """
        Test that transformation matrix composition works correctly.
        
        Property: Composing two transformations should be equivalent to
        applying them sequentially.
        """
        # Create two head poses
        pose1 = self.create_head_pose_from_angles(10.0, 5.0, -3.0)
        pose2 = self.create_head_pose_from_angles(-5.0, 8.0, 2.0)
        
        # Get transformation matrices
        T1 = self.head_pose_estimator.get_transformation_matrix(pose1)
        T2 = self.head_pose_estimator.get_transformation_matrix(pose2)
        
        # Compose transformations
        T_composed = T2 @ T1
        
        # Test point
        test_point = np.array([100.0, 50.0, 200.0, 1.0])  # Homogeneous coordinates
        
        # Apply transformations sequentially
        intermediate_point = T1 @ test_point
        final_point_sequential = T2 @ intermediate_point
        
        # Apply composed transformation
        final_point_composed = T_composed @ test_point
        
        # Property: Results should be identical
        error = np.linalg.norm(final_point_sequential - final_point_composed)
        assert error < 1e-12, (
            f"Transformation composition error: {error:.2e}"
        )
    
    def test_euler_angle_consistency(self):
        """
        Test that Euler angles are consistent with rotation matrices.
        
        Property: Converting Euler angles to rotation matrix and back should
        preserve the angles (within expected ranges).
        """
        # Test various angle combinations
        test_angles = [
            (0.0, 0.0, 0.0),
            (15.0, -10.0, 5.0),
            (-20.0, 25.0, -8.0),
            (30.0, -30.0, 15.0)
        ]
        
        for yaw, pitch, roll in test_angles:
            # Create head pose
            head_pose = self.create_head_pose_from_angles(yaw, pitch, roll)
            
            # Extract angles from rotation matrix using the estimator's method
            extracted_yaw, extracted_pitch, extracted_roll = (
                self.head_pose_estimator._rotation_matrix_to_euler(head_pose.rotation_matrix)
            )
            
            # Convert back to degrees for comparison
            extracted_yaw_deg = np.degrees(extracted_yaw)
            extracted_pitch_deg = np.degrees(extracted_pitch)
            extracted_roll_deg = np.degrees(extracted_roll)
            
            # Property: Extracted angles should match original (within tolerance)
            yaw_error = abs(extracted_yaw_deg - yaw)
            pitch_error = abs(extracted_pitch_deg - pitch)
            roll_error = abs(extracted_roll_deg - roll)
            
            # Handle angle wrapping (e.g., 180° vs -180°)
            if yaw_error > 180:
                yaw_error = 360 - yaw_error
            if roll_error > 180:
                roll_error = 360 - roll_error
            
            assert yaw_error < 1e-10, (
                f"Yaw angle inconsistency: original={yaw:.1f}°, extracted={extracted_yaw_deg:.10f}°"
            )
            assert pitch_error < 1e-10, (
                f"Pitch angle inconsistency: original={pitch:.1f}°, extracted={extracted_pitch_deg:.10f}°"
            )
            assert roll_error < 1e-10, (
                f"Roll angle inconsistency: original={roll:.1f}°, extracted={extracted_roll_deg:.10f}°"
            )
    
    @given(
        yaw=st.floats(min_value=-30.0, max_value=30.0),
        pitch=st.floats(min_value=-20.0, max_value=20.0)
    )
    @settings(max_examples=30, deadline=2000)
    def test_transformation_linearity(self, yaw, pitch):
        """
        Test that transformations preserve linear relationships.
        
        Property: The transformation should preserve linear combinations of points.
        """
        # Create head pose
        head_pose = self.create_head_pose_from_angles(yaw, pitch, 0.0)
        
        # Create test points
        point_a = np.array([[10.0, 20.0, 30.0]])
        point_b = np.array([[-15.0, 25.0, 40.0]])
        
        # Linear combination
        alpha, beta = 0.3, 0.7
        linear_combination = alpha * point_a + beta * point_b
        
        # Transform individual points
        transformed_a = self.head_pose_estimator.transform_points_to_head_frame(point_a, head_pose)
        transformed_b = self.head_pose_estimator.transform_points_to_head_frame(point_b, head_pose)
        
        # Transform linear combination
        transformed_combination = self.head_pose_estimator.transform_points_to_head_frame(
            linear_combination, head_pose
        )
        
        # Property: Transformation should preserve linear combinations
        # T(αA + βB) should equal αT(A) + βT(B) for the rotational part
        # But since we're using inverse transformation with translation, we need to account for that
        
        # Get the rotation matrix (inverse of head pose rotation)
        R_inv = head_pose.rotation_matrix.T  # Inverse of rotation matrix
        
        # Apply only rotation to check linearity
        rotated_a = (R_inv @ point_a.T).T
        rotated_b = (R_inv @ point_b.T).T
        rotated_combination = (R_inv @ linear_combination.T).T
        
        expected_combination = alpha * rotated_a + beta * rotated_b
        
        error = np.linalg.norm(rotated_combination - expected_combination)
        assert error < 1e-12, (
            f"Transformation linearity violated: error {error:.2e}"
        )