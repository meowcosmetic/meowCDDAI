"""
Property-based tests for multi-modal fusion weighting.

**Feature: ai-enhanced-gaze-tracking, Property 8: Multi-Modal Fusion Weighting**
**Validates: Requirements 4.1**

This module tests that the fusion algorithm weights contributions based on confidence
scores and produces estimates within the convex hull of input estimates.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import List

from ai_enhanced_gaze_tracking.components.sensor_fusion.multi_modal_fusion import MultiModalFusion
from ai_enhanced_gaze_tracking.core.data_models import GazeEstimate, HeadPose, QualityMetrics


def create_test_gaze_estimate(
    gaze_vector: np.ndarray,
    gaze_point: tuple,
    confidence: float,
    method: str = "test"
) -> GazeEstimate:
    """Create a test gaze estimate."""
    # Normalize gaze vector
    norm = np.linalg.norm(gaze_vector)
    if norm > 0:
        gaze_vector = gaze_vector / norm
    
    head_pose = HeadPose(
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        translation=np.array([0.0, 0.0, 500.0]),
        rotation_matrix=np.eye(3),
        confidence=0.8
    )
    
    quality_metrics = QualityMetrics(
        overall_quality=0.8,
        head_pose_quality=0.8,
        lighting_quality=0.8,
        occlusion_level=0.1,
        motion_blur=0.1,
        tracking_stability=0.8
    )
    
    return GazeEstimate(
        gaze_vector_3d=gaze_vector,
        gaze_point_2d=gaze_point,
        confidence=confidence,
        head_pose=head_pose,
        timestamp=0.0,
        source_confidences={method: confidence},
        quality_metrics=quality_metrics,
        method=method
    )


# Strategy for generating valid 3D unit vectors
@st.composite
def unit_vector_3d(draw):
    """Generate a random 3D unit vector."""
    # Generate random direction
    theta = draw(st.floats(min_value=0, max_value=2*np.pi))
    phi = draw(st.floats(min_value=0, max_value=np.pi))
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    return np.array([x, y, z])


@st.composite
def gaze_estimate_strategy(draw, method="test"):
    """Generate a random gaze estimate."""
    gaze_vector = draw(unit_vector_3d())
    gaze_x = draw(st.floats(min_value=0, max_value=1920))
    gaze_y = draw(st.floats(min_value=0, max_value=1080))
    confidence = draw(st.floats(min_value=0.1, max_value=1.0))
    
    return create_test_gaze_estimate(
        gaze_vector=gaze_vector,
        gaze_point=(gaze_x, gaze_y),
        confidence=confidence,
        method=method
    )


class TestFusionWeighting:
    """Test suite for multi-modal fusion weighting property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = MultiModalFusion(
            history_size=10,
            min_confidence=0.1,
            conflict_threshold=0.3,
            adaptation_rate=0.1
        )
    
    @given(
        num_estimates=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=2000)
    def test_fusion_produces_valid_estimate(self, num_estimates):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 8: Multi-Modal Fusion Weighting**
        **Validates: Requirements 4.1**
        
        Property: For any combination of available data sources, the fusion algorithm
        should produce a valid gaze estimate.
        """
        # Generate random estimates
        estimates = []
        confidences = []
        
        for i in range(num_estimates):
            # Generate random unit vector
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            gaze_vector = np.array([x, y, z])
            
            gaze_point = (
                np.random.uniform(0, 1920),
                np.random.uniform(0, 1080)
            )
            confidence = np.random.uniform(0.2, 1.0)
            
            est = create_test_gaze_estimate(
                gaze_vector=gaze_vector,
                gaze_point=gaze_point,
                confidence=confidence,
                method=f"source_{i}"
            )
            estimates.append(est)
            confidences.append(confidence)
        
        # Fuse estimates
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Property 1: Fused estimate should be valid
        assert fused is not None, "Fusion should produce a result"
        assert hasattr(fused, 'gaze_vector_3d'), "Should have 3D gaze vector"
        assert hasattr(fused, 'gaze_point_2d'), "Should have 2D gaze point"
        assert hasattr(fused, 'confidence'), "Should have confidence score"
        
        # Property 2: Gaze vector should be normalized
        vector_norm = np.linalg.norm(fused.gaze_vector_3d)
        assert 0.9 <= vector_norm <= 1.1, (
            f"Fused gaze vector should be normalized, got norm {vector_norm:.3f}"
        )
        
        # Property 3: Confidence should be in valid range
        assert 0.0 <= fused.confidence <= 1.0, (
            f"Confidence should be in [0,1], got {fused.confidence:.3f}"
        )
        
        # Property 4: 2D point should be reasonable
        x, y = fused.gaze_point_2d
        assert -1000 <= x <= 3000, f"X coordinate should be reasonable, got {x:.1f}"
        assert -1000 <= y <= 2000, f"Y coordinate should be reasonable, got {y:.1f}"
    
    @given(
        num_estimates=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=100, deadline=2000)
    def test_fusion_respects_confidence_weighting(self, num_estimates):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 8: Multi-Modal Fusion Weighting**
        **Validates: Requirements 4.1**
        
        Property: The fusion should weight contributions based on confidence scores.
        Higher confidence estimates should have more influence.
        """
        # Create estimates with varying confidences
        estimates = []
        confidences = []
        
        # Create a high-confidence estimate pointing in a specific direction
        high_conf_vector = np.array([1.0, 0.0, 0.0])
        high_conf = 0.9
        
        est_high = create_test_gaze_estimate(
            gaze_vector=high_conf_vector,
            gaze_point=(960, 540),
            confidence=high_conf,
            method="high_conf"
        )
        estimates.append(est_high)
        confidences.append(high_conf)
        
        # Create low-confidence estimates pointing in different directions
        for i in range(num_estimates - 1):
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            low_conf_vector = np.array([x, y, z])
            
            low_conf = np.random.uniform(0.1, 0.3)
            
            est_low = create_test_gaze_estimate(
                gaze_vector=low_conf_vector,
                gaze_point=(np.random.uniform(0, 1920), np.random.uniform(0, 1080)),
                confidence=low_conf,
                method=f"low_conf_{i}"
            )
            estimates.append(est_low)
            confidences.append(low_conf)
        
        # Fuse estimates
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Property: Fused vector should be closer to high-confidence estimate
        # Calculate angular distance to high-confidence estimate
        dot_high = np.dot(fused.gaze_vector_3d, high_conf_vector)
        dot_high = np.clip(dot_high, -1.0, 1.0)
        angle_to_high = np.arccos(dot_high)
        
        # The fused result should be reasonably close to the high-confidence estimate
        # Allow up to 60 degrees deviation (generous for property testing)
        max_angle = np.radians(60)
        assert angle_to_high <= max_angle, (
            f"Fused estimate should be influenced by high-confidence source, "
            f"angle deviation: {np.degrees(angle_to_high):.1f}°"
        )
    
    @given(
        num_estimates=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=100, deadline=2000)
    def test_fusion_within_convex_hull(self, num_estimates):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 8: Multi-Modal Fusion Weighting**
        **Validates: Requirements 4.1**
        
        Property: The fused estimate should be within the convex hull of input estimates
        (for 2D gaze points).
        """
        # Generate estimates with similar vectors but different 2D points
        base_vector = np.array([0.0, 0.0, 1.0])
        
        estimates = []
        confidences = []
        points_2d = []
        
        for i in range(num_estimates):
            # Small variation in vector
            noise = np.random.normal(0, 0.1, 3)
            gaze_vector = base_vector + noise
            gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
            
            # Random 2D point
            gaze_x = np.random.uniform(400, 1520)
            gaze_y = np.random.uniform(200, 880)
            points_2d.append((gaze_x, gaze_y))
            
            confidence = np.random.uniform(0.3, 1.0)
            
            est = create_test_gaze_estimate(
                gaze_vector=gaze_vector,
                gaze_point=(gaze_x, gaze_y),
                confidence=confidence,
                method=f"source_{i}"
            )
            estimates.append(est)
            confidences.append(confidence)
        
        # Fuse estimates
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Property: Fused 2D point should be within bounding box of input points
        # (relaxed convex hull check)
        points_array = np.array(points_2d)
        min_x, min_y = points_array.min(axis=0)
        max_x, max_y = points_array.max(axis=0)
        
        fused_x, fused_y = fused.gaze_point_2d
        
        # Allow small margin for numerical errors and Kalman filtering
        margin = 100
        assert min_x - margin <= fused_x <= max_x + margin, (
            f"Fused X should be within input range [{min_x:.1f}, {max_x:.1f}], "
            f"got {fused_x:.1f}"
        )
        assert min_y - margin <= fused_y <= max_y + margin, (
            f"Fused Y should be within input range [{min_y:.1f}, {max_y:.1f}], "
            f"got {fused_y:.1f}"
        )
    
    def test_fusion_basic_functionality(self):
        """Test basic fusion functionality with known inputs."""
        # Create two estimates pointing in similar directions
        est1 = create_test_gaze_estimate(
            gaze_vector=np.array([1.0, 0.0, 0.0]),
            gaze_point=(800, 600),
            confidence=0.8,
            method="source_1"
        )
        
        est2 = create_test_gaze_estimate(
            gaze_vector=np.array([0.9, 0.1, 0.0]),
            gaze_point=(820, 610),
            confidence=0.6,
            method="source_2"
        )
        
        estimates = [est1, est2]
        confidences = [0.8, 0.6]
        
        # Fuse
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Basic checks
        assert fused is not None
        assert 0.0 <= fused.confidence <= 1.0
        assert np.linalg.norm(fused.gaze_vector_3d) > 0.9
        
        # Should be influenced more by higher confidence estimate
        dot_product = np.dot(fused.gaze_vector_3d, est1.gaze_vector_3d)
        assert dot_product > 0.8, "Should be close to higher confidence estimate"
    
    def test_fusion_filters_low_confidence(self):
        """Test that fusion filters out very low confidence estimates."""
        # Create one good estimate and one very poor estimate
        good_est = create_test_gaze_estimate(
            gaze_vector=np.array([1.0, 0.0, 0.0]),
            gaze_point=(800, 600),
            confidence=0.8,
            method="good"
        )
        
        bad_est = create_test_gaze_estimate(
            gaze_vector=np.array([0.0, 1.0, 0.0]),
            gaze_point=(100, 100),
            confidence=0.05,  # Below min_confidence threshold
            method="bad"
        )
        
        estimates = [good_est, bad_est]
        confidences = [0.8, 0.05]
        
        # Fuse
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Should be very close to good estimate (bad one filtered out)
        dot_product = np.dot(fused.gaze_vector_3d, good_est.gaze_vector_3d)
        assert dot_product > 0.95, "Should mostly use good estimate"
