"""
Property-based tests for conflict resolution consistency.

**Feature: ai-enhanced-gaze-tracking, Property 10: Conflict Resolution Consistency**
**Validates: Requirements 4.3**

This module tests that conflict resolution follows temporal consistency and geometric
constraints to produce physically plausible results.
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
    timestamp: float = 0.0,
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
        timestamp=timestamp,
        source_confidences={method: confidence},
        quality_metrics=quality_metrics,
        method=method
    )


class TestConflictResolution:
    """Test suite for conflict resolution consistency property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = MultiModalFusion(
            history_size=10,
            min_confidence=0.1,
            conflict_threshold=0.3,
            adaptation_rate=0.1
        )
    
    @given(
        angle_diff=st.floats(min_value=60, max_value=150)
    )
    @settings(max_examples=100, deadline=2000)
    def test_conflict_resolution_produces_valid_result(self, angle_diff):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 10: Conflict Resolution Consistency**
        **Validates: Requirements 4.3**
        
        Property: For any conflicting predictions, the resolution should produce
        a physically plausible result.
        """
        # Create two conflicting estimates
        vector1 = np.array([1.0, 0.0, 0.0])
        
        # Create second vector at specified angle
        angle_rad = np.radians(angle_diff)
        vector2 = np.array([
            np.cos(angle_rad),
            np.sin(angle_rad),
            0.0
        ])
        
        est1 = create_test_gaze_estimate(
            gaze_vector=vector1,
            gaze_point=(800, 600),
            confidence=0.7,
            method="source_1"
        )
        
        est2 = create_test_gaze_estimate(
            gaze_vector=vector2,
            gaze_point=(1000, 400),
            confidence=0.7,
            method="source_2"
        )
        
        estimates = [est1, est2]
        temporal_history = []
        
        # Resolve conflict
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Property 1: Should produce valid result
        assert resolved is not None, "Conflict resolution should produce a result"
        assert hasattr(resolved, 'gaze_vector_3d'), "Should have 3D gaze vector"
        assert hasattr(resolved, 'confidence'), "Should have confidence score"
        
        # Property 2: Gaze vector should be normalized
        vector_norm = np.linalg.norm(resolved.gaze_vector_3d)
        assert 0.9 <= vector_norm <= 1.1, (
            f"Resolved gaze vector should be normalized, got norm {vector_norm:.3f}"
        )
        
        # Property 3: Confidence should be valid
        assert 0.0 <= resolved.confidence <= 1.0, (
            f"Confidence should be in [0,1], got {resolved.confidence:.3f}"
        )
        
        # Property 4: Result should be physically plausible (not NaN or infinite)
        assert np.all(np.isfinite(resolved.gaze_vector_3d)), (
            "Resolved vector should contain finite values"
        )
        assert np.isfinite(resolved.confidence), "Confidence should be finite"
    
    @given(
        num_history=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=100, deadline=2000)
    def test_conflict_resolution_uses_temporal_consistency(self, num_history):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 10: Conflict Resolution Consistency**
        **Validates: Requirements 4.3**
        
        Property: When temporal history is available, conflict resolution should
        favor estimates consistent with recent history.
        """
        # Create consistent temporal history pointing in one direction
        history_vector = np.array([1.0, 0.0, 0.0])
        temporal_history = []
        
        for i in range(num_history):
            # Add small noise to history
            noise = np.random.normal(0, 0.05, 3)
            noisy_vector = history_vector + noise
            noisy_vector = noisy_vector / np.linalg.norm(noisy_vector)
            
            hist_est = create_test_gaze_estimate(
                gaze_vector=noisy_vector,
                gaze_point=(800 + np.random.normal(0, 10), 600 + np.random.normal(0, 10)),
                confidence=0.8,
                timestamp=float(i),
                method="history"
            )
            temporal_history.append(hist_est)
        
        # Create two conflicting current estimates
        # One consistent with history, one not
        consistent_est = create_test_gaze_estimate(
            gaze_vector=np.array([0.95, 0.1, 0.0]) / np.linalg.norm([0.95, 0.1, 0.0]),
            gaze_point=(810, 605),
            confidence=0.6,
            method="consistent"
        )
        
        inconsistent_est = create_test_gaze_estimate(
            gaze_vector=np.array([0.0, 1.0, 0.0]),
            gaze_point=(400, 300),
            confidence=0.6,
            method="inconsistent"
        )
        
        estimates = [consistent_est, inconsistent_est]
        
        # Resolve conflict
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Property: Resolved estimate should be closer to consistent estimate
        dot_consistent = np.dot(resolved.gaze_vector_3d, consistent_est.gaze_vector_3d)
        dot_inconsistent = np.dot(resolved.gaze_vector_3d, inconsistent_est.gaze_vector_3d)
        
        dot_consistent = np.clip(dot_consistent, -1.0, 1.0)
        dot_inconsistent = np.clip(dot_inconsistent, -1.0, 1.0)
        
        angle_to_consistent = np.arccos(dot_consistent)
        angle_to_inconsistent = np.arccos(dot_inconsistent)
        
        assert angle_to_consistent < angle_to_inconsistent, (
            f"Resolved estimate should favor temporally consistent estimate. "
            f"Angle to consistent: {np.degrees(angle_to_consistent):.1f}°, "
            f"angle to inconsistent: {np.degrees(angle_to_inconsistent):.1f}°"
        )
    
    @given(
        confidence_diff=st.floats(min_value=0.2, max_value=0.6)
    )
    @settings(max_examples=100, deadline=2000)
    def test_conflict_resolution_considers_confidence(self, confidence_diff):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 10: Conflict Resolution Consistency**
        **Validates: Requirements 4.3**
        
        Property: When resolving conflicts without history, higher confidence
        estimates should be preferred.
        """
        # Create two conflicting estimates with different confidences
        high_conf = 0.9
        low_conf = max(0.1, high_conf - confidence_diff)
        
        high_conf_vector = np.array([1.0, 0.0, 0.0])
        low_conf_vector = np.array([0.0, 1.0, 0.0])
        
        high_conf_est = create_test_gaze_estimate(
            gaze_vector=high_conf_vector,
            gaze_point=(800, 600),
            confidence=high_conf,
            method="high_conf"
        )
        
        low_conf_est = create_test_gaze_estimate(
            gaze_vector=low_conf_vector,
            gaze_point=(400, 300),
            confidence=low_conf,
            method="low_conf"
        )
        
        estimates = [high_conf_est, low_conf_est]
        temporal_history = []  # No history
        
        # Resolve conflict
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Property: Should favor high confidence estimate
        dot_high = np.dot(resolved.gaze_vector_3d, high_conf_vector)
        dot_low = np.dot(resolved.gaze_vector_3d, low_conf_vector)
        
        dot_high = np.clip(dot_high, -1.0, 1.0)
        dot_low = np.clip(dot_low, -1.0, 1.0)
        
        angle_to_high = np.arccos(dot_high)
        angle_to_low = np.arccos(dot_low)
        
        assert angle_to_high < angle_to_low, (
            f"Resolved estimate should favor higher confidence estimate. "
            f"High conf: {high_conf:.2f} -> angle {np.degrees(angle_to_high):.1f}°, "
            f"Low conf: {low_conf:.2f} -> angle {np.degrees(angle_to_low):.1f}°"
        )
    
    def test_no_conflict_uses_standard_fusion(self):
        """
        Test that when there's no significant conflict, standard fusion is used.
        """
        # Create two estimates pointing in similar directions (no conflict)
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([0.95, 0.1, 0.0])
        vector2 = vector2 / np.linalg.norm(vector2)
        
        est1 = create_test_gaze_estimate(
            gaze_vector=vector1,
            gaze_point=(800, 600),
            confidence=0.8,
            method="source_1"
        )
        
        est2 = create_test_gaze_estimate(
            gaze_vector=vector2,
            gaze_point=(820, 610),
            confidence=0.7,
            method="source_2"
        )
        
        estimates = [est1, est2]
        temporal_history = []
        
        # Resolve (should use standard fusion since no conflict)
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Should produce valid result
        assert resolved is not None
        assert 0.0 <= resolved.confidence <= 1.0
        assert 0.9 <= np.linalg.norm(resolved.gaze_vector_3d) <= 1.1
        
        # Should be close to both estimates
        dot1 = np.dot(resolved.gaze_vector_3d, vector1)
        dot2 = np.dot(resolved.gaze_vector_3d, vector2)
        
        assert dot1 > 0.8, "Should be close to first estimate"
        assert dot2 > 0.8, "Should be close to second estimate"
    
    def test_single_estimate_returns_itself(self):
        """Test that resolving a single estimate returns it unchanged."""
        est = create_test_gaze_estimate(
            gaze_vector=np.array([1.0, 0.0, 0.0]),
            gaze_point=(800, 600),
            confidence=0.8,
            method="single"
        )
        
        estimates = [est]
        temporal_history = []
        
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Should return the same estimate
        assert resolved is est
    
    @given(
        num_estimates=st.integers(min_value=3, max_value=5)
    )
    @settings(max_examples=50, deadline=2000)
    def test_multiple_conflicts_resolution(self, num_estimates):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 10: Conflict Resolution Consistency**
        **Validates: Requirements 4.3**
        
        Property: When multiple conflicting estimates exist, resolution should
        produce a geometrically consistent result.
        """
        # Create multiple estimates pointing in different directions
        estimates = []
        
        for i in range(num_estimates):
            # Distribute estimates around a circle
            angle = (2 * np.pi * i) / num_estimates
            vector = np.array([
                np.cos(angle),
                np.sin(angle),
                0.0
            ])
            
            confidence = np.random.uniform(0.5, 0.9)
            
            est = create_test_gaze_estimate(
                gaze_vector=vector,
                gaze_point=(
                    960 + 300 * np.cos(angle),
                    540 + 300 * np.sin(angle)
                ),
                confidence=confidence,
                method=f"source_{i}"
            )
            estimates.append(est)
        
        temporal_history = []
        
        # Resolve conflicts
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Property 1: Should produce valid result
        assert resolved is not None
        assert 0.9 <= np.linalg.norm(resolved.gaze_vector_3d) <= 1.1
        assert 0.0 <= resolved.confidence <= 1.0
        
        # Property 2: Result should be geometrically consistent
        # The resolved estimate should match one of the input estimates
        # (since without history, it picks the highest confidence one)
        min_angle = float('inf')
        for est in estimates:
            dot_product = np.dot(resolved.gaze_vector_3d, est.gaze_vector_3d)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle = np.arccos(dot_product)
            min_angle = min(min_angle, angle)
        
        # Should be close to at least one input estimate
        assert min_angle < np.radians(30), (
            f"Resolved estimate should be close to at least one input, "
            f"minimum angle: {np.degrees(min_angle):.1f}°"
        )
    
    def test_geometric_constraints_maintained(self):
        """
        Test that geometric constraints are maintained during conflict resolution.
        """
        # Create estimates that form a geometrically valid configuration
        est1 = create_test_gaze_estimate(
            gaze_vector=np.array([1.0, 0.0, 0.0]),
            gaze_point=(800, 600),
            confidence=0.7,
            method="source_1"
        )
        
        est2 = create_test_gaze_estimate(
            gaze_vector=np.array([0.0, 1.0, 0.0]),
            gaze_point=(600, 800),
            confidence=0.7,
            method="source_2"
        )
        
        estimates = [est1, est2]
        temporal_history = []
        
        resolved = self.fusion.resolve_conflicts(estimates, temporal_history)
        
        # Geometric constraints:
        # 1. Vector should be normalized
        norm = np.linalg.norm(resolved.gaze_vector_3d)
        assert 0.95 <= norm <= 1.05, f"Vector should be normalized, got {norm:.3f}"
        
        # 2. All components should be finite
        assert np.all(np.isfinite(resolved.gaze_vector_3d))
        
        # 3. 2D point should be within reasonable screen bounds
        x, y = resolved.gaze_point_2d
        assert -500 <= x <= 2500, f"X should be reasonable, got {x:.1f}"
        assert -500 <= y <= 1500, f"Y should be reasonable, got {y:.1f}"
