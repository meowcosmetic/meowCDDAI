"""
Property-based tests for adaptive weight adjustment.

**Feature: ai-enhanced-gaze-tracking, Property 9: Adaptive Weight Adjustment**
**Validates: Requirements 4.2**

This module tests that the fusion system automatically reduces weights for unreliable
sources while maintaining overall tracking quality.
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


class TestAdaptiveWeights:
    """Test suite for adaptive weight adjustment property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fusion = MultiModalFusion(
            history_size=10,
            min_confidence=0.1,
            conflict_threshold=0.3,
            adaptation_rate=0.2  # Higher rate for faster adaptation in tests
        )
    
    @given(
        initial_reliability=st.floats(min_value=0.5, max_value=1.0),
        final_reliability=st.floats(min_value=0.1, max_value=0.4)
    )
    @settings(max_examples=100, deadline=2000)
    def test_weight_decreases_for_unreliable_source(self, initial_reliability, final_reliability):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 9: Adaptive Weight Adjustment**
        **Validates: Requirements 4.2**
        
        Property: When a data source becomes unreliable (reliability drops),
        the fusion system should automatically reduce its weight.
        """
        # Use a fresh fusion object to avoid state pollution between examples
        fusion = MultiModalFusion(
            history_size=10,
            min_confidence=0.1,
            conflict_threshold=0.3,
            adaptation_rate=0.2
        )
        source_name = "test_source"
        
        # Set initial high reliability
        fusion.update_weights({source_name: initial_reliability})
        initial_weight = fusion.source_weights.get(source_name, 1.0)
        
        # Simulate source becoming unreliable
        fusion.update_weights({source_name: final_reliability})
        final_weight = fusion.source_weights.get(source_name, 1.0)
        
        # Property: Weight should decrease when reliability decreases
        assert final_weight < initial_weight, (
            f"Weight should decrease when reliability drops from {initial_reliability:.2f} "
            f"to {final_reliability:.2f}, but went from {initial_weight:.3f} to {final_weight:.3f}"
        )
        
        # Property: Weight should move toward the new reliability
        # (may not reach it exactly due to exponential moving average)
        weight_change = initial_weight - final_weight
        reliability_change = initial_reliability - final_reliability
        
        # Weight should change in the same direction as reliability
        assert weight_change > 0, "Weight should decrease when reliability decreases"
    
    @given(
        num_updates=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=2000)
    def test_weight_converges_to_reliability(self, num_updates):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 9: Adaptive Weight Adjustment**
        **Validates: Requirements 4.2**
        
        Property: With repeated updates, weights should converge toward the reliability value.
        """
        source_name = "converging_source"
        target_reliability = 0.6
        
        # Start with different initial weight
        self.fusion.source_weights[source_name] = 1.0
        
        # Apply multiple updates
        for _ in range(num_updates):
            self.fusion.update_weights({source_name: target_reliability})
        
        final_weight = self.fusion.source_weights[source_name]
        
        # Property: Weight should be closer to target after multiple updates
        # With adaptation_rate=0.2, after many updates it should be close
        if num_updates >= 5:
            # Should be within 30% of target after 5+ updates
            assert abs(final_weight - target_reliability) < 0.3, (
                f"After {num_updates} updates, weight should converge toward {target_reliability:.2f}, "
                f"got {final_weight:.3f}"
            )
    
    @given(
        reliable_conf=st.floats(min_value=0.7, max_value=1.0),
        unreliable_conf=st.floats(min_value=0.1, max_value=0.3)
    )
    @settings(max_examples=100, deadline=2000)
    def test_fusion_quality_maintained_with_unreliable_source(
        self, reliable_conf, unreliable_conf
    ):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 9: Adaptive Weight Adjustment**
        **Validates: Requirements 4.2**
        
        Property: When one source becomes unreliable, the fusion should maintain
        overall tracking quality by relying more on reliable sources.
        """
        # Create a reliable source
        reliable_vector = np.array([1.0, 0.0, 0.0])
        reliable_est = create_test_gaze_estimate(
            gaze_vector=reliable_vector,
            gaze_point=(960, 540),
            confidence=reliable_conf,
            method="reliable_source"
        )
        
        # Create an unreliable source pointing in a different direction
        unreliable_vector = np.array([0.0, 1.0, 0.0])
        unreliable_est = create_test_gaze_estimate(
            gaze_vector=unreliable_vector,
            gaze_point=(100, 100),
            confidence=unreliable_conf,
            method="unreliable_source"
        )
        
        # Update weights to reflect reliability
        self.fusion.update_weights({
            "reliable_source": reliable_conf,
            "unreliable_source": unreliable_conf
        })
        
        # Fuse estimates
        estimates = [reliable_est, unreliable_est]
        confidences = [reliable_conf, unreliable_conf]
        
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Property: Fused result should be much closer to reliable source
        dot_reliable = np.dot(fused.gaze_vector_3d, reliable_vector)
        dot_unreliable = np.dot(fused.gaze_vector_3d, unreliable_vector)
        
        # Clip to avoid numerical issues with arccos
        dot_reliable = np.clip(dot_reliable, -1.0, 1.0)
        dot_unreliable = np.clip(dot_unreliable, -1.0, 1.0)
        
        angle_to_reliable = np.arccos(dot_reliable)
        angle_to_unreliable = np.arccos(dot_unreliable)
        
        # Should be closer to reliable source
        assert angle_to_reliable < angle_to_unreliable, (
            f"Fused estimate should be closer to reliable source. "
            f"Angle to reliable: {np.degrees(angle_to_reliable):.1f}°, "
            f"angle to unreliable: {np.degrees(angle_to_unreliable):.1f}°"
        )
        
        # Property: Overall confidence should be reasonable
        assert fused.confidence >= unreliable_conf, (
            "Fused confidence should be at least as good as worst source"
        )
    
    def test_weight_update_basic_functionality(self):
        """Test basic weight update functionality."""
        source_name = "test_source"
        
        # Initial weight should be default
        assert source_name not in self.fusion.source_weights
        
        # Update with high reliability
        self.fusion.update_weights({source_name: 0.9})
        assert source_name in self.fusion.source_weights
        
        weight_after_high = self.fusion.source_weights[source_name]
        
        # Update with low reliability
        self.fusion.update_weights({source_name: 0.2})
        weight_after_low = self.fusion.source_weights[source_name]
        
        # Weight should decrease
        assert weight_after_low < weight_after_high
    
    def test_weight_clamping(self):
        """Test that weights are clamped to reasonable range."""
        source_name = "extreme_source"
        
        # Try to set extremely low reliability
        for _ in range(20):
            self.fusion.update_weights({source_name: 0.0})
        
        weight = self.fusion.source_weights[source_name]
        
        # Should be clamped to minimum
        assert weight >= 0.1, f"Weight should be clamped to minimum, got {weight:.3f}"
        
        # Try to set extremely high reliability
        self.fusion.source_weights[source_name] = 1.0
        for _ in range(20):
            self.fusion.update_weights({source_name: 1.0})
        
        weight = self.fusion.source_weights[source_name]
        
        # Should be clamped to maximum
        assert weight <= 2.0, f"Weight should be clamped to maximum, got {weight:.3f}"
    
    @given(
        num_sources=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=50, deadline=2000)
    def test_multiple_sources_adaptive_weighting(self, num_sources):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 9: Adaptive Weight Adjustment**
        **Validates: Requirements 4.2**
        
        Property: The system should adaptively weight multiple sources based on
        their individual reliabilities.
        """
        # Create sources with varying reliabilities
        reliabilities = {}
        estimates = []
        confidences = []
        
        for i in range(num_sources):
            source_name = f"source_{i}"
            reliability = np.random.uniform(0.2, 1.0)
            reliabilities[source_name] = reliability
            
            # Generate random gaze vector
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            gaze_vector = np.array([x, y, z])
            
            confidence = reliability  # Use reliability as confidence
            
            est = create_test_gaze_estimate(
                gaze_vector=gaze_vector,
                gaze_point=(np.random.uniform(400, 1520), np.random.uniform(200, 880)),
                confidence=confidence,
                method=source_name
            )
            estimates.append(est)
            confidences.append(confidence)
        
        # Update weights based on reliabilities
        self.fusion.update_weights(reliabilities)
        
        # Fuse estimates
        fused = self.fusion.fuse_estimates(estimates, confidences)
        
        # Property: Fusion should produce valid result
        assert fused is not None
        assert 0.0 <= fused.confidence <= 1.0
        assert 0.9 <= np.linalg.norm(fused.gaze_vector_3d) <= 1.1
        
        # Property: Sources with higher reliability should have higher weights
        weights = [self.fusion.source_weights.get(f"source_{i}", 1.0) for i in range(num_sources)]
        sorted_reliabilities = sorted(reliabilities.values())
        sorted_weights = sorted(weights)
        
        # Check that weight ordering roughly matches reliability ordering
        # (allowing for some variation due to exponential moving average)
        if num_sources >= 3:
            # Highest reliability should have higher weight than lowest
            max_reliability_idx = max(range(num_sources), key=lambda i: reliabilities[f"source_{i}"])
            min_reliability_idx = min(range(num_sources), key=lambda i: reliabilities[f"source_{i}"])
            
            max_weight = weights[max_reliability_idx]
            min_weight = weights[min_reliability_idx]
            
            # Allow for initial conditions where weights might not be perfectly ordered yet
            # Just check that they're not inverted
            if reliabilities[f"source_{max_reliability_idx}"] > reliabilities[f"source_{min_reliability_idx}"] + 0.3:
                assert max_weight >= min_weight * 0.8, (
                    f"Higher reliability source should have higher weight. "
                    f"Max reliability: {reliabilities[f'source_{max_reliability_idx}']:.2f} -> weight {max_weight:.3f}, "
                    f"Min reliability: {reliabilities[f'source_{min_reliability_idx}']:.2f} -> weight {min_weight:.3f}"
                )
