"""
Property-based tests for reference point calibration.

**Feature: ai-enhanced-gaze-tracking, Property 12: Reference Point Calibration**
**Validates: Requirements 5.2**

Property 12: Reference Point Calibration
For any set of reference point observations, the calibration system should use 
these to improve gaze accuracy in a measurable way.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import tempfile
import shutil

from ai_enhanced_gaze_tracking.components.calibration.personal_calibration import (
    PersonalCalibrationSystem
)
from ai_enhanced_gaze_tracking.core.data_models import GazeEstimate, HeadPose, QualityMetrics


# Strategies for generating test data
@st.composite
def consistent_reference_observations_strategy(draw):
    """Generate reference observations with consistent systematic error."""
    num_points = draw(st.integers(min_value=5, max_value=12))
    
    # Generate a systematic offset (the "true" calibration error)
    true_offset_x = draw(st.floats(min_value=-50.0, max_value=50.0))
    true_offset_y = draw(st.floats(min_value=-50.0, max_value=50.0))
    true_offset = np.array([true_offset_x, true_offset_y])
    
    # Assume the offset is meaningful
    assume(np.linalg.norm(true_offset) > 5.0)
    
    observations = []
    for _ in range(num_points):
        # Generate a target point
        target_x = draw(st.floats(min_value=100.0, max_value=1820.0))
        target_y = draw(st.floats(min_value=100.0, max_value=980.0))
        target_point = np.array([target_x, target_y])
        
        # Gaze point has systematic error plus small random noise
        noise_x = draw(st.floats(min_value=-5.0, max_value=5.0))
        noise_y = draw(st.floats(min_value=-5.0, max_value=5.0))
        gaze_point = target_point - true_offset + np.array([noise_x, noise_y])
        
        observations.append((gaze_point, target_point))
    
    return observations, true_offset


@st.composite
def gaze_estimate_strategy(draw):
    """Generate a gaze estimate for testing calibration application."""
    gaze_x = draw(st.floats(min_value=0.0, max_value=1920.0))
    gaze_y = draw(st.floats(min_value=0.0, max_value=1080.0))
    
    return GazeEstimate(
        gaze_vector_3d=np.array([0.0, 0.0, 1.0]),
        gaze_point_2d=(gaze_x, gaze_y),
        confidence=0.9,
        head_pose=HeadPose(
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            translation=np.array([0.0, 0.0, 500.0]),
            rotation_matrix=np.eye(3),
            confidence=0.9
        ),
        timestamp=0.0,
        source_confidences={'ai': 0.9},
        quality_metrics=QualityMetrics(
            overall_quality=0.9,
            head_pose_quality=0.9,
            lighting_quality=0.9,
            occlusion_level=0.0,
            motion_blur=0.0,
            tracking_stability=0.9
        ),
        method="test"
    )


class TestReferencePointCalibration:
    """Test reference point calibration property."""
    
    @given(
        ref_data=consistent_reference_observations_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_reference_point_calibration_improves_accuracy(self, ref_data):
        """
        Property 12: Reference Point Calibration
        
        For any set of reference point observations with consistent systematic error,
        the calibration system should use these observations to improve gaze accuracy
        in a measurable way.
        
        This test verifies that:
        1. Reference observations are used to calculate calibration offset
        2. The calculated offset improves accuracy on the reference data
        3. The improvement is measurable and significant
        """
        ref_obs, true_offset = ref_data
        
        tmp_path = tempfile.mkdtemp()
        try:
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            # Perform calibration with reference observations
            face_chars = {
                'interpupillary_distance': 60.0,
                'face_width': 150.0,
                'face_height': 180.0,
                'is_child': False
            }
            
            result = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            # Property: Calibration should use reference observations
            assert len(result['gaze_offset']) == 2, \
                "Calibration should produce 2D offset"
            
            # Property: With consistent systematic error, calibration should improve accuracy
            assert result['accuracy_improvement'] > 0.0, \
                f"Calibration with consistent reference data should improve accuracy, got {result['accuracy_improvement']}"
            
            # Property: The calculated offset should be close to the true systematic error
            calculated_offset = result['gaze_offset']
            offset_error = np.linalg.norm(calculated_offset - true_offset)
            
            # Allow for some error due to noise, but should be reasonably close
            max_acceptable_error = 10.0  # pixels
            assert offset_error < max_acceptable_error, \
                f"Calculated offset should be close to true offset, error: {offset_error:.2f} pixels"
            
            # Property: Applying calibration should reduce error on reference points
            # Calculate average error before calibration
            errors_before = []
            for gaze_pt, target_pt in ref_obs:
                error = np.linalg.norm(gaze_pt.flatten()[:2] - target_pt.flatten()[:2])
                errors_before.append(error)
            
            mean_error_before = np.mean(errors_before)
            
            # Calculate average error after calibration
            errors_after = []
            for gaze_pt, target_pt in ref_obs:
                calibrated_gaze = gaze_pt.flatten()[:2] + calculated_offset
                error = np.linalg.norm(calibrated_gaze - target_pt.flatten()[:2])
                errors_after.append(error)
            
            mean_error_after = np.mean(errors_after)
            
            # Property: Error after calibration should be less than before
            assert mean_error_after < mean_error_before, \
                f"Calibration should reduce error: before={mean_error_before:.2f}, after={mean_error_after:.2f}"
            
            # Property: Improvement should be substantial (at least 30% reduction)
            improvement_ratio = (mean_error_before - mean_error_after) / mean_error_before
            assert improvement_ratio > 0.3, \
                f"Calibration should substantially reduce error, got {improvement_ratio*100:.1f}% improvement"
            
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    @given(
        ref_data=consistent_reference_observations_strategy(),
        gaze_est=gaze_estimate_strategy()
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_calibration_application_to_new_estimates(self, ref_data, gaze_est):
        """
        Test that calibration can be applied to new gaze estimates.
        
        For any calibration profile and new gaze estimate, the system should
        be able to apply the calibration offset to improve the estimate.
        """
        ref_obs, true_offset = ref_data
        
        tmp_path = tempfile.mkdtemp()
        try:
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            # Create calibration profile
            face_chars = {
                'interpupillary_distance': 60.0,
                'face_width': 150.0,
                'face_height': 180.0,
                'is_child': False
            }
            
            result = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            profile_id = result['profile_id']
            
            # Apply calibration to new gaze estimate
            calibrated_estimate = calibration_system.apply_calibration(
                gaze_est,
                profile_id=profile_id
            )
            
            # Property: Calibrated estimate should have offset applied
            original_point = np.array(gaze_est.gaze_point_2d)
            calibrated_point = np.array(calibrated_estimate.gaze_point_2d)
            applied_offset = calibrated_point - original_point
            
            # The applied offset should be close to the calculated offset
            expected_offset = result['gaze_offset']
            offset_diff = np.linalg.norm(applied_offset - expected_offset)
            
            assert offset_diff < 1.0, \
                f"Applied offset should match calculated offset, difference: {offset_diff:.2f}"
            
            # Property: Calibrated estimate should maintain other properties
            assert calibrated_estimate.gaze_vector_3d is not None
            assert calibrated_estimate.head_pose is not None
            assert calibrated_estimate.timestamp == gaze_est.timestamp
            
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    @given(
        ref_data=consistent_reference_observations_strategy()
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_calibration_quality_reflects_improvement(self, ref_data):
        """
        Test that calibration quality metric reflects actual improvement.
        
        For any reference observations that lead to improvement, the calibration
        quality should be reasonably high.
        """
        ref_obs, true_offset = ref_data
        
        tmp_path = tempfile.mkdtemp()
        try:
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            face_chars = {
                'interpupillary_distance': 60.0,
                'face_width': 150.0,
                'face_height': 180.0,
                'is_child': False
            }
            
            result = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            # Property: If accuracy improvement is significant, quality should be good
            if result['accuracy_improvement'] > 10.0:  # Significant improvement
                assert result['calibration_quality'] > 0.3, \
                    f"With significant improvement ({result['accuracy_improvement']:.2f}), quality should be good"
            
            # Property: Quality should correlate with improvement
            # More improvement should generally mean better quality
            if result['accuracy_improvement'] > 20.0:
                assert result['calibration_quality'] > 0.5, \
                    "Very high improvement should result in high quality score"
            
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
