"""
Property-based tests for automatic parameter adaptation.

**Feature: ai-enhanced-gaze-tracking, Property 11: Automatic Parameter Adaptation**
**Validates: Requirements 5.1**

Property 11: Automatic Parameter Adaptation
For any new face detected, the calibration system should automatically adjust 
parameters based on face characteristics and improve accuracy over baseline.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from ai_enhanced_gaze_tracking.components.calibration.personal_calibration import (
    PersonalCalibrationSystem,
    PersonalCalibrationProfile
)


# Strategies for generating test data
@st.composite
def face_characteristics_strategy(draw):
    """Generate valid face characteristics."""
    return {
        'interpupillary_distance': draw(st.floats(min_value=40.0, max_value=75.0)),
        'face_width': draw(st.floats(min_value=100.0, max_value=200.0)),
        'face_height': draw(st.floats(min_value=120.0, max_value=250.0)),
        'eye_aspect_ratio': draw(st.floats(min_value=0.2, max_value=0.4)),
        'is_child': draw(st.booleans())
    }


@st.composite
def reference_observations_strategy(draw):
    """Generate reference gaze-target observation pairs."""
    num_points = draw(st.integers(min_value=3, max_value=15))
    observations = []
    
    for _ in range(num_points):
        # Gaze point (with some error)
        gaze_x = draw(st.floats(min_value=0.0, max_value=1920.0))
        gaze_y = draw(st.floats(min_value=0.0, max_value=1080.0))
        gaze_point = np.array([gaze_x, gaze_y])
        
        # Target point (ground truth)
        # Add realistic gaze error (typically 20-100 pixels)
        error_x = draw(st.floats(min_value=-100.0, max_value=100.0))
        error_y = draw(st.floats(min_value=-100.0, max_value=100.0))
        target_point = np.array([gaze_x - error_x, gaze_y - error_y])
        
        observations.append((gaze_point, target_point))
    
    return observations


class TestParameterAdaptation:
    """Test automatic parameter adaptation property."""
    
    @given(
        face_chars=face_characteristics_strategy(),
        ref_obs=reference_observations_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_parameter_adaptation_improves_accuracy(self, face_chars, ref_obs):
        """
        Property 11: Automatic Parameter Adaptation
        
        For any new face detected with valid characteristics and reference observations,
        the calibration system should automatically adjust parameters and improve
        accuracy over baseline.
        
        This test verifies that:
        1. Parameters are automatically adapted based on face characteristics
        2. Accuracy improvement is measurable and positive
        3. Calibration quality is assessed and reasonable
        """
        # Create temporary directory for this test
        tmp_path = tempfile.mkdtemp()
        try:
            # Create calibration system with temporary storage
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            # Perform auto-calibration
            calibration_result = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            # Verify that calibration returns expected structure
            assert 'profile_id' in calibration_result
            assert 'gaze_offset' in calibration_result
            assert 'accuracy_improvement' in calibration_result
            assert 'calibration_quality' in calibration_result
            
            # Verify profile was created
            profile_id = calibration_result['profile_id']
            assert profile_id in calibration_system.profiles
            
            profile = calibration_system.profiles[profile_id]
            
            # Property: Parameters should be adapted based on face characteristics
            assert len(profile.face_characteristics) > 0, \
                "Face characteristics should be extracted and stored"
            
            # Property: Gaze offset should be calculated from reference observations
            assert profile.gaze_offset.shape == (2,), \
                "Gaze offset should be 2D vector"
            
            # Property: Accuracy improvement should be non-negative
            # (calibration should not make things worse)
            assert calibration_result['accuracy_improvement'] >= 0.0, \
                f"Calibration should not decrease accuracy, got improvement: {calibration_result['accuracy_improvement']}"
            
            # Property: With sufficient reference points and consistent error, accuracy should improve
            if len(ref_obs) >= 5:
                # Check if errors are truly consistent (points have similar error vectors)
                offsets = [
                    target_pt.flatten()[:2] - gaze_pt.flatten()[:2]
                    for gaze_pt, target_pt in ref_obs
                ]
                offsets_array = np.array(offsets)
                
                # Filter to only offsets with meaningful magnitude
                significant_offsets = [offset for offset in offsets_array if np.linalg.norm(offset) > 0.5]
                
                if len(significant_offsets) >= 3:
                    # Check if significant offsets are consistent with each other
                    # Calculate pairwise consistency
                    consistent_pairs = 0
                    total_pairs = 0
                    
                    for i in range(len(significant_offsets)):
                        for j in range(i + 1, len(significant_offsets)):
                            offset1 = significant_offsets[i]
                            offset2 = significant_offsets[j]
                            # Offsets are consistent if they point in similar direction
                            # (dot product is positive and magnitudes are similar)
                            dot_prod = np.dot(offset1, offset2)
                            norm_prod = np.linalg.norm(offset1) * np.linalg.norm(offset2)
                            if norm_prod > 0:
                                similarity = dot_prod / norm_prod
                                if similarity > 0.5:  # Cosine similarity > 0.5
                                    consistent_pairs += 1
                                total_pairs += 1
                    
                    # If majority of pairs are consistent, we expect improvement
                    if total_pairs > 0 and consistent_pairs / total_pairs >= 0.7:
                        assert calibration_result['accuracy_improvement'] > 0.0, \
                            f"With highly consistent error pattern ({consistent_pairs}/{total_pairs} pairs consistent), calibration should improve accuracy"
            
            # Property: Calibration quality should be in valid range [0, 1]
            assert 0.0 <= calibration_result['calibration_quality'] <= 1.0, \
                f"Calibration quality should be in [0, 1], got {calibration_result['calibration_quality']}"
            
            # Property: More reference points with actual improvement should lead to better calibration quality
            if len(ref_obs) >= 9 and calibration_result['accuracy_improvement'] > 0:
                assert calibration_result['calibration_quality'] >= 0.3, \
                    "With many reference points and positive improvement, calibration quality should be reasonable"
        finally:
            # Clean up temporary directory
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    @given(
        face_chars=face_characteristics_strategy(),
        ref_obs=reference_observations_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_parameter_adaptation_consistency(self, face_chars, ref_obs):
        """
        Test that parameter adaptation is consistent for the same face.
        
        For any face characteristics, repeated calibration with the same data
        should produce consistent results.
        """
        tmp_path = tempfile.mkdtemp()
        try:
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            # First calibration
            result1 = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            # Second calibration with same data
            result2 = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            # Property: Same face should get same profile ID
            assert result1['profile_id'] == result2['profile_id'], \
                "Same face characteristics should produce same profile ID"
            
            # Property: Gaze offset should be consistent
            offset_diff = np.linalg.norm(result1['gaze_offset'] - result2['gaze_offset'])
            assert offset_diff < 1.0, \
                f"Gaze offset should be consistent, difference: {offset_diff}"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    @given(
        face_chars=face_characteristics_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_parameter_adaptation_without_references(self, face_chars):
        """
        Test parameter adaptation with insufficient reference observations.
        
        For any face characteristics with insufficient reference data,
        the system should still create a profile but with zero improvement.
        """
        tmp_path = tempfile.mkdtemp()
        try:
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            # Calibrate with no reference observations
            result = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=[]
            )
            
            # Property: Profile should still be created
            assert 'profile_id' in result
            assert result['profile_id'] in calibration_system.profiles
            
            # Property: Without reference data, improvement should be zero
            assert result['accuracy_improvement'] == 0.0, \
                "Without reference observations, accuracy improvement should be zero"
            
            # Property: Gaze offset should be zero without reference data
            assert np.allclose(result['gaze_offset'], np.zeros(2)), \
                "Without reference observations, gaze offset should be zero"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    @given(
        face_chars=face_characteristics_strategy(),
        ref_obs=reference_observations_strategy()
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_head_pose_bias_adaptation(self, face_chars, ref_obs):
        """
        Test that head pose biases are adapted based on face characteristics.
        
        For any face characteristics, the system should calculate appropriate
        head pose biases (e.g., children vs adults).
        """
        tmp_path = tempfile.mkdtemp()
        try:
            calibration_system = PersonalCalibrationSystem(storage_path=tmp_path)
            
            result = calibration_system.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            profile = calibration_system.profiles[result['profile_id']]
            
            # Property: Head pose biases should be calculated
            assert 'head_pose_bias' in result
            assert isinstance(result['head_pose_bias'], dict)
            
            # Property: Biases should include yaw, pitch, roll
            assert 'yaw' in result['head_pose_bias']
            assert 'pitch' in result['head_pose_bias']
            assert 'roll' in result['head_pose_bias']
            
            # Property: Children should have different biases than adults
            is_child = face_chars.get('is_child', False)
            if is_child:
                # Children tend to look upward at screens
                assert result['head_pose_bias']['pitch'] != 0.0, \
                    "Children should have non-zero pitch bias"
            else:
                # Adults typically have neutral biases
                assert result['head_pose_bias']['pitch'] == 0.0, \
                    "Adults should have zero pitch bias"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)
    
    @given(
        face_chars=face_characteristics_strategy(),
        ref_obs=reference_observations_strategy()
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_profile_persistence(self, face_chars, ref_obs):
        """
        Test that calibration profiles are persisted and can be reloaded.
        
        For any calibration, the profile should be saved to disk and
        recoverable in a new session.
        """
        tmp_path = tempfile.mkdtemp()
        try:
            # Create first calibration system
            calibration_system1 = PersonalCalibrationSystem(storage_path=tmp_path)
            
            result = calibration_system1.auto_calibrate(
                face_characteristics=face_chars,
                reference_observations=ref_obs
            )
            
            profile_id = result['profile_id']
            original_offset = result['gaze_offset'].copy()
            
            # Create new calibration system (simulating new session)
            calibration_system2 = PersonalCalibrationSystem(storage_path=tmp_path)
            
            # Property: Profile should be loaded from disk
            assert profile_id in calibration_system2.profiles, \
                "Profile should be persisted and reloaded"
            
            # Property: Loaded profile should have same parameters
            loaded_profile = calibration_system2.profiles[profile_id]
            assert np.allclose(loaded_profile.gaze_offset, original_offset), \
                "Loaded profile should have same gaze offset"
            
            assert loaded_profile.face_characteristics == \
                   calibration_system1.profiles[profile_id].face_characteristics, \
                "Loaded profile should have same face characteristics"
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
