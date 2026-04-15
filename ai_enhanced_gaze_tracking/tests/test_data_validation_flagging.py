"""
Property-based tests for data validation and unreliable segment flagging.

**Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
**Validates: Requirements 9.5**

Property: For any potentially unreliable data segment, the validation system
should correctly identify and flag it based on quality metrics.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from ai_enhanced_gaze_tracking.components.quality_assessment.quality_assessor import GazeQualityAssessor
from ai_enhanced_gaze_tracking.core.data_models import (
    GazeEstimate, HeadPose, QualityMetrics, FaceDetection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_quality_metrics(overall: float, **kwargs) -> QualityMetrics:
    """Create a QualityMetrics with a given overall_quality."""
    return QualityMetrics(
        overall_quality=overall,
        head_pose_quality=kwargs.get("head_pose_quality", overall),
        lighting_quality=kwargs.get("lighting_quality", overall),
        occlusion_level=kwargs.get("occlusion_level", 1.0 - overall),
        motion_blur=kwargs.get("motion_blur", 1.0 - overall),
        tracking_stability=kwargs.get("tracking_stability", overall),
        eye_visibility=kwargs.get("eye_visibility", overall),
        landmark_quality=kwargs.get("landmark_quality", overall),
        temporal_consistency=kwargs.get("temporal_consistency", overall),
    )


def make_gaze_estimate(quality_score: float) -> GazeEstimate:
    """Create a minimal GazeEstimate with the given overall quality."""
    head_pose = HeadPose(
        yaw=0.0, pitch=0.0, roll=0.0,
        translation=np.zeros(3),
        rotation_matrix=np.eye(3),
        confidence=quality_score,
    )
    return GazeEstimate(
        gaze_vector_3d=np.array([0.0, 0.0, 1.0]),
        gaze_point_2d=(320.0, 240.0),
        confidence=quality_score,
        head_pose=head_pose,
        timestamp=0.0,
        source_confidences={},
        quality_metrics=make_quality_metrics(quality_score),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataValidationFlagging:
    """
    Tests for Property 23: Data Validation Flagging.

    **Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
    **Validates: Requirements 9.5**
    """

    def setup_method(self):
        self.assessor = GazeQualityAssessor(quality_threshold=0.4)

    # --- Core property: low-quality segments are flagged ---

    @given(
        quality_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=2000)
    def test_unreliable_segments_are_flagged(self, quality_scores):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
        **Validates: Requirements 9.5**

        For any sequence of quality metrics, every index whose overall_quality
        is below the threshold must appear in the flagged list.
        """
        threshold = self.assessor.quality_threshold
        metrics_list = [make_quality_metrics(q) for q in quality_scores]

        flagged = self.assessor.flag_unreliable_data(metrics_list)

        # Every flagged index must have quality below threshold
        for idx in flagged:
            assert metrics_list[idx].overall_quality < threshold, (
                f"Index {idx} was flagged but quality "
                f"({metrics_list[idx].overall_quality:.3f}) >= threshold ({threshold})"
            )

        # Every index below threshold must be flagged
        expected_flagged = {
            i for i, qm in enumerate(metrics_list)
            if qm.overall_quality < threshold
        }
        assert set(flagged) == expected_flagged, (
            f"Flagged set {set(flagged)} does not match expected {expected_flagged}"
        )

    @given(
        quality_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100, deadline=2000)
    def test_reliable_segments_are_not_flagged(self, quality_scores):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
        **Validates: Requirements 9.5**

        For any sequence of quality metrics, no index whose overall_quality
        is at or above the threshold should appear in the flagged list.
        """
        threshold = self.assessor.quality_threshold
        metrics_list = [make_quality_metrics(q) for q in quality_scores]

        flagged = set(self.assessor.flag_unreliable_data(metrics_list))

        for i, qm in enumerate(metrics_list):
            if qm.overall_quality >= threshold:
                assert i not in flagged, (
                    f"Index {i} should NOT be flagged (quality={qm.overall_quality:.3f} "
                    f">= threshold={threshold}), but it was"
                )

    @given(
        quality_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=50,
        ),
        threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=100, deadline=2000)
    def test_flagging_consistent_with_validate_data_segment(self, quality_scores, threshold):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
        **Validates: Requirements 9.5**

        flag_unreliable_data and validate_data_segment must be consistent:
        an index flagged as unreliable must correspond to a False in the
        validation mask, and vice versa.
        """
        assessor = GazeQualityAssessor(quality_threshold=threshold)
        metrics_list = [make_quality_metrics(q) for q in quality_scores]
        gaze_sequence = [make_gaze_estimate(q) for q in quality_scores]

        flagged = set(assessor.flag_unreliable_data(metrics_list))
        valid_mask = assessor.validate_data_segment(gaze_sequence, quality_threshold=threshold)

        for i in range(len(quality_scores)):
            if i in flagged:
                assert not valid_mask[i], (
                    f"Index {i} is flagged as unreliable but validate_data_segment "
                    f"returned True (quality={quality_scores[i]:.3f}, threshold={threshold:.3f})"
                )
            else:
                assert valid_mask[i], (
                    f"Index {i} is not flagged but validate_data_segment "
                    f"returned False (quality={quality_scores[i]:.3f}, threshold={threshold:.3f})"
                )

    @given(
        n=st.integers(min_value=1, max_value=30),
        threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50, deadline=2000)
    def test_all_high_quality_yields_no_flags(self, n, threshold):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
        **Validates: Requirements 9.5**

        When all quality scores are above the threshold, no segments should be flagged.
        """
        # Scores strictly above threshold
        high_quality = threshold + (1.0 - threshold) * 0.5  # midpoint above threshold
        assume(high_quality > threshold)

        assessor = GazeQualityAssessor(quality_threshold=threshold)
        metrics_list = [make_quality_metrics(high_quality) for _ in range(n)]

        flagged = assessor.flag_unreliable_data(metrics_list)
        assert flagged == [], (
            f"No segments should be flagged when all quality={high_quality:.3f} "
            f"> threshold={threshold:.3f}, but got flagged={flagged}"
        )

    @given(
        n=st.integers(min_value=1, max_value=30),
        threshold=st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=50, deadline=2000)
    def test_all_low_quality_flags_all(self, n, threshold):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 23: Data Validation Flagging**
        **Validates: Requirements 9.5**

        When all quality scores are below the threshold, all segments should be flagged.
        """
        low_quality = threshold * 0.5  # midpoint below threshold
        assume(low_quality < threshold)

        assessor = GazeQualityAssessor(quality_threshold=threshold)
        metrics_list = [make_quality_metrics(low_quality) for _ in range(n)]

        flagged = assessor.flag_unreliable_data(metrics_list)
        assert len(flagged) == n, (
            f"All {n} segments should be flagged when quality={low_quality:.3f} "
            f"< threshold={threshold:.3f}, but only {len(flagged)} were flagged"
        )
        assert sorted(flagged) == list(range(n)), (
            f"Expected all indices 0..{n-1} to be flagged, got {flagged}"
        )

    def test_empty_sequence_returns_empty(self):
        """
        Edge case: empty input should return empty flagged list and empty valid mask.
        """
        assert self.assessor.flag_unreliable_data([]) == []
        assert self.assessor.validate_data_segment([]) == []
