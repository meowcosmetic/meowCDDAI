"""
Property-based tests for confidence score completeness.

**Feature: ai-enhanced-gaze-tracking, Property 21: Confidence Score Completeness**
**Validates: Requirements 9.1**

Property: For any gaze estimate generated, the system should provide confidence
scores that reflect the reliability of the estimate.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from ai_enhanced_gaze_tracking.components.quality_assessment.quality_assessor import GazeQualityAssessor
from ai_enhanced_gaze_tracking.core.data_models import (
    GazeEstimate, HeadPose, QualityMetrics, FaceDetection,
)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def head_pose_strategy(draw):
    """Generate a HeadPose with random but bounded angles."""
    yaw = draw(st.floats(min_value=-1.0, max_value=1.0))
    pitch = draw(st.floats(min_value=-1.0, max_value=1.0))
    roll = draw(st.floats(min_value=-0.5, max_value=0.5))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    return HeadPose(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        translation=np.zeros(3),
        rotation_matrix=np.eye(3),
        confidence=confidence,
    )


@st.composite
def face_detection_strategy(draw):
    """Generate a FaceDetection with random quality and confidence."""
    confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    quality_score = draw(st.floats(min_value=0.0, max_value=1.0))
    # Minimal landmarks array (468 points)
    landmarks = np.zeros((468, 2), dtype=np.float32)
    # Place a few eye landmarks inside frame bounds
    for idx in [33, 133, 159, 145, 362, 263, 386, 374]:
        landmarks[idx] = [
            draw(st.floats(min_value=1.0, max_value=639.0)),
            draw(st.floats(min_value=1.0, max_value=479.0)),
        ]
    return FaceDetection(
        bbox=(100.0, 100.0, 200.0, 200.0),
        landmarks=landmarks,
        confidence=confidence,
        quality_score=quality_score,
    )


@st.composite
def gaze_estimate_strategy(draw, head_pose=None, quality_metrics=None):
    """Generate a GazeEstimate with random but valid fields."""
    hp = head_pose or draw(head_pose_strategy())
    qm = quality_metrics or QualityMetrics(
        overall_quality=draw(st.floats(min_value=0.0, max_value=1.0)),
        head_pose_quality=draw(st.floats(min_value=0.0, max_value=1.0)),
        lighting_quality=draw(st.floats(min_value=0.0, max_value=1.0)),
        occlusion_level=draw(st.floats(min_value=0.0, max_value=1.0)),
        motion_blur=draw(st.floats(min_value=0.0, max_value=1.0)),
        tracking_stability=draw(st.floats(min_value=0.0, max_value=1.0)),
    )
    gaze_vec = np.array([
        draw(st.floats(min_value=-1.0, max_value=1.0)),
        draw(st.floats(min_value=-1.0, max_value=1.0)),
        draw(st.floats(min_value=0.1, max_value=1.0)),
    ])
    return GazeEstimate(
        gaze_vector_3d=gaze_vec,
        gaze_point_2d=(
            draw(st.floats(min_value=0.0, max_value=1280.0)),
            draw(st.floats(min_value=0.0, max_value=720.0)),
        ),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        head_pose=hp,
        timestamp=draw(st.floats(min_value=0.0, max_value=1e6)),
        source_confidences={},
        quality_metrics=qm,
    )


def make_frame(brightness: float = 128.0) -> np.ndarray:
    """Create a simple synthetic BGR frame."""
    val = int(np.clip(brightness, 0, 255))
    return np.full((480, 640, 3), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConfidenceScoreCompleteness:
    """
    Tests for Property 21: Confidence Score Completeness.

    **Feature: ai-enhanced-gaze-tracking, Property 21: Confidence Score Completeness**
    **Validates: Requirements 9.1**
    """

    def setup_method(self):
        self.assessor = GazeQualityAssessor(quality_threshold=0.4)

    # --- Property 21 core: every estimate gets a confidence score in [0, 1] ---

    @given(
        gaze_est=gaze_estimate_strategy(),
        face_det=face_detection_strategy(),
        brightness=st.floats(min_value=10.0, max_value=245.0),
    )
    @settings(max_examples=100, deadline=3000)
    def test_every_estimate_has_confidence_score(self, gaze_est, face_det, brightness):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 21: Confidence Score Completeness**
        **Validates: Requirements 9.1**

        For any gaze estimate, assess_quality must return a QualityMetrics whose
        overall_quality is a valid float in [0, 1].
        """
        frame = make_frame(brightness)
        metrics = self.assessor.assess_quality(gaze_est, frame, face_det)

        # The confidence/quality score must exist and be in range
        assert metrics is not None, "assess_quality must return a QualityMetrics object"
        assert isinstance(metrics.overall_quality, float), \
            "overall_quality must be a float"
        assert 0.0 <= metrics.overall_quality <= 1.0, (
            f"overall_quality must be in [0, 1], got {metrics.overall_quality}"
        )

    @given(
        gaze_est=gaze_estimate_strategy(),
        face_det=face_detection_strategy(),
        brightness=st.floats(min_value=10.0, max_value=245.0),
    )
    @settings(max_examples=100, deadline=3000)
    def test_all_quality_dimensions_present_and_valid(self, gaze_est, face_det, brightness):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 21: Confidence Score Completeness**
        **Validates: Requirements 9.1**

        For any gaze estimate, all quality dimension scores must be present
        and in [0, 1].
        """
        frame = make_frame(brightness)
        metrics = self.assessor.assess_quality(gaze_est, frame, face_det)

        dimensions = {
            "overall_quality": metrics.overall_quality,
            "head_pose_quality": metrics.head_pose_quality,
            "lighting_quality": metrics.lighting_quality,
            "occlusion_level": metrics.occlusion_level,
            "motion_blur": metrics.motion_blur,
            "tracking_stability": metrics.tracking_stability,
            "eye_visibility": metrics.eye_visibility,
            "landmark_quality": metrics.landmark_quality,
            "temporal_consistency": metrics.temporal_consistency,
        }

        for name, value in dimensions.items():
            assert value is not None, f"{name} must not be None"
            assert isinstance(value, float), f"{name} must be a float, got {type(value)}"
            assert 0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}"

    @given(
        gaze_est=gaze_estimate_strategy(),
        face_det=face_detection_strategy(),
        brightness=st.floats(min_value=10.0, max_value=245.0),
    )
    @settings(max_examples=100, deadline=3000)
    def test_confidence_reflects_reliability(self, gaze_est, face_det, brightness):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 21: Confidence Score Completeness**
        **Validates: Requirements 9.1**

        For any gaze estimate, the overall_quality score must be consistent
        with the individual quality dimensions — it cannot be higher than the
        maximum possible given the inputs.
        """
        frame = make_frame(brightness)
        metrics = self.assessor.assess_quality(gaze_est, frame, face_det)

        # If all individual dimensions are low, overall must also be low
        individual_scores = [
            metrics.head_pose_quality,
            metrics.lighting_quality,
            1.0 - metrics.occlusion_level,   # occlusion is a penalty
            1.0 - metrics.motion_blur,        # blur is a penalty
            metrics.eye_visibility,
            metrics.landmark_quality,
        ]
        max_possible = max(individual_scores)

        # Overall quality cannot exceed the best individual dimension by more than 0.1
        assert metrics.overall_quality <= max_possible + 0.1, (
            f"overall_quality ({metrics.overall_quality:.3f}) should not greatly exceed "
            f"the best individual dimension ({max_possible:.3f})"
        )

    @given(
        face_det=face_detection_strategy(),
        brightness=st.floats(min_value=10.0, max_value=245.0),
    )
    @settings(max_examples=50, deadline=3000)
    def test_high_confidence_face_yields_nonzero_quality(self, face_det, brightness):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 21: Confidence Score Completeness**
        **Validates: Requirements 9.1**

        For a face detection with high confidence and quality, the resulting
        quality score should be above zero.
        """
        # Force high-quality face detection
        high_conf_face = FaceDetection(
            bbox=(100.0, 100.0, 200.0, 200.0),
            landmarks=face_det.landmarks,
            confidence=0.95,
            quality_score=0.95,
        )
        head_pose = HeadPose(
            yaw=0.0, pitch=0.0, roll=0.0,
            translation=np.zeros(3),
            rotation_matrix=np.eye(3),
            confidence=0.95,
        )
        qm = QualityMetrics(
            overall_quality=0.9, head_pose_quality=0.9,
            lighting_quality=0.9, occlusion_level=0.1,
            motion_blur=0.1, tracking_stability=0.9,
        )
        gaze_est = GazeEstimate(
            gaze_vector_3d=np.array([0.0, 0.0, 1.0]),
            gaze_point_2d=(320.0, 240.0),
            confidence=0.9,
            head_pose=head_pose,
            timestamp=0.0,
            source_confidences={},
            quality_metrics=qm,
        )
        frame = make_frame(brightness)
        metrics = self.assessor.assess_quality(gaze_est, frame, high_conf_face)

        assert metrics.overall_quality > 0.0, (
            "High-confidence face should yield non-zero quality score"
        )
