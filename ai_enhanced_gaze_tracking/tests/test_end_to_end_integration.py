"""
End-to-end integration tests for the AI-Enhanced Gaze Tracking pipeline.

Tests the complete pipeline from video input to gaze output, validating that
all components work together correctly.

Requirements: All requirements through system integration
"""

import time
import numpy as np
import pytest

from ai_enhanced_gaze_tracking.pipeline.gaze_pipeline import GazeTrackingPipeline
from ai_enhanced_gaze_tracking.compatibility.legacy_adapter import (
    LEGACY_FIELDS,
    ENHANCED_FIELDS,
    LegacyResponseAdapter,
)
from ai_enhanced_gaze_tracking.core.data_models import (
    AttentionType,
    FaceDetection,
    GazeEstimate,
    HeadPose,
    QualityMetrics,
    SessionSummary,
)
from ai_enhanced_gaze_tracking.components.face_detection.hybrid_face_detector import (
    HybridFaceDetector,
)
from ai_enhanced_gaze_tracking.components.gaze_estimation.compensated_gaze_estimator import (
    CompensatedGazeEstimator,
)
from ai_enhanced_gaze_tracking.components.head_pose.pnp_head_pose_estimator import (
    PnPHeadPoseEstimator,
)
from ai_enhanced_gaze_tracking.components.sensor_fusion.multi_modal_fusion import (
    MultiModalFusion,
)
from ai_enhanced_gaze_tracking.components.focus_detection.focus_detector import (
    ImprovedFocusDetector,
)
from ai_enhanced_gaze_tracking.components.quality_assessment.quality_assessor import (
    GazeQualityAssessor,
)
from ai_enhanced_gaze_tracking.components.error_handling.error_handler import (
    ErrorHandler,
    SystemState,
)


# ---------------------------------------------------------------------------
# Helpers / synthetic data generators
# ---------------------------------------------------------------------------

def _make_synthetic_frame(
    width: int = 640,
    height: int = 480,
    brightness: int = 128,
) -> np.ndarray:
    """Create a synthetic BGR frame with a simple face-like pattern."""
    frame = np.full((height, width, 3), brightness, dtype=np.uint8)
    # Draw a rough face region so the frame is non-trivial
    cx, cy = width // 2, height // 2
    # Face oval
    for y in range(cy - 80, cy + 80):
        for x in range(cx - 60, cx + 60):
            if 0 <= y < height and 0 <= x < width:
                frame[y, x] = [200, 180, 160]
    return frame


def _make_face_detection(
    confidence: float = 0.9,
    n_landmarks: int = 468,
) -> FaceDetection:
    """Create a synthetic FaceDetection with MediaPipe-style landmarks."""
    landmarks = np.zeros((n_landmarks, 2), dtype=np.float32)
    # Populate key landmark positions realistically
    cx, cy = 320.0, 240.0
    landmarks[1] = [cx, cy - 10]          # Nose tip
    landmarks[152] = [cx, cy + 60]        # Chin
    landmarks[33] = [cx - 40, cy - 20]    # Left eye left corner
    landmarks[263] = [cx + 40, cy - 20]   # Right eye right corner
    landmarks[61] = [cx - 25, cy + 30]    # Left mouth corner
    landmarks[291] = [cx + 25, cy + 30]   # Right mouth corner
    landmarks[159] = [cx - 20, cy - 20]   # Left eye center
    landmarks[386] = [cx + 20, cy - 20]   # Right eye center
    landmarks[13] = [cx, cy + 20]         # Upper lip
    landmarks[14] = [cx, cy + 28]         # Lower lip

    return FaceDetection(
        bbox=(cx - 60, cy - 80, 120, 160),
        landmarks=landmarks,
        confidence=confidence,
        face_id=0,
        is_child=True,
        quality_score=0.9,
    )


def _make_head_pose(yaw: float = 0.0, pitch: float = 0.0) -> HeadPose:
    return HeadPose(
        yaw=yaw,
        pitch=pitch,
        roll=0.0,
        translation=np.array([0.0, 0.0, 500.0]),
        rotation_matrix=np.eye(3),
        confidence=0.85,
    )


def _make_quality_metrics(quality: float = 0.8) -> QualityMetrics:
    return QualityMetrics(
        overall_quality=quality,
        head_pose_quality=quality,
        lighting_quality=quality,
        occlusion_level=0.0,
        motion_blur=0.0,
        tracking_stability=quality,
    )


def _make_gaze_estimate(
    gaze_x: float = 0.0,
    gaze_y: float = 0.0,
    confidence: float = 0.85,
) -> GazeEstimate:
    vec = np.array([gaze_x, gaze_y, 1.0])
    vec /= np.linalg.norm(vec)
    return GazeEstimate(
        gaze_vector_3d=vec,
        gaze_point_2d=(320.0 + gaze_x * 100, 240.0 + gaze_y * 100),
        confidence=confidence,
        head_pose=_make_head_pose(),
        timestamp=time.time(),
        source_confidences={"2d_landmarks": 0.8, "ai_model": 0.9},
        quality_metrics=_make_quality_metrics(),
        method="multi_modal_fusion",
    )


# ---------------------------------------------------------------------------
# 1. Pipeline instantiation
# ---------------------------------------------------------------------------

class TestPipelineInstantiation:
    """The pipeline should instantiate with default components."""

    def test_default_pipeline_creates_successfully(self):
        pipeline = GazeTrackingPipeline()
        assert pipeline is not None

    def test_pipeline_has_all_required_components(self):
        pipeline = GazeTrackingPipeline()
        assert pipeline.face_detector is not None
        assert pipeline.head_pose_estimator is not None
        assert pipeline.gaze_estimator is not None
        assert pipeline.sensor_fusion is not None
        assert pipeline.focus_detector is not None
        assert pipeline.quality_assessor is not None
        assert pipeline.legacy_adapter is not None

    def test_pipeline_accepts_custom_components(self):
        custom_fusion = MultiModalFusion(history_size=5)
        pipeline = GazeTrackingPipeline(sensor_fusion=custom_fusion)
        assert pipeline.sensor_fusion is custom_fusion


# ---------------------------------------------------------------------------
# 2. Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    """start_session / end_session should manage state correctly."""

    def test_start_session_resets_counters(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        assert pipeline._frame_count == 0
        assert pipeline._valid_frame_count == 0
        assert pipeline._gaze_history == []

    def test_end_session_returns_session_summary(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        summary = pipeline.end_session(session_id="test-001")
        assert isinstance(summary, SessionSummary)
        assert summary.session_id == "test-001"
        assert summary.total_frames == 0

    def test_session_summary_timestamps_are_ordered(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        time.sleep(0.01)
        summary = pipeline.end_session()
        assert summary.end_time >= summary.start_time


# ---------------------------------------------------------------------------
# 3. Frame processing — basic contract
# ---------------------------------------------------------------------------

class TestFrameProcessing:
    """process_frame should return a well-formed result dict."""

    def test_process_frame_returns_dict(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        result = pipeline.process_frame(frame)
        assert isinstance(result, dict)

    def test_process_frame_contains_required_keys(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        result = pipeline.process_frame(frame)
        for key in ("timestamp", "frame_index", "gaze_estimate", "face_detected",
                    "focus_event", "quality_metrics", "processing_time_ms"):
            assert key in result, f"Missing key: {key}"

    def test_frame_index_increments(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        r1 = pipeline.process_frame(frame)
        r2 = pipeline.process_frame(frame)
        assert r2["frame_index"] == r1["frame_index"] + 1

    def test_processing_time_is_non_negative(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        result = pipeline.process_frame(frame)
        assert result["processing_time_ms"] >= 0.0

    def test_process_frame_does_not_raise_on_blank_frame(self):
        """Graceful degradation: blank frames must not crash the pipeline."""
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(blank)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 4. Multi-frame session accumulation
# ---------------------------------------------------------------------------

class TestMultiFrameSession:
    """Processing multiple frames should accumulate state correctly."""

    def test_frame_count_accumulates(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        n = 5
        for _ in range(n):
            pipeline.process_frame(frame)
        assert pipeline._frame_count == n

    def test_session_summary_total_frames_matches(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        n = 8
        for _ in range(n):
            pipeline.process_frame(frame)
        summary = pipeline.end_session()
        assert summary.total_frames == n

    def test_session_summary_valid_frames_lte_total(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        for _ in range(6):
            pipeline.process_frame(frame)
        summary = pipeline.end_session()
        assert summary.valid_frames <= summary.total_frames


# ---------------------------------------------------------------------------
# 5. Legacy-compatible output (Requirements 10.1, 10.3)
# ---------------------------------------------------------------------------

class TestLegacyOutput:
    """get_legacy_response must produce all legacy fields."""

    def test_legacy_response_contains_all_legacy_fields(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        for _ in range(3):
            pipeline.process_frame(frame)
        response = pipeline.get_legacy_response(duration=1.0)
        missing = LEGACY_FIELDS - set(response.keys())
        assert not missing, f"Missing legacy fields: {missing}"

    def test_legacy_response_contains_enhanced_fields(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        response = pipeline.get_legacy_response(duration=0.0)
        missing = ENHANCED_FIELDS - set(response.keys())
        assert not missing, f"Missing enhanced fields: {missing}"

    def test_legacy_response_total_frames_is_int(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        frame = _make_synthetic_frame()
        pipeline.process_frame(frame)
        response = pipeline.get_legacy_response(duration=1.0)
        assert isinstance(response["total_frames"], int)

    def test_legacy_response_numeric_fields_are_numeric(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()
        response = pipeline.get_legacy_response(duration=0.0)
        for field in ("eye_contact_percentage", "analyzed_duration", "risk_score"):
            assert isinstance(response[field], (int, float)), (
                f"Field '{field}' should be numeric"
            )


# ---------------------------------------------------------------------------
# 6. Component integration — gaze estimator + head pose
# ---------------------------------------------------------------------------

class TestGazeEstimatorIntegration:
    """CompensatedGazeEstimator should produce valid GazeEstimate objects."""

    def test_gaze_estimate_has_unit_vector(self):
        head_pose_est = PnPHeadPoseEstimator()
        gaze_est = CompensatedGazeEstimator(head_pose_estimator=head_pose_est)
        face = _make_face_detection()
        frame = _make_synthetic_frame()
        result = gaze_est.estimate_gaze(face, frame)
        norm = np.linalg.norm(result.gaze_vector_3d)
        assert abs(norm - 1.0) < 0.01, f"Gaze vector not unit: norm={norm}"

    def test_gaze_estimate_confidence_in_range(self):
        gaze_est = CompensatedGazeEstimator()
        face = _make_face_detection()
        frame = _make_synthetic_frame()
        result = gaze_est.estimate_gaze(face, frame)
        assert 0.0 <= result.confidence <= 1.0

    def test_gaze_estimate_has_quality_metrics(self):
        gaze_est = CompensatedGazeEstimator()
        face = _make_face_detection()
        frame = _make_synthetic_frame()
        result = gaze_est.estimate_gaze(face, frame)
        assert result.quality_metrics is not None


# ---------------------------------------------------------------------------
# 7. Sensor fusion integration
# ---------------------------------------------------------------------------

class TestSensorFusionIntegration:
    """MultiModalFusion should combine estimates correctly."""

    def test_fusion_returns_single_estimate(self):
        fusion = MultiModalFusion()
        estimates = [_make_gaze_estimate(0.1), _make_gaze_estimate(-0.1)]
        confidences = [0.8, 0.7]
        result = fusion.fuse_estimates(estimates, confidences)
        assert isinstance(result, GazeEstimate)

    def test_fused_vector_is_unit(self):
        fusion = MultiModalFusion()
        estimates = [_make_gaze_estimate(0.2), _make_gaze_estimate(-0.2)]
        confidences = [0.9, 0.6]
        result = fusion.fuse_estimates(estimates, confidences)
        norm = np.linalg.norm(result.gaze_vector_3d)
        assert abs(norm - 1.0) < 0.01

    def test_fusion_confidence_in_range(self):
        fusion = MultiModalFusion()
        estimates = [_make_gaze_estimate(), _make_gaze_estimate(0.1)]
        confidences = [0.8, 0.7]
        result = fusion.fuse_estimates(estimates, confidences)
        assert 0.0 <= result.confidence <= 1.0

    def test_single_estimate_passthrough(self):
        fusion = MultiModalFusion()
        est = _make_gaze_estimate(0.3)
        result = fusion.fuse_estimates([est], [0.9])
        # Direction should be preserved
        dot = np.dot(result.gaze_vector_3d, est.gaze_vector_3d)
        assert dot > 0.99


# ---------------------------------------------------------------------------
# 8. Quality assessor integration
# ---------------------------------------------------------------------------

class TestQualityAssessorIntegration:
    """GazeQualityAssessor should produce valid QualityMetrics."""

    def test_quality_metrics_all_in_range(self):
        assessor = GazeQualityAssessor()
        gaze = _make_gaze_estimate()
        frame = _make_synthetic_frame()
        face = _make_face_detection()
        metrics = assessor.assess_quality(gaze, frame, face)
        for attr in ("overall_quality", "head_pose_quality", "lighting_quality",
                     "occlusion_level", "motion_blur", "tracking_stability"):
            val = getattr(metrics, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_flag_unreliable_returns_indices(self):
        assessor = GazeQualityAssessor(quality_threshold=0.5)
        good = _make_quality_metrics(0.9)
        bad = _make_quality_metrics(0.1)
        flagged = assessor.flag_unreliable_data([good, bad, good, bad])
        assert set(flagged) == {1, 3}

    def test_validate_data_segment_length_matches(self):
        assessor = GazeQualityAssessor()
        estimates = [_make_gaze_estimate() for _ in range(5)]
        valid = assessor.validate_data_segment(estimates)
        assert len(valid) == 5


# ---------------------------------------------------------------------------
# 9. Focus detector integration
# ---------------------------------------------------------------------------

class TestFocusDetectorIntegration:
    """ImprovedFocusDetector should integrate with gaze estimates."""

    def test_no_focus_without_stable_gaze(self):
        detector = ImprovedFocusDetector(min_focus_duration=1.0)
        # Single frame — not enough for stable focus
        gaze_vec = np.array([0.0, 0.0, 1.0])
        result = detector.detect_focus(gaze_vec, [], {})
        # With only one data point, focus should not be detected
        assert result is None

    def test_focus_not_classified_as_object_without_objects(self):
        """Requirement 6.2: no object focus when no objects present."""
        detector = ImprovedFocusDetector(min_focus_duration=0.01)
        gaze_vec = np.array([0.0, 0.0, 1.0])
        # Feed many frames with stable gaze but no objects
        for _ in range(20):
            result = detector.detect_focus(gaze_vec, [], {})
        # If focus is detected, it must NOT be object type
        if result is not None:
            assert result.focus_type != AttentionType.OBJECT

    def test_attention_shift_tracking_returns_dict(self):
        from ai_enhanced_gaze_tracking.core.data_models import FocusEvent
        detector = ImprovedFocusDetector()
        events = [
            FocusEvent("obj_a", AttentionType.OBJECT, 0.0, 1.0, 0.9, 0.8),
            FocusEvent("obj_b", AttentionType.OBJECT, 1.5, 1.0, 0.9, 0.8),
        ]
        analysis = detector.track_attention_shifts(events)
        assert isinstance(analysis, dict)
        assert "total_shifts" in analysis
        assert analysis["total_shifts"] == 1


# ---------------------------------------------------------------------------
# 10. Error handler integration
# ---------------------------------------------------------------------------

class TestErrorHandlerIntegration:
    """ErrorHandler should integrate with the pipeline for graceful degradation."""

    def test_normal_state_on_init(self):
        handler = ErrorHandler()
        assert handler.get_system_state() == SystemState.NORMAL

    def test_resource_exhaustion_degrades_gracefully(self):
        """Property 20: system must not crash under resource exhaustion."""
        handler = ErrorHandler()
        # Simulate critical memory usage
        state = handler.handle_resource_exhaustion(memory_mb=4000.0)
        assert state in (SystemState.DEGRADED, SystemState.MINIMAL)
        # System is still operational (not FAILED)
        assert state != SystemState.FAILED

    def test_component_failure_does_not_raise(self):
        handler = ErrorHandler()
        try:
            handler.report_failure("face_detection", RuntimeError("test error"))
        except Exception as exc:
            pytest.fail(f"report_failure raised unexpectedly: {exc}")

    def test_diagnostics_returns_dict(self):
        handler = ErrorHandler()
        handler.report_failure("ai_model", RuntimeError("model error"))
        diag = handler.get_diagnostics()
        assert isinstance(diag, dict)
        assert "system_state" in diag
        assert "component_health" in diag

    def test_user_guidance_is_string(self):
        handler = ErrorHandler()
        guidance = handler.get_user_guidance()
        assert isinstance(guidance, str)
        assert len(guidance) > 0


# ---------------------------------------------------------------------------
# 11. Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestFullPipelineSmokeTest:
    """
    Smoke test: run a short synthetic session through the complete pipeline
    and verify the output is coherent.
    """

    def test_full_session_produces_valid_summary(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()

        frame = _make_synthetic_frame()
        n_frames = 10
        for i in range(n_frames):
            pipeline.process_frame(frame, timestamp=float(i) / 30.0)

        summary = pipeline.end_session(session_id="smoke-test")

        assert summary.total_frames == n_frames
        assert summary.valid_frames <= summary.total_frames
        assert 0.0 <= summary.overall_quality <= 1.0
        assert summary.end_time >= summary.start_time

    def test_full_session_legacy_response_is_complete(self):
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()

        frame = _make_synthetic_frame()
        for _ in range(5):
            pipeline.process_frame(frame)

        response = pipeline.get_legacy_response(duration=5.0 / 30.0)

        missing_legacy = LEGACY_FIELDS - set(response.keys())
        missing_enhanced = ENHANCED_FIELDS - set(response.keys())
        assert not missing_legacy, f"Missing legacy fields: {missing_legacy}"
        assert not missing_enhanced, f"Missing enhanced fields: {missing_enhanced}"

    def test_pipeline_handles_mixed_quality_frames(self):
        """Pipeline should not crash when alternating good and bad frames."""
        pipeline = GazeTrackingPipeline()
        pipeline.start_session()

        good_frame = _make_synthetic_frame(brightness=128)
        dark_frame = _make_synthetic_frame(brightness=10)
        bright_frame = _make_synthetic_frame(brightness=250)

        for frame in [good_frame, dark_frame, bright_frame] * 3:
            result = pipeline.process_frame(frame)
            assert isinstance(result, dict)

        summary = pipeline.end_session()
        assert summary.total_frames == 9

    def test_pipeline_restart_clears_state(self):
        """Starting a new session should clear all previous state."""
        pipeline = GazeTrackingPipeline()

        # First session
        pipeline.start_session()
        frame = _make_synthetic_frame()
        for _ in range(4):
            pipeline.process_frame(frame)
        assert pipeline._frame_count == 4

        # Second session — state must be reset
        pipeline.start_session()
        assert pipeline._frame_count == 0
        assert pipeline._gaze_history == []
