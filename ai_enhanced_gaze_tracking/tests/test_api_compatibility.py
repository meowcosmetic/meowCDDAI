"""
Integration tests for backward API compatibility.

Verifies that:
- All legacy GazeAnalysisResponse fields are present in enhanced output (Req 10.1, 10.3)
- Legacy configuration parameters are accepted alongside new ones (Req 10.2)
- Enhanced fields are added on top of the legacy schema without removing anything

Requirements: 10.1, 10.2, 10.3
"""

import time
import pytest
import numpy as np

from ai_enhanced_gaze_tracking.compatibility.legacy_adapter import (
    LegacyResponseAdapter,
    build_legacy_response,
    LEGACY_FIELDS,
    ENHANCED_FIELDS,
)
from ai_enhanced_gaze_tracking.compatibility.config_bridge import (
    ConfigBridge,
    LEGACY_TO_ENHANCED,
)
from ai_enhanced_gaze_tracking.config import EnhancedGazeConfig
from ai_enhanced_gaze_tracking.core.data_models import (
    AttentionType,
    FocusEvent,
    GazeEstimate,
    HeadPose,
    QualityMetrics,
    SessionSummary,
    WanderingPeriod,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_head_pose(yaw: float = 0.0, pitch: float = 0.0) -> HeadPose:
    return HeadPose(
        yaw=yaw,
        pitch=pitch,
        roll=0.0,
        translation=np.array([0.0, 0.0, 500.0]),
        rotation_matrix=np.eye(3),
        confidence=0.9,
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
    method: str = "multi_modal_fusion",
) -> GazeEstimate:
    gaze_vec = np.array([gaze_x, gaze_y, 1.0])
    gaze_vec /= np.linalg.norm(gaze_vec)
    return GazeEstimate(
        gaze_vector_3d=gaze_vec,
        gaze_point_2d=(320.0 + gaze_x * 100, 240.0 + gaze_y * 100),
        confidence=confidence,
        head_pose=_make_head_pose(),
        timestamp=time.time(),
        source_confidences={"2d_landmarks": 0.8, "ai_model": 0.9},
        quality_metrics=_make_quality_metrics(),
        method=method,
    )


def _make_session_summary(n_focus: int = 2, n_wander: int = 1) -> SessionSummary:
    now = time.time()
    focus_events = [
        FocusEvent(
            target_object_id=f"book_{i}",
            focus_type=AttentionType.OBJECT,
            start_time=now + i * 5,
            duration=2.0,
            stability_score=0.9,
            confidence=0.85,
        )
        for i in range(n_focus)
    ]
    wandering_periods = [
        WanderingPeriod(
            start_time=now + 10,
            end_time=now + 12,
            duration=2.0,
            stability_score=0.3,
            average_position=(320.0, 240.0),
            variance=0.05,
        )
        for _ in range(n_wander)
    ]
    return SessionSummary(
        session_id="test-session-001",
        start_time=now,
        end_time=now + 30,
        total_frames=900,
        valid_frames=850,
        focus_events=focus_events,
        wandering_periods=wandering_periods,
        overall_quality=0.82,
        attention_statistics={
            "eye_contact_percentage": 60.0,
            "attention_to_person_percentage": 20.0,
            "attention_to_objects_percentage": 40.0,
            "attention_to_book_percentage": 35.0,
        },
    )


# ---------------------------------------------------------------------------
# Tests: legacy field presence (Requirement 10.1, 10.3)
# ---------------------------------------------------------------------------

class TestLegacyFieldPresence:
    """All legacy GazeAnalysisResponse fields must be present in the output."""

    def test_build_legacy_response_contains_all_legacy_fields(self):
        """build_legacy_response must include every legacy field."""
        estimates = [_make_gaze_estimate() for _ in range(10)]
        response = build_legacy_response(estimates, duration=5.0)

        missing = LEGACY_FIELDS - set(response.keys())
        assert not missing, f"Missing legacy fields: {missing}"

    def test_from_gaze_estimates_contains_all_legacy_fields(self):
        """LegacyResponseAdapter.from_gaze_estimates must include every legacy field."""
        adapter = LegacyResponseAdapter()
        estimates = [_make_gaze_estimate(gaze_x=0.1 * i) for i in range(5)]
        response = adapter.from_gaze_estimates(estimates, duration=3.0)

        missing = LEGACY_FIELDS - set(response.keys())
        assert not missing, f"Missing legacy fields: {missing}"

    def test_from_session_summary_contains_all_legacy_fields(self):
        """LegacyResponseAdapter.from_session_summary must include every legacy field."""
        adapter = LegacyResponseAdapter()
        summary = _make_session_summary()
        response = adapter.from_session_summary(summary)

        missing = LEGACY_FIELDS - set(response.keys())
        assert not missing, f"Missing legacy fields: {missing}"

    def test_empty_estimates_still_has_all_legacy_fields(self):
        """Even with zero estimates, all legacy fields must be present."""
        response = build_legacy_response([], duration=0.0)
        missing = LEGACY_FIELDS - set(response.keys())
        assert not missing, f"Missing legacy fields: {missing}"


# ---------------------------------------------------------------------------
# Tests: enhanced fields are added (Requirement 10.1)
# ---------------------------------------------------------------------------

class TestEnhancedFieldsPresent:
    """Enhanced fields must be present alongside legacy fields."""

    def test_enhanced_fields_present_in_response(self):
        estimates = [_make_gaze_estimate()]
        response = build_legacy_response(estimates, duration=1.0)

        missing = ENHANCED_FIELDS - set(response.keys())
        assert not missing, f"Missing enhanced fields: {missing}"

    def test_enhanced_version_field_is_correct(self):
        estimates = [_make_gaze_estimate()]
        response = build_legacy_response(estimates, duration=1.0)
        assert response["enhanced_version"] == "2.0.0"

    def test_enhanced_flags_are_boolean(self):
        estimates = [_make_gaze_estimate()]
        response = build_legacy_response(estimates, duration=1.0)
        for flag in (
            "head_pose_compensation_active",
            "camera_angle_correction_active",
            "ai_model_active",
            "sensor_fusion_active",
        ):
            assert isinstance(response[flag], bool), f"{flag} should be bool"


# ---------------------------------------------------------------------------
# Tests: legacy field value types (Requirement 10.3)
# ---------------------------------------------------------------------------

class TestLegacyFieldTypes:
    """Legacy fields must have the correct types expected by old consumers."""

    def test_numeric_fields_are_float(self):
        estimates = [_make_gaze_estimate()]
        response = build_legacy_response(estimates, duration=1.0)

        float_fields = [
            "eye_contact_percentage",
            "analyzed_duration",
            "focusing_duration",
            "attention_to_person_percentage",
            "attention_to_objects_percentage",
            "attention_to_book_percentage",
            "book_focusing_score",
            "risk_score",
            "gaze_wandering_score",
            "gaze_wandering_percentage",
            "fatigue_score",
            "focus_level",
        ]
        for field in float_fields:
            assert isinstance(response[field], (int, float)), (
                f"Field '{field}' should be numeric, got {type(response[field])}"
            )

    def test_list_fields_are_lists(self):
        estimates = [_make_gaze_estimate()]
        response = build_legacy_response(estimates, duration=1.0)

        list_fields = [
            "detected_objects",
            "detected_books",
            "object_interaction_events",
            "focus_timeline",
            "wandering_periods",
        ]
        for field in list_fields:
            assert isinstance(response[field], list), (
                f"Field '{field}' should be list, got {type(response[field])}"
            )

    def test_dict_fields_are_dicts(self):
        estimates = [_make_gaze_estimate()]
        response = build_legacy_response(estimates, duration=1.0)

        dict_fields = [
            "gaze_direction_stats",
            "object_focus_stats",
            "pattern_analysis",
            "fatigue_indicators",
            "focus_level_details",
        ]
        for field in dict_fields:
            assert isinstance(response[field], dict), (
                f"Field '{field}' should be dict, got {type(response[field])}"
            )

    def test_total_frames_is_int(self):
        estimates = [_make_gaze_estimate() for _ in range(7)]
        response = build_legacy_response(estimates, duration=2.0)
        assert isinstance(response["total_frames"], int)
        assert response["total_frames"] == 7

    def test_fatigue_level_is_string(self):
        adapter = LegacyResponseAdapter()
        summary = _make_session_summary()
        response = adapter.from_session_summary(summary)
        assert isinstance(response["fatigue_level"], str)

    def test_gaze_direction_stats_has_all_directions(self):
        estimates = [_make_gaze_estimate(gaze_x=0.3), _make_gaze_estimate(gaze_x=-0.3)]
        response = build_legacy_response(estimates, duration=1.0)
        stats = response["gaze_direction_stats"]
        for direction in ("left", "right", "center", "up", "down"):
            assert direction in stats, f"Missing direction '{direction}' in gaze_direction_stats"


# ---------------------------------------------------------------------------
# Tests: object detection metadata (Requirement 10.3)
# ---------------------------------------------------------------------------

class TestObjectDetectionMetadata:
    """object_detection_model and object_detection_available must be forwarded."""

    def test_object_detection_model_is_forwarded(self):
        response = build_legacy_response(
            [],
            duration=0.0,
            object_detection_model="yolov8l-oiv7",
            object_detection_available=True,
        )
        assert response["object_detection_model"] == "yolov8l-oiv7"
        assert response["object_detection_available"] is True

    def test_object_detection_defaults_to_none_and_false(self):
        response = build_legacy_response([], duration=0.0)
        assert response["object_detection_model"] is None
        assert response["object_detection_available"] is False


# ---------------------------------------------------------------------------
# Tests: session summary → legacy response (Requirement 10.1, 10.3)
# ---------------------------------------------------------------------------

class TestSessionSummaryConversion:
    """SessionSummary must map correctly to legacy fields."""

    def test_focus_timeline_populated_from_focus_events(self):
        adapter = LegacyResponseAdapter()
        summary = _make_session_summary(n_focus=3)
        response = adapter.from_session_summary(summary)

        assert len(response["focus_timeline"]) == 3
        for entry in response["focus_timeline"]:
            assert "start_time" in entry
            assert "duration" in entry
            assert "focus_type" in entry

    def test_wandering_periods_populated(self):
        adapter = LegacyResponseAdapter()
        summary = _make_session_summary(n_wander=2)
        response = adapter.from_session_summary(summary)

        assert len(response["wandering_periods"]) == 2
        for period in response["wandering_periods"]:
            assert "start_time" in period
            assert "end_time" in period
            assert "duration" in period

    def test_total_frames_matches_summary(self):
        adapter = LegacyResponseAdapter()
        summary = _make_session_summary()
        response = adapter.from_session_summary(summary)
        assert response["total_frames"] == summary.total_frames

    def test_analyzed_duration_matches_summary(self):
        adapter = LegacyResponseAdapter()
        summary = _make_session_summary()
        response = adapter.from_session_summary(summary)
        expected_duration = summary.end_time - summary.start_time
        assert abs(response["analyzed_duration"] - expected_duration) < 0.01


# ---------------------------------------------------------------------------
# Tests: ConfigBridge (Requirement 10.2)
# ---------------------------------------------------------------------------

class TestConfigBridge:
    """Legacy config parameters must be accepted and translated correctly."""

    def test_legacy_keys_are_translated(self):
        legacy = {
            "FACE_DETECTION_CONFIDENCE": 0.7,
            "HEAD_POSE_ENABLED": True,
            "MIN_FOCUS_DURATION": 1.5,
            "TARGET_FPS": 25,
        }
        config = ConfigBridge.from_legacy_dict(legacy)
        assert config.face_detection_confidence == 0.7
        assert config.head_pose_compensation is True
        assert config.min_focus_duration == 1.5
        assert config.target_fps == 25

    def test_enhanced_keys_pass_through(self):
        enhanced = {
            "face_detection_confidence": 0.6,
            "gaze_estimation_method": "multi_modal",
        }
        config = ConfigBridge.from_legacy_dict(enhanced)
        assert config.face_detection_confidence == 0.6
        assert config.gaze_estimation_method == "multi_modal"

    def test_unknown_keys_are_ignored(self):
        """Unknown legacy keys must not raise an error."""
        legacy = {
            "SOME_UNKNOWN_LEGACY_KEY": "value",
            "face_detection_confidence": 0.5,
        }
        config = ConfigBridge.from_legacy_dict(legacy)
        assert config.face_detection_confidence == 0.5

    def test_string_bool_coercion(self):
        legacy = {"HEAD_POSE_ENABLED": "true", "USE_GPU": "false"}
        config = ConfigBridge.from_legacy_dict(legacy)
        assert config.head_pose_compensation is True
        assert config.gpu_acceleration is False

    def test_merge_configs_enhanced_takes_precedence(self):
        legacy = {"FACE_DETECTION_CONFIDENCE": 0.5}
        enhanced = {"face_detection_confidence": 0.9}
        config = ConfigBridge.merge_configs(legacy, enhanced)
        assert config.face_detection_confidence == 0.9

    def test_to_legacy_dict_contains_legacy_keys(self):
        config = EnhancedGazeConfig(face_detection_confidence=0.75)
        legacy_dict = ConfigBridge.to_legacy_dict(config)
        # At least one legacy key should appear
        assert any(k in LEGACY_TO_ENHANCED for k in legacy_dict), (
            "to_legacy_dict should produce at least some legacy-named keys"
        )

    def test_roundtrip_legacy_to_enhanced_to_legacy(self):
        """Translating legacy → enhanced → legacy should preserve values."""
        original_legacy = {"FACE_DETECTION_CONFIDENCE": 0.65, "TARGET_FPS": 25}
        config = ConfigBridge.from_legacy_dict(original_legacy)
        roundtripped = ConfigBridge.to_legacy_dict(config)

        # The values should survive the round-trip
        assert roundtripped.get("FACE_DETECTION_CONFIDENCE") == 0.65
        assert roundtripped.get("TARGET_FPS") == 25
