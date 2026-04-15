"""
Property-based tests for non-object focus rejection.

**Feature: ai-enhanced-gaze-tracking, Property 14: Non-Object Focus Rejection**
**Validates: Requirements 6.2**

This module tests that when no tracked objects are present in the scene,
the focus detection system never classifies gaze as object focus, regardless
of gaze direction or stability.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch
from typing import List, Dict, Any, Optional

from ai_enhanced_gaze_tracking.components.focus_detection.focus_detector import ImprovedFocusDetector
from ai_enhanced_gaze_tracking.core.data_models import AttentionType


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def gaze_vector_strategy(draw) -> np.ndarray:
    """Generate a random forward-facing 3D gaze vector."""
    x = draw(st.floats(min_value=-0.8, max_value=0.8))
    y = draw(st.floats(min_value=-0.8, max_value=0.8))
    z = draw(st.floats(min_value=0.2, max_value=1.0))
    vec = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    assume(norm > 1e-6)
    return vec / norm


@st.composite
def stable_gaze_sequence_strategy(draw, min_frames: int = 20, max_frames: int = 50):
    """Generate a sequence of similar (stable) gaze vectors."""
    num_frames = draw(st.integers(min_value=min_frames, max_value=max_frames))
    x = draw(st.floats(min_value=-0.4, max_value=0.4))
    y = draw(st.floats(min_value=-0.4, max_value=0.4))
    base = np.array([x, y, 1.0], dtype=np.float64)
    base = base / np.linalg.norm(base)

    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**31)))
    vectors = []
    for _ in range(num_frames):
        noise = rng.normal(0, 0.003, 3)
        v = base + noise
        v[2] = abs(v[2]) + 0.1
        v = v / np.linalg.norm(v)
        vectors.append(v)
    return vectors


# ---------------------------------------------------------------------------
# Helper: run detector with simulated time (no real sleeping)
# ---------------------------------------------------------------------------

def run_detector_with_simulated_time(
    focus_detector: ImprovedFocusDetector,
    gaze_vectors: List[np.ndarray],
    tracked_objects: List[Dict[str, Any]],
    fps: int = 30,
    start_time: float = 1000.0,
) -> List[Optional[object]]:
    """
    Feed gaze vectors through the detector using a monotonically advancing
    simulated clock instead of real wall-clock time.  This avoids any
    time.sleep() calls and keeps tests fast.
    """
    frame_time = 1.0 / fps
    results = []
    current_time = start_time

    for vec in gaze_vectors:
        with patch(
            "ai_enhanced_gaze_tracking.components.focus_detection.focus_detector.time.time",
            return_value=current_time,
        ):
            event = focus_detector.detect_focus(
                gaze_vector=vec,
                tracked_objects=tracked_objects,
                stability_metrics={"variance": 5.0},
            )
        results.append(event)
        current_time += frame_time

    return results


def make_detector() -> ImprovedFocusDetector:
    return ImprovedFocusDetector(
        min_focus_duration=0.5,
        stability_threshold=20.0,
        wandering_stability_threshold=15.0,
        ray_intersection_threshold=50.0,
        history_size=30,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestNonObjectFocusRejection:
    """
    Property 14: Non-Object Focus Rejection

    For any gaze pattern toward the camera with no tracked objects present,
    the system should NOT classify this as object focus.
    """

    # ------------------------------------------------------------------
    # Core property: no objects → never OBJECT or PERSON focus
    # ------------------------------------------------------------------

    @given(gaze_vectors=stable_gaze_sequence_strategy(min_frames=30, max_frames=50))
    @settings(max_examples=100, deadline=5000)
    def test_no_objects_never_yields_object_focus(self, gaze_vectors):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 14: Non-Object Focus Rejection**
        **Validates: Requirements 6.2**

        Property: For any stable gaze sequence with an empty object list,
        no returned FocusEvent should have focus_type OBJECT or PERSON.
        """
        detector = make_detector()
        results = run_detector_with_simulated_time(detector, gaze_vectors, tracked_objects=[])

        for event in results:
            if event is not None:
                assert event.focus_type not in (AttentionType.OBJECT, AttentionType.PERSON), (
                    f"Expected no object/person focus when no objects are present, "
                    f"but got focus_type={event.focus_type}"
                )

    @given(gaze_vector=gaze_vector_strategy())
    @settings(max_examples=100, deadline=2000)
    def test_single_frame_no_objects_not_object_focus(self, gaze_vector):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 14: Non-Object Focus Rejection**
        **Validates: Requirements 6.2**

        Property: For any single gaze vector with no tracked objects,
        the result is never classified as object focus.
        """
        detector = make_detector()
        with patch(
            "ai_enhanced_gaze_tracking.components.focus_detection.focus_detector.time.time",
            return_value=1000.0,
        ):
            event = detector.detect_focus(
                gaze_vector=gaze_vector,
                tracked_objects=[],
                stability_metrics={"variance": 5.0},
            )

        if event is not None:
            assert event.focus_type not in (AttentionType.OBJECT, AttentionType.PERSON), (
                f"Single-frame result with no objects must not be object/person focus, "
                f"got {event.focus_type}"
            )

    @given(gaze_vectors=stable_gaze_sequence_strategy(min_frames=40, max_frames=50))
    @settings(max_examples=100, deadline=5000)
    def test_stable_gaze_no_objects_not_object_focus(self, gaze_vectors):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 14: Non-Object Focus Rejection**
        **Validates: Requirements 6.2**

        Property: Even when gaze is highly stable (which would normally trigger
        focus detection), the absence of tracked objects must prevent any
        OBJECT or PERSON classification.
        """
        detector = make_detector()
        results = run_detector_with_simulated_time(detector, gaze_vectors, tracked_objects=[])

        object_focus_events = [
            e for e in results
            if e is not None and e.focus_type in (AttentionType.OBJECT, AttentionType.PERSON)
        ]

        assert len(object_focus_events) == 0, (
            f"Stable gaze with no objects produced {len(object_focus_events)} "
            f"object/person focus event(s). First: {object_focus_events[0]}"
        )

    # ------------------------------------------------------------------
    # Sanity check: objects present → object focus IS possible
    # ------------------------------------------------------------------

    def test_with_objects_present_object_focus_is_possible(self):
        """
        Sanity check: when a tracked object IS present and gaze is stable
        and long enough, object focus CAN be detected.

        This validates that the rejection property is specific to the
        no-objects case and not a blanket suppression.
        """
        detector = make_detector()

        tracked_objects = [{
            "id": "toy_1",
            "class_name": "toy",
            "bbox": (270, 190, 100, 100),
            "confidence": 0.9,
            "depth_estimate": 500.0,
        }]

        base_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        fps = 30
        duration = 0.7  # above min_focus_duration=0.5
        num_frames = int(duration * fps)
        frame_time = 1.0 / fps
        rng = np.random.default_rng(42)

        focus_event = None
        current_time = 1000.0
        for _ in range(num_frames):
            noise = rng.normal(0, 0.003, 3)
            v = base_vec + noise
            v[2] = abs(v[2]) + 0.1
            v = v / np.linalg.norm(v)
            with patch(
                "ai_enhanced_gaze_tracking.components.focus_detection.focus_detector.time.time",
                return_value=current_time,
            ):
                focus_event = detector.detect_focus(
                    gaze_vector=v,
                    tracked_objects=tracked_objects,
                    stability_metrics={"variance": 3.0},
                )
            current_time += frame_time

        assert focus_event is not None, (
            "Expected a focus event when an object is present and gaze is stable"
        )
        assert focus_event.focus_type in (AttentionType.OBJECT, AttentionType.PERSON), (
            f"Expected OBJECT or PERSON focus type, got {focus_event.focus_type}"
        )
