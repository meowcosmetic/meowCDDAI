"""
Property-based tests for wandering behavior classification.

**Feature: ai-enhanced-gaze-tracking, Property 16: Wandering Behavior Classification**
**Validates: Requirements 6.4**

For any stable gaze pattern with no identifiable target, the system should
classify this as wandering behavior.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch
from typing import List, Dict, Any, Optional

from ai_enhanced_gaze_tracking.components.focus_detection.focus_detector import ImprovedFocusDetector
from ai_enhanced_gaze_tracking.core.data_models import AttentionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(
    min_focus_duration: float = 0.5,
    stability_threshold: float = 20.0,
    wandering_stability_threshold: float = 15.0,
) -> ImprovedFocusDetector:
    return ImprovedFocusDetector(
        min_focus_duration=min_focus_duration,
        stability_threshold=stability_threshold,
        wandering_stability_threshold=wandering_stability_threshold,
        ray_intersection_threshold=50.0,
        history_size=30,
    )


def run_detector_frames(
    detector: ImprovedFocusDetector,
    gaze_vectors: List[np.ndarray],
    tracked_objects: List[Dict[str, Any]],
    fps: int = 30,
    start_time: float = 1000.0,
) -> List[Optional[object]]:
    """
    Feed gaze vectors through the detector using a simulated monotonic clock.
    Returns all FocusEvent results (including None entries).
    """
    frame_dt = 1.0 / fps
    results = []
    current_time = start_time

    for vec in gaze_vectors:
        with patch(
            "ai_enhanced_gaze_tracking.components.focus_detection.focus_detector.time.time",
            return_value=current_time,
        ):
            event = detector.detect_focus(
                gaze_vector=vec,
                tracked_objects=tracked_objects,
                stability_metrics={"variance": 5.0},
            )
        results.append(event)
        current_time += frame_dt

    return results


def make_stable_gaze_vectors(
    num_frames: int,
    center: np.ndarray = None,
    noise_scale: float = 0.003,
    seed: int = 42,
) -> List[np.ndarray]:
    """Generate a sequence of nearly-identical (stable) gaze vectors."""
    if center is None:
        center = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    center = center / np.linalg.norm(center)
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(num_frames):
        noise = rng.normal(0, noise_scale, 3)
        v = center + noise
        v[2] = abs(v[2]) + 0.1
        v = v / np.linalg.norm(v)
        vectors.append(v)
    return vectors


def make_unstable_gaze_vectors(num_frames: int, seed: int = 99) -> List[np.ndarray]:
    """Generate highly variable (unstable) gaze vectors."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(num_frames):
        v = rng.normal(0, 1, 3)
        v[2] = abs(v[2]) + 0.1
        v = v / np.linalg.norm(v)
        vectors.append(v)
    return vectors


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def stable_gaze_direction_strategy(draw) -> np.ndarray:
    """Generate a random forward-facing unit vector as gaze center."""
    x = draw(st.floats(min_value=-0.5, max_value=0.5))
    y = draw(st.floats(min_value=-0.5, max_value=0.5))
    z = draw(st.floats(min_value=0.3, max_value=1.0))
    vec = np.array([x, y, z], dtype=np.float64)
    assume(np.linalg.norm(vec) > 1e-6)
    return vec / np.linalg.norm(vec)


@st.composite
def long_stable_sequence_strategy(draw):
    """
    Generate a stable gaze sequence long enough to exceed min_focus_duration=0.5s
    at 30 fps (needs > 15 frames; use 25-50 to be safe).
    """
    num_frames = draw(st.integers(min_value=25, max_value=50))
    center = draw(stable_gaze_direction_strategy())
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return make_stable_gaze_vectors(num_frames, center=center, noise_scale=0.002, seed=seed)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestWanderingBehaviorClassification:
    """
    Property 16: Wandering Behavior Classification

    For any stable gaze pattern with no identifiable target, the system
    should classify this as wandering behavior.
    """

    # ------------------------------------------------------------------
    # Core property: stable gaze + no objects → WANDERING
    # ------------------------------------------------------------------

    @given(gaze_vectors=long_stable_sequence_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_stable_gaze_no_objects_classified_as_wandering(self, gaze_vectors):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 16: Wandering Behavior Classification**
        **Validates: Requirements 6.4**

        Property: For any stable gaze sequence with no tracked objects present,
        once the minimum duration is exceeded the system should produce a
        WANDERING focus event.
        """
        detector = make_detector()
        results = run_detector_frames(detector, gaze_vectors, tracked_objects=[])

        non_none = [e for e in results if e is not None]

        # After enough stable frames there must be at least one event
        assert len(non_none) > 0, (
            "Expected at least one FocusEvent for a long stable gaze with no objects"
        )

        # Every non-None event must be WANDERING
        for event in non_none:
            assert event.focus_type == AttentionType.WANDERING, (
                f"Expected WANDERING, got {event.focus_type} "
                f"(target={event.target_object_id})"
            )

    @given(
        center=stable_gaze_direction_strategy(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=5000)
    def test_wandering_event_has_no_target_object(self, center, seed):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 16: Wandering Behavior Classification**
        **Validates: Requirements 6.4**

        Property: For any wandering event, target_object_id must be None because
        there is no specific object being focused on.
        """
        detector = make_detector()
        # 40 frames @ 30 fps ≈ 1.33 s — well above min_focus_duration
        gaze_vectors = make_stable_gaze_vectors(40, center=center, seed=seed)
        results = run_detector_frames(detector, gaze_vectors, tracked_objects=[])

        for event in results:
            if event is not None and event.focus_type == AttentionType.WANDERING:
                assert event.target_object_id is None, (
                    f"Wandering event should have no target object, "
                    f"got target_object_id={event.target_object_id}"
                )

    # ------------------------------------------------------------------
    # Property: unstable gaze → NOT classified as wandering
    # ------------------------------------------------------------------

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=100, deadline=5000)
    def test_unstable_gaze_not_classified_as_wandering(self, seed):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 16: Wandering Behavior Classification**
        **Validates: Requirements 6.4**

        Property: Highly unstable gaze (large variance) should NOT be classified
        as wandering, because wandering requires stable gaze with no target.
        """
        detector = make_detector()
        gaze_vectors = make_unstable_gaze_vectors(50, seed=seed)
        results = run_detector_frames(detector, gaze_vectors, tracked_objects=[])

        wandering_events = [
            e for e in results
            if e is not None and e.focus_type == AttentionType.WANDERING
        ]

        assert len(wandering_events) == 0, (
            f"Unstable gaze should not be classified as wandering, "
            f"but got {len(wandering_events)} wandering event(s)"
        )

    # ------------------------------------------------------------------
    # Property: stable gaze + objects present → NOT wandering
    # ------------------------------------------------------------------

    @given(
        center=stable_gaze_direction_strategy(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=5000)
    def test_stable_gaze_with_objects_not_wandering(self, center, seed):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 16: Wandering Behavior Classification**
        **Validates: Requirements 6.4**

        Property: When tracked objects are present in the scene, stable gaze
        should NOT be classified as wandering — it should either be object focus
        or no event (if gaze misses all objects).
        """
        detector = make_detector()
        gaze_vectors = make_stable_gaze_vectors(40, center=center, seed=seed)

        tracked_objects = [{
            "id": "toy_1",
            "class_name": "toy",
            "bbox": (270.0, 190.0, 100.0, 100.0),
            "confidence": 0.9,
            "depth_estimate": 500.0,
        }]

        results = run_detector_frames(detector, gaze_vectors, tracked_objects=tracked_objects)

        wandering_events = [
            e for e in results
            if e is not None and e.focus_type == AttentionType.WANDERING
        ]

        assert len(wandering_events) == 0, (
            f"Stable gaze with objects present should not be classified as wandering, "
            f"but got {len(wandering_events)} wandering event(s)"
        )

    # ------------------------------------------------------------------
    # Property: wandering event has positive confidence and stability
    # ------------------------------------------------------------------

    @given(
        center=stable_gaze_direction_strategy(),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=100, deadline=5000)
    def test_wandering_event_has_valid_metadata(self, center, seed):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 16: Wandering Behavior Classification**
        **Validates: Requirements 6.4**

        Property: For any wandering event, confidence and stability_score must
        be in [0, 1] and duration must be non-negative.
        """
        detector = make_detector()
        gaze_vectors = make_stable_gaze_vectors(40, center=center, seed=seed)
        results = run_detector_frames(detector, gaze_vectors, tracked_objects=[])

        for event in results:
            if event is not None and event.focus_type == AttentionType.WANDERING:
                assert 0.0 <= event.confidence <= 1.0, (
                    f"Wandering event confidence {event.confidence} out of [0, 1]"
                )
                assert 0.0 <= event.stability_score <= 1.0, (
                    f"Wandering event stability_score {event.stability_score} out of [0, 1]"
                )
                assert event.duration >= 0.0, (
                    f"Wandering event duration {event.duration} must be non-negative"
                )
