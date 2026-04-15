"""
Property-based tests for 3D ray casting object selection.

**Feature: ai-enhanced-gaze-tracking, Property 15: 3D Ray Casting Object Selection**
**Validates: Requirements 6.3**

For any scene with multiple objects, the focus detection should use 3D ray casting
to determine which specific object intersects with the gaze ray.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import patch
from typing import List, Dict, Any, Optional, Tuple

from ai_enhanced_gaze_tracking.components.focus_detection.focus_detector import ImprovedFocusDetector
from ai_enhanced_gaze_tracking.core.data_models import AttentionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(ray_threshold: float = 50.0) -> ImprovedFocusDetector:
    return ImprovedFocusDetector(
        min_focus_duration=0.5,
        stability_threshold=20.0,
        wandering_stability_threshold=15.0,
        ray_intersection_threshold=ray_threshold,
        history_size=30,
    )


def make_object(
    obj_id: str,
    center_x: float,
    center_y: float,
    width: float = 80.0,
    height: float = 80.0,
    depth: float = 500.0,
    class_name: str = "toy",
) -> Dict[str, Any]:
    """Build a tracked-object dict compatible with ImprovedFocusDetector."""
    return {
        "id": obj_id,
        "class_name": class_name,
        "bbox": (center_x - width / 2, center_y - height / 2, width, height),
        "confidence": 0.9,
        "depth_estimate": depth,
    }


def gaze_toward_screen_point(screen_x: float, screen_y: float, depth: float = 500.0) -> np.ndarray:
    """
    Build a unit gaze vector that points toward a given 2D screen coordinate.

    The detector's _ray_cast_to_objects projects objects to 3D as:
        obj_3d = [(cx - 320) / 320 * depth,
                  -(cy - 240) / 240 * depth,
                  depth]
    So we construct a ray that passes through that same 3D point.
    """
    x3d = (screen_x - 320.0) / 320.0 * depth
    y3d = -(screen_y - 240.0) / 240.0 * depth
    z3d = depth
    vec = np.array([x3d, y3d, z3d], dtype=np.float64)
    return vec / np.linalg.norm(vec)


def run_detector_frames(
    detector: ImprovedFocusDetector,
    gaze_vector: np.ndarray,
    tracked_objects: List[Dict[str, Any]],
    num_frames: int = 40,
    fps: int = 30,
    start_time: float = 1000.0,
) -> Optional[object]:
    """
    Feed the same gaze vector for `num_frames` frames using a simulated clock.
    Returns the last non-None FocusEvent, or None.
    """
    frame_dt = 1.0 / fps
    last_event = None
    current_time = start_time

    for _ in range(num_frames):
        with patch(
            "ai_enhanced_gaze_tracking.components.focus_detection.focus_detector.time.time",
            return_value=current_time,
        ):
            event = detector.detect_focus(
                gaze_vector=gaze_vector,
                tracked_objects=tracked_objects,
                stability_metrics={"variance": 3.0},
            )
        if event is not None:
            last_event = event
        current_time += frame_dt

    return last_event


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def two_separated_objects_strategy(draw):
    """
    Generate two objects whose screen-space centers are well separated
    so that a ray aimed at one clearly misses the other.

    Returns (obj_left, obj_right, gaze_at_left, gaze_at_right).
    """
    # Left object: x in [80, 200], right object: x in [440, 560]
    left_x = draw(st.floats(min_value=80.0, max_value=200.0))
    right_x = draw(st.floats(min_value=440.0, max_value=560.0))
    y = draw(st.floats(min_value=100.0, max_value=380.0))
    depth = draw(st.floats(min_value=300.0, max_value=700.0))

    obj_left = make_object("left", left_x, y, depth=depth)
    obj_right = make_object("right", right_x, y, depth=depth)

    gaze_left = gaze_toward_screen_point(left_x, y, depth)
    gaze_right = gaze_toward_screen_point(right_x, y, depth)

    return obj_left, obj_right, gaze_left, gaze_right


@st.composite
def near_far_objects_strategy(draw):
    """
    Generate two objects at the same screen position but different depths.
    The nearer one should be selected.
    """
    cx = draw(st.floats(min_value=200.0, max_value=440.0))
    cy = draw(st.floats(min_value=100.0, max_value=380.0))
    near_depth = draw(st.floats(min_value=200.0, max_value=400.0))
    far_depth = draw(st.floats(min_value=600.0, max_value=900.0))

    obj_near = make_object("near", cx, cy, depth=near_depth)
    obj_far = make_object("far", cx, cy, depth=far_depth)

    # Gaze aimed at the shared screen position (use near depth for direction)
    gaze = gaze_toward_screen_point(cx, cy, near_depth)

    return obj_near, obj_far, gaze


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRayCastingObjectSelection:
    """
    Property 15: 3D Ray Casting Object Selection

    For any scene with multiple objects, the focus detection should use 3D ray
    casting to determine which specific object intersects with the gaze ray.
    """

    # ------------------------------------------------------------------
    # Property: gaze aimed at object A selects A, not B
    # ------------------------------------------------------------------

    @given(scene=two_separated_objects_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_gaze_at_left_object_selects_left(self, scene):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 15: 3D Ray Casting Object Selection**
        **Validates: Requirements 6.3**

        For any two well-separated objects, when gaze is directed at the left
        object the detected focus target should be the left object, not the right.
        """
        obj_left, obj_right, gaze_left, _ = scene
        detector = make_detector()
        tracked_objects = [obj_left, obj_right]

        event = run_detector_frames(detector, gaze_left, tracked_objects)

        assert event is not None, (
            "Expected a focus event when gaze is stably directed at an object"
        )
        assert event.target_object_id == "left", (
            f"Expected focus on 'left' object, got '{event.target_object_id}'"
        )

    @given(scene=two_separated_objects_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_gaze_at_right_object_selects_right(self, scene):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 15: 3D Ray Casting Object Selection**
        **Validates: Requirements 6.3**

        For any two well-separated objects, when gaze is directed at the right
        object the detected focus target should be the right object, not the left.
        """
        obj_left, obj_right, _, gaze_right = scene
        detector = make_detector()
        tracked_objects = [obj_left, obj_right]

        event = run_detector_frames(detector, gaze_right, tracked_objects)

        assert event is not None, (
            "Expected a focus event when gaze is stably directed at an object"
        )
        assert event.target_object_id == "right", (
            f"Expected focus on 'right' object, got '{event.target_object_id}'"
        )

    # ------------------------------------------------------------------
    # Property: gaze direction change switches selected object
    # ------------------------------------------------------------------

    @given(scene=two_separated_objects_strategy())
    @settings(max_examples=100, deadline=10000)
    def test_gaze_switch_changes_selected_object(self, scene):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 15: 3D Ray Casting Object Selection**
        **Validates: Requirements 6.3**

        For any two well-separated objects, switching gaze from one to the other
        should change the detected focus target accordingly.
        """
        obj_left, obj_right, gaze_left, gaze_right = scene
        tracked_objects = [obj_left, obj_right]

        # Phase 1: look at left
        detector = make_detector()
        event_left = run_detector_frames(
            detector, gaze_left, tracked_objects, num_frames=40, start_time=1000.0
        )

        # Phase 2: look at right (fresh detector to avoid state carry-over)
        detector2 = make_detector()
        event_right = run_detector_frames(
            detector2, gaze_right, tracked_objects, num_frames=40, start_time=1000.0
        )

        # Both should produce focus events
        assert event_left is not None, "Expected focus event when looking at left object"
        assert event_right is not None, "Expected focus event when looking at right object"

        # They should select different objects
        assert event_left.target_object_id != event_right.target_object_id, (
            f"Expected different targets for different gaze directions, "
            f"but both selected '{event_left.target_object_id}'"
        )

    # ------------------------------------------------------------------
    # Property: gaze aimed away from all objects yields no object focus
    # ------------------------------------------------------------------

    @given(scene=two_separated_objects_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_gaze_away_from_all_objects_yields_no_object_focus(self, scene):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 15: 3D Ray Casting Object Selection**
        **Validates: Requirements 6.3**

        When gaze is directed away from all tracked objects (e.g. far corner),
        no object focus should be detected even though objects are present.
        """
        obj_left, obj_right, _, _ = scene
        tracked_objects = [obj_left, obj_right]

        # Gaze aimed at top-left corner — far from both objects
        gaze_away = gaze_toward_screen_point(10.0, 10.0, 500.0)

        detector = make_detector()
        event = run_detector_frames(detector, gaze_away, tracked_objects)

        if event is not None:
            assert event.focus_type not in (AttentionType.OBJECT, AttentionType.PERSON), (
                f"Gaze directed away from all objects should not yield object focus, "
                f"got focus_type={event.focus_type}, target={event.target_object_id}"
            )

    # ------------------------------------------------------------------
    # Property: nearest object is selected when two objects share screen pos
    # ------------------------------------------------------------------

    @given(scene=near_far_objects_strategy())
    @settings(max_examples=100, deadline=5000)
    def test_nearest_object_selected_when_overlapping(self, scene):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 15: 3D Ray Casting Object Selection**
        **Validates: Requirements 6.3**

        When two objects occupy the same screen position but at different depths,
        the nearer object should be selected by the ray casting algorithm.
        """
        obj_near, obj_far, gaze = scene
        tracked_objects = [obj_near, obj_far]

        detector = make_detector()
        event = run_detector_frames(detector, gaze, tracked_objects)

        assert event is not None, (
            "Expected a focus event when gaze is directed at overlapping objects"
        )
        assert event.target_object_id == "near", (
            f"Expected nearest object 'near' to be selected, "
            f"got '{event.target_object_id}'"
        )

    # ------------------------------------------------------------------
    # Sanity: single object in scene is selected when gaze points at it
    # ------------------------------------------------------------------

    def test_single_object_selected_when_gaze_points_at_it(self):
        """
        Sanity check: with a single object and gaze aimed directly at it,
        that object should be the focus target.
        """
        obj = make_object("solo", center_x=320.0, center_y=240.0, depth=500.0)
        gaze = gaze_toward_screen_point(320.0, 240.0, 500.0)

        detector = make_detector()
        event = run_detector_frames(detector, gaze, [obj])

        assert event is not None, "Expected focus event for single object"
        assert event.target_object_id == "solo", (
            f"Expected 'solo', got '{event.target_object_id}'"
        )
        assert event.focus_type in (AttentionType.OBJECT, AttentionType.PERSON)
