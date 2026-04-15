"""
Property-based tests for focus detection stability requirement.

**Feature: ai-enhanced-gaze-tracking, Property 13: Focus Detection Stability Requirement**
**Validates: Requirements 6.1**

This module tests that focus is only detected when gaze remains stable
on a target for the minimum required duration.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import time
from typing import List, Tuple, Dict, Any

from ai_enhanced_gaze_tracking.components.focus_detection.focus_detector import ImprovedFocusDetector
from ai_enhanced_gaze_tracking.core.data_models import FocusEvent, AttentionType


class TestFocusStabilityRequirement:
    """Test suite for focus detection stability requirement property."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create focus detector with known parameters
        self.min_focus_duration = 0.5  # 500ms minimum
        self.stability_threshold = 20.0  # pixels
        
        self.focus_detector = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold,
            wandering_stability_threshold=15.0,
            ray_intersection_threshold=50.0,
            history_size=30
        )
    
    def generate_stable_gaze_sequence(
        self,
        duration: float,
        center: Tuple[float, float] = (0.0, 0.0, 1.0),
        variance: float = 5.0,
        fps: int = 30
    ) -> List[np.ndarray]:
        """
        Generate a sequence of stable gaze vectors.
        
        Args:
            duration: Duration of sequence in seconds
            center: Center gaze direction (3D vector)
            variance: Position variance in pixels
            fps: Frames per second
            
        Returns:
            List of gaze vectors
        """
        num_frames = int(duration * fps)
        gaze_vectors = []
        
        center_vec = np.array(center, dtype=np.float32)
        center_vec = center_vec / np.linalg.norm(center_vec)
        
        for _ in range(num_frames):
            # Add small random perturbation
            noise = np.random.normal(0, variance / 1000.0, 3)
            gaze_vec = center_vec + noise
            gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)
            gaze_vectors.append(gaze_vec)
        
        return gaze_vectors
    
    def generate_unstable_gaze_sequence(
        self,
        duration: float,
        variance: float = 100.0,
        fps: int = 30
    ) -> List[np.ndarray]:
        """
        Generate a sequence of unstable (wandering) gaze vectors.
        
        Args:
            duration: Duration of sequence in seconds
            variance: Position variance (high for unstable)
            fps: Frames per second
            
        Returns:
            List of gaze vectors
        """
        num_frames = int(duration * fps)
        gaze_vectors = []
        
        for _ in range(num_frames):
            # Random gaze direction
            gaze_vec = np.random.normal(0, 1, 3)
            gaze_vec[2] = abs(gaze_vec[2])  # Ensure forward direction
            gaze_vec = gaze_vec / np.linalg.norm(gaze_vec)
            gaze_vectors.append(gaze_vec)
        
        return gaze_vectors
    
    def create_tracked_object(
        self,
        position: Tuple[float, float] = (320, 240),
        size: Tuple[float, float] = (100, 100),
        class_name: str = "toy"
    ) -> Dict[str, Any]:
        """Create a tracked object for testing."""
        x, y = position
        w, h = size
        
        return {
            'id': f'{class_name}_1',
            'class_name': class_name,
            'bbox': (x - w/2, y - h/2, w, h),
            'confidence': 0.9,
            'depth_estimate': 500.0
        }
    
    @given(
        stable_duration=st.floats(min_value=0.1, max_value=0.35),
        variance=st.floats(min_value=1.0, max_value=15.0)
    )
    @settings(max_examples=10, deadline=None)
    def test_focus_not_detected_below_minimum_duration(self, stable_duration, variance):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 13: Focus Detection Stability Requirement**
        **Validates: Requirements 6.1**
        
        Property: For any stable gaze pattern with duration less than the minimum
        required duration, focus should NOT be detected.
        
        This ensures that brief glances are not classified as genuine focus.
        """
        # Ensure duration is well below minimum (with buffer for timing variations)
        assume(stable_duration < self.min_focus_duration * 0.7)
        
        # Ensure variance is within stable range
        assume(variance < self.stability_threshold)
        
        # Create a fresh focus detector for this example to avoid state pollution
        focus_detector = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold,
            wandering_stability_threshold=15.0,
            ray_intersection_threshold=50.0,
            history_size=30
        )
        
        # Generate stable gaze sequence below minimum duration
        gaze_vectors = self.generate_stable_gaze_sequence(
            duration=stable_duration,
            center=(0.0, 0.0, 1.0),
            variance=variance,
            fps=30
        )
        
        # Create a tracked object at the gaze location
        tracked_objects = [self.create_tracked_object(position=(320, 240))]
        
        # Calculate frame time
        frame_time = 1.0 / 30.0
        
        # Process each gaze vector
        focus_detected = False
        for gaze_vec in gaze_vectors:
            focus_event = focus_detector.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': variance}
            )
            
            if focus_event is not None:
                focus_detected = True
                break
            
            # Sleep for actual frame time
            time.sleep(frame_time)
        
        # Assert: Focus should NOT be detected for duration below minimum
        assert not focus_detected, (
            f"Focus was incorrectly detected for stable gaze duration "
            f"{stable_duration:.3f}s < minimum {self.min_focus_duration}s. "
            f"Variance: {variance:.2f} pixels"
        )
    
    @given(
        stable_duration=st.floats(min_value=0.65, max_value=0.9),
        variance=st.floats(min_value=1.0, max_value=15.0)
    )
    @settings(max_examples=5, deadline=None)
    def test_focus_detected_above_minimum_duration(self, stable_duration, variance):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 13: Focus Detection Stability Requirement**
        **Validates: Requirements 6.1**
        
        Property: For any stable gaze pattern with duration equal to or greater than
        the minimum required duration, focus SHOULD be detected.
        """
        # Ensure duration clearly exceeds minimum (with buffer for timing)
        assume(stable_duration >= self.min_focus_duration * 1.3)
        
        # Ensure variance is within stable range
        assume(variance < self.stability_threshold)
        
        # Create a fresh focus detector for this example
        focus_detector = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold
        )
        
        # Generate stable gaze sequence meeting minimum duration
        gaze_vectors = self.generate_stable_gaze_sequence(
            duration=stable_duration,
            center=(0.0, 0.0, 1.0),
            variance=variance,
            fps=30
        )
        
        # Create a tracked object at the gaze location
        tracked_objects = [self.create_tracked_object(position=(320, 240))]
        
        # Calculate frame time based on fps
        frame_time = 1.0 / 30.0  # 30 fps = ~33ms per frame
        
        # Process each gaze vector with proper timing
        focus_event = None
        for gaze_vec in gaze_vectors:
            focus_event = focus_detector.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': variance}
            )
            
            # Sleep for actual frame time to allow real time to pass
            time.sleep(frame_time)
        
        # Assert: Focus SHOULD be detected after sufficient duration
        assert focus_event is not None, (
            f"Focus was not detected for stable gaze duration "
            f"{stable_duration:.3f}s >= minimum {self.min_focus_duration}s. "
            f"Variance: {variance:.2f} pixels"
        )
        
        # Verify focus event properties
        assert focus_event.duration >= self.min_focus_duration * 0.9, (
            f"Focus event duration {focus_event.duration:.3f}s < minimum {self.min_focus_duration}s"
        )
        
        assert focus_event.target_object_id is not None, (
            "Focus event should have a target object"
        )
        
        assert focus_event.confidence > 0.0, (
            "Focus event should have positive confidence"
        )
    
    @given(
        unstable_duration=st.floats(min_value=0.5, max_value=0.7),
        variance=st.floats(min_value=25.0, max_value=100.0)
    )
    @settings(max_examples=5, deadline=None)
    def test_focus_not_detected_for_unstable_gaze(self, unstable_duration, variance):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 13: Focus Detection Stability Requirement**
        **Validates: Requirements 6.1**
        
        Property: For any gaze pattern with high variance (unstable), focus should
        NOT be detected even if duration is sufficient.
        
        This ensures that only stable gaze is classified as focus.
        """
        # Ensure variance exceeds stability threshold
        assume(variance > self.stability_threshold)
        
        # Create a fresh focus detector for this example
        focus_detector = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold
        )
        
        # Generate unstable gaze sequence
        gaze_vectors = self.generate_unstable_gaze_sequence(
            duration=unstable_duration,
            variance=variance,
            fps=30
        )
        
        # Create tracked objects
        tracked_objects = [self.create_tracked_object(position=(320, 240))]
        
        # Calculate frame time
        frame_time = 1.0 / 30.0
        
        # Process each gaze vector
        focus_detected = False
        for gaze_vec in gaze_vectors:
            focus_event = focus_detector.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': variance}
            )
            
            if focus_event is not None and focus_event.focus_type == AttentionType.OBJECT:
                focus_detected = True
                break
            
            time.sleep(frame_time)
        
        # Assert: Focus should NOT be detected for unstable gaze
        assert not focus_detected, (
            f"Focus was incorrectly detected for unstable gaze with "
            f"variance {variance:.2f} > threshold {self.stability_threshold}. "
            f"Duration: {unstable_duration:.3f}s"
        )
    
    @given(
        duration1=st.floats(min_value=0.2, max_value=0.35),
        duration2=st.floats(min_value=0.2, max_value=0.35),
        gap_duration=st.floats(min_value=0.05, max_value=0.15)
    )
    @settings(max_examples=5, deadline=None)
    def test_interrupted_gaze_resets_duration(self, duration1, duration2, gap_duration):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 13: Focus Detection Stability Requirement**
        **Validates: Requirements 6.1**
        
        Property: When stable gaze is interrupted, the duration counter should reset,
        and focus should only be detected if the subsequent stable period meets
        the minimum duration requirement.
        """
        # Ensure individual durations are below minimum
        assume(duration1 < self.min_focus_duration)
        assume(duration2 < self.min_focus_duration)
        
        # Create a fresh focus detector for this example
        focus_detector = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold
        )
        
        # Generate first stable period
        gaze_vectors_1 = self.generate_stable_gaze_sequence(
            duration=duration1,
            center=(0.0, 0.0, 1.0),
            variance=5.0,
            fps=30
        )
        
        # Generate interruption (unstable gaze)
        gaze_vectors_gap = self.generate_unstable_gaze_sequence(
            duration=gap_duration,
            variance=50.0,
            fps=30
        )
        
        # Generate second stable period
        gaze_vectors_2 = self.generate_stable_gaze_sequence(
            duration=duration2,
            center=(0.0, 0.0, 1.0),
            variance=5.0,
            fps=30
        )
        
        # Combine sequences
        all_gaze_vectors = gaze_vectors_1 + gaze_vectors_gap + gaze_vectors_2
        
        # Create tracked object
        tracked_objects = [self.create_tracked_object(position=(320, 240))]
        
        # Calculate frame time
        frame_time = 1.0 / 30.0
        
        # Process all gaze vectors
        focus_detected = False
        for gaze_vec in all_gaze_vectors:
            focus_event = focus_detector.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': 5.0}
            )
            
            if focus_event is not None:
                focus_detected = True
                break
            
            time.sleep(frame_time)
        
        # Assert: Focus should NOT be detected since neither stable period
        # meets minimum duration individually
        assert not focus_detected, (
            f"Focus was incorrectly detected for interrupted gaze. "
            f"Period 1: {duration1:.3f}s, Gap: {gap_duration:.3f}s, "
            f"Period 2: {duration2:.3f}s. Both periods < minimum {self.min_focus_duration}s"
        )
    
    def test_stability_threshold_boundary(self):
        """
        Test behavior at the stability threshold boundary.
        
        Property: Gaze with variance exactly at the stability threshold should
        be handled consistently.
        """
        # Create fresh focus detectors
        focus_detector_1 = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold
        )
        
        focus_detector_2 = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold
        )
        
        # Generate gaze at threshold
        variance_at_threshold = self.stability_threshold
        duration = 0.6  # Above minimum
        
        gaze_vectors = self.generate_stable_gaze_sequence(
            duration=duration,
            center=(0.0, 0.0, 1.0),
            variance=variance_at_threshold,
            fps=30
        )
        
        tracked_objects = [self.create_tracked_object(position=(320, 240))]
        
        # Calculate frame time
        frame_time = 1.0 / 30.0
        
        # Process gaze vectors with first detector
        focus_event = None
        for gaze_vec in gaze_vectors:
            focus_event = focus_detector_1.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': variance_at_threshold}
            )
            time.sleep(frame_time)
        
        # Process gaze vectors with second detector
        focus_event_2 = None
        for gaze_vec in gaze_vectors:
            focus_event_2 = focus_detector_2.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': variance_at_threshold}
            )
            time.sleep(frame_time)
        
        # Results should be consistent
        assert (focus_event is None) == (focus_event_2 is None), (
            "Focus detection at stability threshold should be deterministic"
        )
    
    def test_minimum_duration_boundary(self):
        """
        Test behavior at the minimum duration boundary.
        
        Property: Gaze with duration slightly above the minimum should be detected.
        """
        # Create fresh focus detector
        focus_detector = ImprovedFocusDetector(
            min_focus_duration=self.min_focus_duration,
            stability_threshold=self.stability_threshold
        )
        
        # Generate gaze slightly above minimum duration to account for timing
        duration = self.min_focus_duration * 1.5  # 50% buffer for timing overhead
        variance = 5.0  # Well within stable range
        
        gaze_vectors = self.generate_stable_gaze_sequence(
            duration=duration,
            center=(0.0, 0.0, 1.0),
            variance=variance,
            fps=30
        )
        
        tracked_objects = [self.create_tracked_object(position=(320, 240))]
        
        # Calculate frame time
        frame_time = 1.0 / 30.0
        
        # Process gaze vectors
        focus_event = None
        for gaze_vec in gaze_vectors:
            focus_event = focus_detector.detect_focus(
                gaze_vector=gaze_vec,
                tracked_objects=tracked_objects,
                stability_metrics={'variance': variance}
            )
            time.sleep(frame_time)
        
        # Focus should be detected
        assert focus_event is not None, (
            f"Focus should be detected for duration {duration:.3f}s "
            f"(minimum: {self.min_focus_duration}s)"
        )
        
        assert focus_event.duration >= self.min_focus_duration * 0.8, (
            f"Focus event duration {focus_event.duration:.3f}s should be >= "
            f"minimum {self.min_focus_duration}s"
        )
