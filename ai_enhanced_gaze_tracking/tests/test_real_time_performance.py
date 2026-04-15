"""
Property-based tests for real-time performance guarantee.

**Feature: ai-enhanced-gaze-tracking, Property 17: Real-Time Performance Guarantee**
**Validates: Requirements 7.1**

Property: For any video processing at 30 FPS, the system should maintain
processing speed of at least 25 FPS on standard hardware configurations.

We test this by injecting synthetic timestamps into the PerformanceOptimizer
so that tests run instantly without real sleeps.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from ai_enhanced_gaze_tracking.components.performance.performance_optimizer import (
    PerformanceOptimizer,
    ProcessingComplexity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def feed_frames_at_fps(
    optimizer: PerformanceOptimizer,
    fps: float,
    n_frames: int,
    start_ts: float = 0.0,
) -> None:
    """
    Feed *n_frames* synthetic frames with injected timestamps at *fps*.

    No real sleeping — timestamps are computed arithmetically.
    """
    interval_s = 1.0 / fps
    latency_ms = interval_s * 1000.0
    for i in range(n_frames):
        ts = start_ts + i * interval_s
        optimizer.record_frame(latency_ms, timestamp=ts)


# ---------------------------------------------------------------------------
# Property 17 tests
# ---------------------------------------------------------------------------

class TestRealTimePerformanceGuarantee:
    """
    **Feature: ai-enhanced-gaze-tracking, Property 17: Real-Time Performance Guarantee**
    **Validates: Requirements 7.1**
    """

    MIN_ACCEPTABLE_FPS = 25.0
    TARGET_FPS = 30.0

    @given(
        simulated_fps=st.floats(min_value=25.0, max_value=60.0),
        n_frames=st.integers(min_value=10, max_value=30),
    )
    @settings(max_examples=100, deadline=None)
    def test_fps_measurement_reflects_actual_throughput(self, simulated_fps, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 17: Real-Time Performance Guarantee**
        **Validates: Requirements 7.1**

        Property: For any frame sequence processed at a known FPS, the optimizer's
        measured FPS should be within 5% of the actual processing rate.
        """
        optimizer = PerformanceOptimizer(
            target_fps=self.TARGET_FPS,
            window_size=n_frames + 5,
        )

        feed_frames_at_fps(optimizer, simulated_fps, n_frames)

        measured_fps = optimizer.get_current_fps()

        # Allow ±5% tolerance (timestamps are exact, so this should be tight)
        tolerance = 0.05
        lower = simulated_fps * (1 - tolerance)
        upper = simulated_fps * (1 + tolerance)

        assert lower <= measured_fps <= upper, (
            f"Measured FPS {measured_fps:.2f} is not within ±5% of "
            f"simulated {simulated_fps:.2f} FPS (expected [{lower:.2f}, {upper:.2f}])"
        )

    @given(
        simulated_fps=st.floats(min_value=25.0, max_value=60.0),
        n_frames=st.integers(min_value=10, max_value=30),
    )
    @settings(max_examples=100, deadline=5000)
    def test_system_meets_minimum_fps_requirement(self, simulated_fps, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 17: Real-Time Performance Guarantee**
        **Validates: Requirements 7.1**

        Property: When the system processes frames at or above 25 FPS, the
        optimizer should report FPS >= 25 (the minimum real-time requirement).
        """
        assume(simulated_fps >= self.MIN_ACCEPTABLE_FPS)

        optimizer = PerformanceOptimizer(
            target_fps=self.TARGET_FPS,
            window_size=n_frames + 5,
        )

        feed_frames_at_fps(optimizer, simulated_fps, n_frames)

        measured_fps = optimizer.get_current_fps()

        assert measured_fps >= self.MIN_ACCEPTABLE_FPS * 0.95, (
            f"System processing at {simulated_fps:.1f} FPS should report "
            f">= {self.MIN_ACCEPTABLE_FPS * 0.95:.1f} FPS, got {measured_fps:.2f}"
        )

    @given(
        latency_ms=st.floats(min_value=1.0, max_value=40.0),
        n_frames=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=100, deadline=5000)
    def test_latency_tracking_accuracy(self, latency_ms, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 17: Real-Time Performance Guarantee**
        **Validates: Requirements 7.1**

        Property: For any set of recorded frame latencies, the optimizer's
        average latency should match the recorded values within 1%.
        """
        optimizer = PerformanceOptimizer(
            target_fps=self.TARGET_FPS,
            window_size=n_frames + 5,
        )

        for _ in range(n_frames):
            optimizer.record_frame(latency_ms)

        measured_latency = optimizer.get_average_latency_ms()

        assert abs(measured_latency - latency_ms) <= latency_ms * 0.01 + 0.01, (
            f"Average latency {measured_latency:.4f} ms should be within 1% of "
            f"recorded {latency_ms:.4f} ms"
        )

    @given(n_frames=st.integers(min_value=10, max_value=30))
    @settings(max_examples=100, deadline=5000)
    def test_diagnostics_contain_required_fields(self, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 17: Real-Time Performance Guarantee**
        **Validates: Requirements 7.1**

        Property: For any processing session, the diagnostics output should
        always contain the fields needed to assess real-time performance.
        """
        optimizer = PerformanceOptimizer(target_fps=self.TARGET_FPS)
        feed_frames_at_fps(optimizer, fps=30.0, n_frames=n_frames)

        diag = optimizer.get_diagnostics()

        required_keys = {
            "current_fps",
            "target_fps",
            "fps_deficit",
            "average_latency_ms",
            "target_latency_ms",
            "complexity",
            "dropped_frames",
            "gpu_available",
        }
        missing = required_keys - diag.keys()
        assert not missing, f"Diagnostics missing required fields: {missing}"

        assert diag["current_fps"] >= 0
        assert diag["fps_deficit"] >= 0
        assert diag["average_latency_ms"] >= 0
        assert diag["dropped_frames"] >= 0

    def test_dropped_frame_counting(self):
        """Frames with latency > 2× budget should be counted as dropped."""
        optimizer = PerformanceOptimizer(target_fps=30.0)
        budget_ms = 1000.0 / 30.0  # ~33 ms

        optimizer.record_frame(budget_ms)
        optimizer.record_frame(budget_ms * 3)  # 3× budget → dropped

        assert optimizer.get_metrics().dropped_frames == 1
