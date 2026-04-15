"""
Property-based tests for adaptive performance scaling.

**Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
**Validates: Requirements 7.2, 7.4**

Property: For any high computational load situation, the system should
automatically reduce processing complexity while maintaining core functionality.
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

_COMPLEXITY_ORDER = [
    ProcessingComplexity.LOW,
    ProcessingComplexity.MEDIUM,
    ProcessingComplexity.HIGH,
    ProcessingComplexity.FULL,
]


def complexity_rank(c: ProcessingComplexity) -> int:
    return _COMPLEXITY_ORDER.index(c)


def feed_frames_at_fps(
    optimizer: PerformanceOptimizer,
    fps: float,
    n_frames: int,
    start_ts: float = 0.0,
) -> None:
    """Feed frames with injected timestamps — no real sleeping."""
    interval_s = 1.0 / fps
    latency_ms = interval_s * 1000.0
    for i in range(n_frames):
        ts = start_ts + i * interval_s
        optimizer.record_frame(latency_ms, timestamp=ts)


# ---------------------------------------------------------------------------
# Property 18 tests
# ---------------------------------------------------------------------------

class TestAdaptivePerformanceScaling:
    """
    **Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
    **Validates: Requirements 7.2, 7.4**
    """

    @given(
        low_fps=st.floats(min_value=10.0, max_value=22.0),
        n_frames=st.integers(min_value=10, max_value=30),
    )
    @settings(max_examples=100, deadline=None)
    def test_complexity_reduces_under_high_load(self, low_fps, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
        **Validates: Requirements 7.2**

        Property: For any processing rate below the scale-down threshold (25 FPS),
        the system should reduce its processing complexity level.
        """
        assume(low_fps < 25.0)

        optimizer = PerformanceOptimizer(
            target_fps=30.0,
            window_size=n_frames + 5,
            adaptation_cooldown_s=0.0,
        )

        initial_complexity = optimizer.get_complexity()
        initial_rank = complexity_rank(initial_complexity)

        feed_frames_at_fps(optimizer, low_fps, n_frames)
        new_complexity = optimizer.adapt_complexity()
        new_rank = complexity_rank(new_complexity)

        if initial_rank == 0:
            assert new_rank == 0, "Already at minimum complexity, should stay there"
        else:
            assert new_rank < initial_rank, (
                f"Complexity should decrease under {low_fps:.1f} FPS load. "
                f"Was {initial_complexity.value}, now {new_complexity.value}"
            )

    @given(
        high_fps=st.floats(min_value=28.0, max_value=60.0),
        n_frames=st.integers(min_value=10, max_value=30),
    )
    @settings(max_examples=100, deadline=5000)
    def test_complexity_increases_when_performance_recovers(self, high_fps, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
        **Validates: Requirements 7.2**

        Property: When performance recovers (FPS >= 28), the system should be
        able to increase its processing complexity level.
        """
        assume(high_fps >= 28.0)

        optimizer = PerformanceOptimizer(
            target_fps=30.0,
            window_size=n_frames + 5,
            adaptation_cooldown_s=0.0,
        )

        # Force complexity down to LOW first
        optimizer._complexity = ProcessingComplexity.LOW

        feed_frames_at_fps(optimizer, high_fps, n_frames)
        new_complexity = optimizer.adapt_complexity()
        new_rank = complexity_rank(new_complexity)

        assert new_rank > complexity_rank(ProcessingComplexity.LOW), (
            f"Complexity should increase when FPS recovers to {high_fps:.1f}. "
            f"Got {new_complexity.value}"
        )

    @given(
        low_fps=st.floats(min_value=5.0, max_value=22.0),
        n_frames=st.integers(min_value=10, max_value=30),
        n_cycles=st.integers(min_value=2, max_value=6),
    )
    @settings(max_examples=100, deadline=5000)
    def test_complexity_never_goes_below_minimum(self, low_fps, n_frames, n_cycles):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
        **Validates: Requirements 7.2, 7.4**

        Property: For any load level, the system should never reduce complexity
        below the minimum (LOW) level — core functionality is always maintained.
        """
        optimizer = PerformanceOptimizer(
            target_fps=30.0,
            window_size=n_frames + 5,
            adaptation_cooldown_s=0.0,
        )

        ts = 0.0
        for _ in range(n_cycles):
            feed_frames_at_fps(optimizer, low_fps, n_frames, start_ts=ts)
            ts += n_frames / low_fps
            optimizer.adapt_complexity()

        final_complexity = optimizer.get_complexity()
        assert final_complexity in _COMPLEXITY_ORDER
        assert complexity_rank(final_complexity) >= 0, (
            "Complexity should never go below LOW (core functionality preserved)"
        )

    @given(
        memory_usage_mb=st.floats(min_value=100.0, max_value=3000.0),
    )
    @settings(max_examples=100, deadline=5000)
    def test_memory_cleanup_triggered_above_limit(self, memory_usage_mb):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
        **Validates: Requirements 7.4**

        Property: For any memory usage above the configured limit, the system
        should trigger cleanup. For usage below the limit, no cleanup occurs.
        """
        limit_mb = 2048.0
        optimizer = PerformanceOptimizer(
            target_fps=30.0,
            memory_limit_mb=limit_mb,
        )

        cleanup_triggered = optimizer.manage_memory(memory_usage_mb)

        if memory_usage_mb > limit_mb:
            assert cleanup_triggered, (
                f"Memory cleanup should be triggered when usage "
                f"{memory_usage_mb:.0f} MB > limit {limit_mb:.0f} MB"
            )
        else:
            assert not cleanup_triggered, (
                f"Memory cleanup should NOT be triggered when usage "
                f"{memory_usage_mb:.0f} MB <= limit {limit_mb:.0f} MB"
            )

    @given(n_frames=st.integers(min_value=5, max_value=30))
    @settings(max_examples=100, deadline=5000)
    def test_metrics_reflect_complexity_level(self, n_frames):
        """
        **Feature: ai-enhanced-gaze-tracking, Property 18: Adaptive Performance Scaling**
        **Validates: Requirements 7.2**

        Property: For any processing session, the metrics returned by the
        optimizer should always include the current complexity level.
        """
        optimizer = PerformanceOptimizer(target_fps=30.0)
        feed_frames_at_fps(optimizer, fps=30.0, n_frames=n_frames)

        metrics = optimizer.get_metrics(memory_mb=512.0)

        assert metrics.processing_time_breakdown is not None
        assert "complexity" in metrics.processing_time_breakdown
        assert metrics.processing_time_breakdown["complexity"] in [
            c.value for c in ProcessingComplexity
        ]

    def test_adaptation_cooldown_prevents_rapid_changes(self):
        """The cooldown period should prevent complexity from changing on every frame."""
        optimizer = PerformanceOptimizer(
            target_fps=30.0,
            window_size=15,
            adaptation_cooldown_s=1000.0,  # Very long cooldown
        )

        # First adaptation cycle
        feed_frames_at_fps(optimizer, fps=15.0, n_frames=12)
        first_complexity = optimizer.adapt_complexity()

        # Second cycle — cooldown blocks further changes
        feed_frames_at_fps(optimizer, fps=15.0, n_frames=12, start_ts=1.0)
        second_complexity = optimizer.adapt_complexity()

        assert first_complexity == second_complexity, (
            "Cooldown should prevent rapid complexity changes. "
            f"First: {first_complexity.value}, Second: {second_complexity.value}"
        )

    def test_reset_clears_statistics(self):
        """After reset, FPS and latency should return to zero."""
        optimizer = PerformanceOptimizer(target_fps=30.0)
        feed_frames_at_fps(optimizer, fps=30.0, n_frames=10)

        assert optimizer.get_current_fps() > 0

        optimizer.reset_stats()

        assert optimizer.get_current_fps() == 0.0
        assert optimizer.get_average_latency_ms() == 0.0
        assert optimizer.get_metrics().dropped_frames == 0
