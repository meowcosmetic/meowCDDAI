"""
Real-time performance optimization for the AI-Enhanced Gaze Tracking System.

This module provides GPU acceleration, adaptive processing complexity,
memory management, and performance monitoring to meet real-time requirements.

Requirements: 7.1, 7.2, 7.3, 7.4
"""

import time
import logging
import gc
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any

from ...core.data_models import ProcessingMetrics

logger = logging.getLogger(__name__)


class ProcessingComplexity(Enum):
    """Processing complexity levels for adaptive scaling."""
    FULL = "full"        # All features enabled, highest accuracy
    HIGH = "high"        # Most features enabled
    MEDIUM = "medium"    # Core features only
    LOW = "low"          # Minimal processing, basic tracking only


@dataclass
class PerformanceSnapshot:
    """A single performance measurement snapshot."""
    timestamp: float
    fps: float
    latency_ms: float
    memory_mb: float
    complexity: ProcessingComplexity
    dropped_frames: int = 0


class PerformanceOptimizer:
    """
    Manages real-time performance for the gaze tracking pipeline.

    Responsibilities:
    - Track FPS and latency over a rolling window
    - Adaptively scale processing complexity when load is high (Req 7.2, 7.4)
    - Manage memory by triggering GC when usage exceeds limits (Req 7.4)
    - Provide GPU availability detection (Req 7.3)
    - Expose performance diagnostics (Req 7.1)
    """

    # FPS thresholds that trigger complexity changes
    _SCALE_DOWN_FPS_THRESHOLD = 25.0   # Below this → reduce complexity
    _SCALE_UP_FPS_THRESHOLD = 28.0     # Above this → allow increasing complexity

    # Complexity ordering for up/down scaling
    _COMPLEXITY_ORDER: List[ProcessingComplexity] = [
        ProcessingComplexity.LOW,
        ProcessingComplexity.MEDIUM,
        ProcessingComplexity.HIGH,
        ProcessingComplexity.FULL,
    ]

    def __init__(
        self,
        target_fps: float = 30.0,
        memory_limit_mb: float = 2048.0,
        window_size: int = 30,
        adaptation_cooldown_s: float = 2.0,
    ):
        """
        Args:
            target_fps: Desired processing frame rate.
            memory_limit_mb: Memory usage limit in MB before cleanup is triggered.
            window_size: Number of recent frames used for rolling FPS/latency.
            adaptation_cooldown_s: Minimum seconds between complexity changes.
        """
        self.target_fps = target_fps
        self.memory_limit_mb = memory_limit_mb
        self.window_size = window_size
        self.adaptation_cooldown_s = adaptation_cooldown_s

        self._complexity = ProcessingComplexity.FULL
        self._frame_times: deque = deque(maxlen=window_size)
        self._latencies_ms: deque = deque(maxlen=window_size)
        self._dropped_frames: int = 0
        self._last_adaptation_time: float = 0.0
        self._lock = threading.Lock()

        # GPU availability (detected once at startup)
        self._gpu_available: bool = self._detect_gpu()

        logger.info(
            "PerformanceOptimizer initialized. target_fps=%.1f, "
            "memory_limit_mb=%.0f, gpu_available=%s",
            target_fps, memory_limit_mb, self._gpu_available,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_frame(self, latency_ms: float, timestamp: Optional[float] = None) -> None:
        """
        Record timing for a processed frame.

        Call this once per frame after processing completes.

        Args:
            latency_ms: Wall-clock time taken to process the frame (ms).
            timestamp: Optional explicit timestamp (monotonic seconds). If None,
                       the current time is used. Providing explicit timestamps
                       allows deterministic testing without real sleeps.
        """
        with self._lock:
            now = timestamp if timestamp is not None else time.monotonic()
            self._frame_times.append(now)
            self._latencies_ms.append(latency_ms)

            # Drop frame if latency exceeds 2× target budget
            budget_ms = 1000.0 / self.target_fps
            if latency_ms > budget_ms * 2:
                self._dropped_frames += 1

    def get_current_fps(self) -> float:
        """Return rolling average FPS over the last *window_size* frames."""
        with self._lock:
            return self._compute_fps()

    def get_average_latency_ms(self) -> float:
        """Return rolling average processing latency in milliseconds."""
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            return sum(self._latencies_ms) / len(self._latencies_ms)

    def get_complexity(self) -> ProcessingComplexity:
        """Return the current processing complexity level."""
        return self._complexity

    def adapt_complexity(self) -> ProcessingComplexity:
        """
        Evaluate current performance and adjust complexity if needed.

        Implements Property 18: Adaptive Performance Scaling.
        - If FPS < scale-down threshold → reduce complexity
        - If FPS > scale-up threshold and not at FULL → increase complexity

        Returns:
            The (possibly updated) complexity level.
        """
        with self._lock:
            now = time.monotonic()
            if now - self._last_adaptation_time < self.adaptation_cooldown_s:
                return self._complexity

            fps = self._compute_fps()
            if fps <= 0:
                return self._complexity

            current_idx = self._COMPLEXITY_ORDER.index(self._complexity)

            if fps < self._SCALE_DOWN_FPS_THRESHOLD and current_idx > 0:
                self._complexity = self._COMPLEXITY_ORDER[current_idx - 1]
                self._last_adaptation_time = now
                logger.warning(
                    "Performance degraded (%.1f FPS < %.1f). "
                    "Reducing complexity to %s.",
                    fps, self._SCALE_DOWN_FPS_THRESHOLD, self._complexity.value,
                )

            elif fps >= self._SCALE_UP_FPS_THRESHOLD and current_idx < len(self._COMPLEXITY_ORDER) - 1:
                self._complexity = self._COMPLEXITY_ORDER[current_idx + 1]
                self._last_adaptation_time = now
                logger.info(
                    "Performance recovered (%.1f FPS). "
                    "Increasing complexity to %s.",
                    fps, self._complexity.value,
                )

            return self._complexity

    def manage_memory(self, current_usage_mb: float) -> bool:
        """
        Trigger memory cleanup if usage exceeds the configured limit.

        Args:
            current_usage_mb: Current memory usage in MB.

        Returns:
            True if cleanup was triggered, False otherwise.
        """
        if current_usage_mb > self.memory_limit_mb:
            logger.warning(
                "Memory usage %.0f MB exceeds limit %.0f MB. Running GC.",
                current_usage_mb, self.memory_limit_mb,
            )
            gc.collect()
            return True
        return False

    def get_metrics(self, memory_mb: float = 0.0) -> ProcessingMetrics:
        """
        Return a snapshot of current performance metrics.

        Args:
            memory_mb: Current memory usage (caller-supplied).

        Returns:
            ProcessingMetrics dataclass instance.
        """
        with self._lock:
            fps = self._compute_fps()
            latency = (
                sum(self._latencies_ms) / len(self._latencies_ms)
                if self._latencies_ms else 0.0
            )
            return ProcessingMetrics(
                fps=fps,
                latency_ms=latency,
                memory_usage_mb=memory_mb,
                dropped_frames=self._dropped_frames,
                processing_time_breakdown={
                    "complexity": self._complexity.value,
                    "window_size": len(self._frame_times),
                },
            )

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return detailed diagnostic information for troubleshooting.

        Implements Requirement 7.5 (performance diagnostics).
        """
        with self._lock:
            fps = self._compute_fps()
            latency = (
                sum(self._latencies_ms) / len(self._latencies_ms)
                if self._latencies_ms else 0.0
            )
            return {
                "current_fps": fps,
                "target_fps": self.target_fps,
                "fps_deficit": max(0.0, self.target_fps - fps),
                "average_latency_ms": latency,
                "target_latency_ms": 1000.0 / self.target_fps,
                "complexity": self._complexity.value,
                "dropped_frames": self._dropped_frames,
                "gpu_available": self._gpu_available,
                "memory_limit_mb": self.memory_limit_mb,
                "samples_collected": len(self._frame_times),
            }

    @property
    def gpu_available(self) -> bool:
        """Whether GPU acceleration is available on this system."""
        return self._gpu_available

    def reset_stats(self) -> None:
        """Reset all rolling statistics (useful between sessions)."""
        with self._lock:
            self._frame_times.clear()
            self._latencies_ms.clear()
            self._dropped_frames = 0
            self._last_adaptation_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_fps(self) -> float:
        """Compute FPS from the rolling frame-time window (must hold lock)."""
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed

    @staticmethod
    def _detect_gpu() -> bool:
        """Detect whether a CUDA-capable GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        try:
            import cv2  # noqa: F401
            count = cv2.cuda.getCudaEnabledDeviceCount()
            return count > 0
        except Exception:
            pass
        return False
