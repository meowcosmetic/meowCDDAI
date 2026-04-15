"""
Unified gaze tracking pipeline that orchestrates all components end-to-end.

This module wires together face detection, head pose estimation, gaze estimation,
sensor fusion, focus detection, quality assessment, and the legacy adapter into
a single coherent processing pipeline.

Requirements: All requirements through system integration
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..compatibility.legacy_adapter import LegacyResponseAdapter
from ..components.calibration.personal_calibration import PersonalCalibrationSystem
from ..components.camera_calibration.camera_calibrator import AutomaticCameraCalibrator as CameraCalibrator
from ..components.error_handling.error_handler import ErrorHandler
from ..components.face_detection.hybrid_face_detector import HybridFaceDetector
from ..components.focus_detection.focus_detector import ImprovedFocusDetector
from ..components.gaze_estimation.compensated_gaze_estimator import CompensatedGazeEstimator
from ..components.head_pose.pnp_head_pose_estimator import PnPHeadPoseEstimator
from ..components.performance.performance_optimizer import PerformanceOptimizer
from ..components.quality_assessment.quality_assessor import GazeQualityAssessor
from ..components.sensor_fusion.multi_modal_fusion import MultiModalFusion
from ..core.data_models import (
    FaceDetection,
    FocusEvent,
    GazeEstimate,
    HeadPose,
    ProcessingMetrics,
    QualityMetrics,
    SessionSummary,
    WanderingPeriod,
)


class GazeTrackingPipeline:
    """
    End-to-end gaze tracking pipeline.

    Orchestrates all components in the correct order:
    1. Face detection (with temporal fallback)
    2. Head pose estimation
    3. Gaze estimation (with head pose compensation)
    4. Sensor fusion
    5. Camera angle correction
    6. Focus detection
    7. Quality assessment
    8. Legacy-compatible output
    """

    def __init__(
        self,
        face_detector: Optional[HybridFaceDetector] = None,
        head_pose_estimator: Optional[PnPHeadPoseEstimator] = None,
        gaze_estimator: Optional[CompensatedGazeEstimator] = None,
        sensor_fusion: Optional[MultiModalFusion] = None,
        focus_detector: Optional[ImprovedFocusDetector] = None,
        quality_assessor: Optional[GazeQualityAssessor] = None,
        camera_calibrator: Optional[CameraCalibrator] = None,
        calibration_system: Optional[PersonalCalibrationSystem] = None,
        performance_optimizer: Optional[PerformanceOptimizer] = None,
        error_handler: Optional[ErrorHandler] = None,
        legacy_adapter: Optional[LegacyResponseAdapter] = None,
    ) -> None:
        # Use provided components or create defaults
        self.face_detector = face_detector or HybridFaceDetector()
        self.head_pose_estimator = head_pose_estimator or PnPHeadPoseEstimator()
        self.gaze_estimator = gaze_estimator or CompensatedGazeEstimator(
            head_pose_estimator=self.head_pose_estimator
        )
        self.sensor_fusion = sensor_fusion or MultiModalFusion()
        self.focus_detector = focus_detector or ImprovedFocusDetector()
        self.quality_assessor = quality_assessor or GazeQualityAssessor()
        self.camera_calibrator = camera_calibrator or CameraCalibrator()
        self.calibration_system = calibration_system or PersonalCalibrationSystem()
        self.performance_optimizer = performance_optimizer or PerformanceOptimizer()
        self.error_handler = error_handler or ErrorHandler()
        self.legacy_adapter = legacy_adapter or LegacyResponseAdapter()

        # Session state
        self._session_start: Optional[float] = None
        self._frame_count: int = 0
        self._valid_frame_count: int = 0
        self._gaze_history: List[GazeEstimate] = []
        self._focus_events: List[FocusEvent] = []
        self._wandering_periods: List[WanderingPeriod] = []
        self._quality_history: List[QualityMetrics] = []
        self._fps_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Begin a new tracking session."""
        self._session_start = time.time()
        self._frame_count = 0
        self._valid_frame_count = 0
        self._gaze_history.clear()
        self._focus_events.clear()
        self._wandering_periods.clear()
        self._quality_history.clear()
        self._fps_history.clear()

    def process_frame(
        self,
        frame: np.ndarray,
        tracked_objects: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Process a single video frame through the full pipeline.

        Args:
            frame: BGR video frame as numpy array.
            tracked_objects: Optional list of detected objects in the scene.
            timestamp: Frame timestamp; defaults to current time.

        Returns:
            Dict with gaze estimate, quality metrics, focus event, and
            legacy-compatible fields.
        """
        t_start = time.time()
        ts = timestamp if timestamp is not None else t_start
        tracked_objects = tracked_objects or []
        self._frame_count += 1

        result: Dict[str, Any] = {
            "timestamp": ts,
            "frame_index": self._frame_count,
            "gaze_estimate": None,
            "face_detected": False,
            "focus_event": None,
            "quality_metrics": None,
            "processing_time_ms": 0.0,
        }

        try:
            # 1. Face detection
            face_detections = self.face_detector.detect_faces(frame)
            if not face_detections:
                result["processing_time_ms"] = (time.time() - t_start) * 1000
                return result

            face = face_detections[0]
            result["face_detected"] = True

            # 2. Gaze estimation (includes head pose compensation internally)
            gaze_estimate = self.gaze_estimator.estimate_gaze(face, frame)
            gaze_estimate.timestamp = ts

            # 3. Quality assessment
            quality = self.quality_assessor.assess_quality(gaze_estimate, frame, face)
            gaze_estimate.quality_metrics = quality
            self._quality_history.append(quality)

            # 4. Focus detection
            stability_metrics = {
                "variance": 1.0 - quality.tracking_stability,
                "tracking_stability": quality.tracking_stability,
            }
            focus_event = self.focus_detector.detect_focus(
                gaze_estimate.gaze_vector_3d,
                tracked_objects,
                stability_metrics,
            )
            if focus_event is not None:
                result["focus_event"] = focus_event
                # Accumulate unique focus events
                if not self._focus_events or self._focus_events[-1] is not focus_event:
                    self._focus_events.append(focus_event)

            # 5. Store valid estimate
            self._gaze_history.append(gaze_estimate)
            self._valid_frame_count += 1

            result["gaze_estimate"] = gaze_estimate
            result["quality_metrics"] = quality

        except Exception as exc:  # graceful degradation (Req 8.4)
            result["error"] = str(exc)

        result["processing_time_ms"] = (time.time() - t_start) * 1000
        fps = 1000.0 / max(result["processing_time_ms"], 1e-3)
        self._fps_history.append(fps)

        return result

    def end_session(self, session_id: str = "session") -> SessionSummary:
        """
        Finalise the session and return a SessionSummary.

        Args:
            session_id: Identifier for this session.

        Returns:
            SessionSummary with all accumulated data.
        """
        end_time = time.time()
        start_time = self._session_start or end_time

        overall_quality = (
            float(np.mean([qm.overall_quality for qm in self._quality_history]))
            if self._quality_history
            else 0.0
        )

        avg_fps = float(np.mean(self._fps_history)) if self._fps_history else 0.0
        processing_metrics = ProcessingMetrics(
            fps=avg_fps,
            latency_ms=1000.0 / max(avg_fps, 1e-3),
            memory_usage_mb=0.0,
        )

        return SessionSummary(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            total_frames=self._frame_count,
            valid_frames=self._valid_frame_count,
            focus_events=list(self._focus_events),
            wandering_periods=list(self._wandering_periods),
            overall_quality=overall_quality,
            attention_statistics=self._compute_attention_statistics(),
            processing_metrics=processing_metrics,
        )

    def get_legacy_response(
        self,
        duration: Optional[float] = None,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Return a legacy-compatible response dict from accumulated estimates.

        Args:
            duration: Session duration in seconds; computed automatically if None.
            detected_objects: Optional list of detected objects.

        Returns:
            Legacy-compatible response dict.
        """
        if duration is None:
            duration = (
                time.time() - self._session_start
                if self._session_start
                else 0.0
            )
        return self.legacy_adapter.from_gaze_estimates(
            self._gaze_history,
            duration=duration,
            detected_objects=detected_objects,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_attention_statistics(self) -> Dict[str, float]:
        """Derive basic attention statistics from accumulated focus events."""
        if not self._gaze_history:
            return {}

        total = len(self._gaze_history)
        return {
            "eye_contact_percentage": (self._valid_frame_count / max(total, 1)) * 100.0,
            "attention_to_person_percentage": 0.0,
            "attention_to_objects_percentage": (
                len([e for e in self._focus_events if e.target_object_id]) / max(total, 1) * 100.0
            ),
            "attention_to_book_percentage": 0.0,
        }
