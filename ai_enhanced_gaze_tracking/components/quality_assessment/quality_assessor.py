"""
Data quality assessment system for the AI-Enhanced Gaze Tracking System.

Implements:
- Confidence scoring for all gaze estimates (Req 9.1)
- Comprehensive quality metrics considering multiple factors (Req 9.2)
- Quality-based alerting and user notifications (Req 9.3)
- Data validation and unreliable segment flagging (Req 9.5)

Requirements: 9.1, 9.2, 9.3, 9.5
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ...core.interfaces import QualityAssessor
from ...core.data_models import GazeEstimate, QualityMetrics, FaceDetection, HeadPose

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """Alert generated when tracking quality degrades."""
    severity: str          # "low", "medium", "high"
    message: str           # Human-readable description
    suggested_action: str  # Corrective action for the user
    quality_score: float   # Quality score that triggered the alert


class GazeQualityAssessor(QualityAssessor):
    """
    Comprehensive quality assessor for gaze tracking data.

    Evaluates each gaze estimate across multiple quality dimensions:
    head pose, lighting, occlusion, motion blur, and temporal consistency.
    Flags unreliable segments and generates user-facing alerts.

    Property 21: Confidence Score Completeness
    For any gaze estimate generated, the system should provide confidence
    scores that reflect the reliability of the estimate.

    Property 23: Data Validation Flagging
    For any potentially unreliable data segment, the validation system
    should correctly identify and flag it based on quality metrics.
    """

    # Thresholds for quality dimensions
    _MIN_HEAD_POSE_QUALITY = 0.3   # Below this → poor head pose
    _MAX_PITCH_DEG = 45.0          # Max pitch for good quality
    _MAX_YAW_DEG = 30.0            # Max yaw for good quality
    _MAX_ROLL_DEG = 15.0           # Max roll for good quality

    _DARK_BRIGHTNESS_THRESHOLD = 70.0    # Below → too dark
    _BRIGHT_BRIGHTNESS_THRESHOLD = 210.0 # Above → too bright
    _HIGH_BLUR_THRESHOLD = 100.0         # Laplacian variance below → blurry

    _DEFAULT_QUALITY_THRESHOLD = 0.4     # Default threshold for flagging

    def __init__(
        self,
        quality_threshold: float = 0.4,
        alert_threshold: float = 0.3,
        temporal_window: int = 10,
    ):
        """
        Args:
            quality_threshold: Minimum quality score to consider data reliable.
            alert_threshold: Quality score below which alerts are generated.
            temporal_window: Number of recent frames used for temporal consistency.
        """
        self.quality_threshold = quality_threshold
        self.alert_threshold = alert_threshold
        self.temporal_window = temporal_window
        self._recent_estimates: List[GazeEstimate] = []

    # ------------------------------------------------------------------
    # QualityAssessor interface
    # ------------------------------------------------------------------

    def assess_quality(
        self,
        gaze_estimate: GazeEstimate,
        frame: np.ndarray,
        face_detection: FaceDetection,
    ) -> QualityMetrics:
        """
        Assess quality of a gaze estimate across all relevant dimensions.

        Implements Property 22 (Quality Factor Integration) and
        Property 21 (Confidence Score Completeness).

        Args:
            gaze_estimate: The gaze estimate to assess.
            frame: The video frame associated with the estimate.
            face_detection: The face detection result.

        Returns:
            QualityMetrics with scores for each quality dimension.
        """
        head_pose_quality = self._assess_head_pose_quality(gaze_estimate.head_pose)
        lighting_quality = self._assess_lighting_quality(frame)
        occlusion_level = self._assess_occlusion(face_detection)
        motion_blur = self._assess_motion_blur(frame)
        eye_visibility = self._assess_eye_visibility(face_detection)
        landmark_quality = self._assess_landmark_quality(face_detection)
        temporal_consistency = self._assess_temporal_consistency(gaze_estimate)

        # Update rolling history for future temporal assessments
        self._recent_estimates.append(gaze_estimate)
        if len(self._recent_estimates) > self.temporal_window:
            self._recent_estimates.pop(0)

        # Overall quality: weighted combination of all factors
        # Occlusion and motion blur are "penalty" factors (higher = worse)
        overall_quality = self._compute_overall_quality(
            head_pose_quality=head_pose_quality,
            lighting_quality=lighting_quality,
            occlusion_level=occlusion_level,
            motion_blur=motion_blur,
            eye_visibility=eye_visibility,
            landmark_quality=landmark_quality,
            temporal_consistency=temporal_consistency,
        )

        return QualityMetrics(
            overall_quality=overall_quality,
            head_pose_quality=head_pose_quality,
            lighting_quality=lighting_quality,
            occlusion_level=occlusion_level,
            motion_blur=motion_blur,
            tracking_stability=temporal_consistency,
            eye_visibility=eye_visibility,
            landmark_quality=landmark_quality,
            temporal_consistency=temporal_consistency,
        )

    def validate_data_segment(
        self,
        gaze_sequence: List[GazeEstimate],
        quality_threshold: Optional[float] = None,
    ) -> List[bool]:
        """
        Validate reliability of each estimate in a sequence.

        Args:
            gaze_sequence: Sequence of gaze estimates to validate.
            quality_threshold: Override the instance-level threshold.

        Returns:
            Boolean list — True means the estimate is reliable.
        """
        threshold = quality_threshold if quality_threshold is not None else self.quality_threshold
        return [
            est.quality_metrics.overall_quality >= threshold
            for est in gaze_sequence
        ]

    def flag_unreliable_data(
        self,
        quality_metrics: List[QualityMetrics],
    ) -> List[int]:
        """
        Return indices of unreliable data segments.

        Implements Property 23: Data Validation Flagging.

        Args:
            quality_metrics: Quality assessments for a data sequence.

        Returns:
            Sorted list of indices where quality is below threshold.
        """
        return [
            i
            for i, qm in enumerate(quality_metrics)
            if qm.overall_quality < self.quality_threshold
        ]

    # ------------------------------------------------------------------
    # Alerting (Req 9.3)
    # ------------------------------------------------------------------

    def check_for_alerts(self, quality_metrics: QualityMetrics) -> List[QualityAlert]:
        """
        Generate user-facing alerts when quality degrades.

        Args:
            quality_metrics: Quality metrics to evaluate.

        Returns:
            List of QualityAlert objects (may be empty).
        """
        alerts: List[QualityAlert] = []

        if quality_metrics.overall_quality < self.alert_threshold:
            alerts.append(QualityAlert(
                severity="high",
                message="Gaze tracking quality is critically low.",
                suggested_action=(
                    "Ensure the subject is well-lit, facing the camera, "
                    "and that the camera is unobstructed."
                ),
                quality_score=quality_metrics.overall_quality,
            ))

        if quality_metrics.lighting_quality < 0.4:
            alerts.append(QualityAlert(
                severity="medium",
                message="Poor lighting conditions detected.",
                suggested_action="Improve room lighting or reposition the light source.",
                quality_score=quality_metrics.lighting_quality,
            ))

        if quality_metrics.occlusion_level > 0.5:
            alerts.append(QualityAlert(
                severity="medium",
                message="Significant face occlusion detected.",
                suggested_action="Ensure the subject's face is fully visible to the camera.",
                quality_score=1.0 - quality_metrics.occlusion_level,
            ))

        if quality_metrics.head_pose_quality < 0.3:
            alerts.append(QualityAlert(
                severity="low",
                message="Extreme head pose detected.",
                suggested_action="Ask the subject to face the camera more directly.",
                quality_score=quality_metrics.head_pose_quality,
            ))

        return alerts

    def get_session_quality_summary(
        self,
        quality_sequence: List[QualityMetrics],
    ) -> Dict[str, Any]:
        """
        Compute overall quality metrics for a complete session.

        Implements Requirement 9.4.

        Args:
            quality_sequence: All quality metrics from the session.

        Returns:
            Summary dict with mean scores and reliability percentage.
        """
        if not quality_sequence:
            return {"error": "No quality data available"}

        overall_scores = [qm.overall_quality for qm in quality_sequence]
        reliable_count = sum(1 for s in overall_scores if s >= self.quality_threshold)

        return {
            "mean_overall_quality": float(np.mean(overall_scores)),
            "min_overall_quality": float(np.min(overall_scores)),
            "max_overall_quality": float(np.max(overall_scores)),
            "reliability_percentage": reliable_count / len(quality_sequence) * 100.0,
            "total_frames": len(quality_sequence),
            "reliable_frames": reliable_count,
            "unreliable_frames": len(quality_sequence) - reliable_count,
            "mean_head_pose_quality": float(np.mean([qm.head_pose_quality for qm in quality_sequence])),
            "mean_lighting_quality": float(np.mean([qm.lighting_quality for qm in quality_sequence])),
            "mean_eye_visibility": float(np.mean([qm.eye_visibility for qm in quality_sequence])),
        }

    # ------------------------------------------------------------------
    # Individual quality dimension assessors
    # ------------------------------------------------------------------

    def _assess_head_pose_quality(self, head_pose: HeadPose) -> float:
        """
        Score head pose quality based on rotation angles.

        Extreme angles reduce quality because gaze estimation becomes
        less reliable when the head is significantly rotated.
        """
        yaw_deg = abs(np.degrees(head_pose.yaw))
        pitch_deg = abs(np.degrees(head_pose.pitch))
        roll_deg = abs(np.degrees(head_pose.roll))

        # Penalty factors: 0 at limit, 1 at 0 degrees
        yaw_factor = max(0.0, 1.0 - yaw_deg / self._MAX_YAW_DEG)
        pitch_factor = max(0.0, 1.0 - pitch_deg / self._MAX_PITCH_DEG)
        roll_factor = max(0.0, 1.0 - roll_deg / self._MAX_ROLL_DEG)

        # Also incorporate the pose estimator's own confidence
        pose_confidence = float(np.clip(head_pose.confidence, 0.0, 1.0))

        quality = (yaw_factor * 0.35 + pitch_factor * 0.35 + roll_factor * 0.15 + pose_confidence * 0.15)
        return float(np.clip(quality, 0.0, 1.0))

    def _assess_lighting_quality(self, frame: np.ndarray) -> float:
        """
        Score lighting quality from frame brightness and contrast.

        Both too-dark and too-bright conditions reduce quality.
        """
        if frame is None or frame.size == 0:
            return 0.0

        gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
        brightness = float(gray.mean())

        # Optimal brightness is around 100-160
        if brightness < self._DARK_BRIGHTNESS_THRESHOLD:
            # Too dark — linear penalty from 0 at 0 to 0.7 at threshold
            quality = brightness / self._DARK_BRIGHTNESS_THRESHOLD * 0.7
        elif brightness > self._BRIGHT_BRIGHTNESS_THRESHOLD:
            # Too bright
            excess = brightness - self._BRIGHT_BRIGHTNESS_THRESHOLD
            max_excess = 255.0 - self._BRIGHT_BRIGHTNESS_THRESHOLD
            quality = max(0.0, 1.0 - excess / max_excess)
        else:
            # Good range — score based on distance from optimal (128)
            optimal = 128.0
            span = max(self._BRIGHT_BRIGHTNESS_THRESHOLD - optimal, optimal - self._DARK_BRIGHTNESS_THRESHOLD)
            quality = 1.0 - abs(brightness - optimal) / span * 0.3

        return float(np.clip(quality, 0.0, 1.0))

    def _assess_occlusion(self, face_detection: FaceDetection) -> float:
        """
        Estimate occlusion level from face detection confidence and quality.

        Returns a value in [0, 1] where 1 = fully occluded.
        """
        # Use the face detection's own quality score as a proxy for occlusion
        # Low quality often indicates occlusion
        face_quality = float(np.clip(face_detection.quality_score, 0.0, 1.0))
        detection_conf = float(np.clip(face_detection.confidence, 0.0, 1.0))

        # Occlusion is inversely related to quality and confidence
        occlusion = 1.0 - (face_quality * 0.6 + detection_conf * 0.4)
        return float(np.clip(occlusion, 0.0, 1.0))

    def _assess_motion_blur(self, frame: np.ndarray) -> float:
        """
        Estimate motion blur using Laplacian variance.

        Returns a value in [0, 1] where 1 = maximum blur.
        """
        if frame is None or frame.size == 0:
            return 1.0

        try:
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            # Fallback: use numpy gradient
            gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
            gy, gx = np.gradient(gray)
            laplacian_var = float(np.var(np.sqrt(gx**2 + gy**2)))

        # High variance = sharp image = low blur
        # Normalize: variance > threshold → not blurry
        blur_score = max(0.0, 1.0 - laplacian_var / self._HIGH_BLUR_THRESHOLD)
        return float(np.clip(blur_score, 0.0, 1.0))

    def _assess_eye_visibility(self, face_detection: FaceDetection) -> float:
        """
        Estimate eye visibility from landmark positions.

        Checks whether eye landmarks are within the frame bounds.
        """
        landmarks = face_detection.landmarks
        if landmarks is None or len(landmarks) == 0:
            return 0.0

        # Eye landmark indices (MediaPipe Face Mesh)
        left_eye_indices = [33, 133, 159, 145]
        right_eye_indices = [362, 263, 386, 374]
        eye_indices = left_eye_indices + right_eye_indices

        visible_count = 0
        total_count = 0

        for idx in eye_indices:
            if idx < len(landmarks):
                total_count += 1
                x, y = landmarks[idx]
                if x > 0 and y > 0:
                    visible_count += 1

        if total_count == 0:
            return 0.5  # Unknown

        return float(visible_count / total_count)

    def _assess_landmark_quality(self, face_detection: FaceDetection) -> float:
        """
        Assess overall landmark quality from detection confidence and quality score.
        """
        return float(np.clip(
            face_detection.confidence * 0.5 + face_detection.quality_score * 0.5,
            0.0, 1.0,
        ))

    def _assess_temporal_consistency(self, gaze_estimate: GazeEstimate) -> float:
        """
        Assess temporal consistency by comparing to recent estimates.

        Large sudden jumps in gaze direction indicate low consistency.
        """
        if len(self._recent_estimates) < 2:
            return 1.0  # No history to compare against

        recent = self._recent_estimates[-1]
        prev_vec = recent.gaze_vector_3d
        curr_vec = gaze_estimate.gaze_vector_3d

        prev_norm = np.linalg.norm(prev_vec)
        curr_norm = np.linalg.norm(curr_vec)

        if prev_norm == 0 or curr_norm == 0:
            return 0.5

        # Cosine similarity between consecutive gaze vectors
        cos_sim = float(np.dot(prev_vec / prev_norm, curr_vec / curr_norm))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))

        # Map cosine similarity to quality: 1.0 = same direction, 0.0 = opposite
        consistency = (cos_sim + 1.0) / 2.0
        return float(np.clip(consistency, 0.0, 1.0))

    def _compute_overall_quality(
        self,
        head_pose_quality: float,
        lighting_quality: float,
        occlusion_level: float,
        motion_blur: float,
        eye_visibility: float,
        landmark_quality: float,
        temporal_consistency: float,
    ) -> float:
        """
        Compute weighted overall quality score.

        Occlusion and motion blur are penalty factors (higher = worse quality).
        """
        occlusion_quality = 1.0 - occlusion_level
        sharpness_quality = 1.0 - motion_blur

        overall = (
            head_pose_quality    * 0.20 +
            lighting_quality     * 0.15 +
            occlusion_quality    * 0.20 +
            sharpness_quality    * 0.10 +
            eye_visibility       * 0.20 +
            landmark_quality     * 0.10 +
            temporal_consistency * 0.05
        )
        return float(np.clip(overall, 0.0, 1.0))
