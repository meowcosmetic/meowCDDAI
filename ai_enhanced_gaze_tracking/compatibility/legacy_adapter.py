"""
Legacy response adapter.

Translates enhanced GazeEstimate / session data into the legacy
GazeAnalysisResponse format so existing API consumers keep working.

Requirements: 10.1, 10.3
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.data_models import (
    AttentionType,
    FocusEvent,
    GazeEstimate,
    QualityMetrics,
    SessionSummary,
    WanderingPeriod,
)


# ---------------------------------------------------------------------------
# Canonical legacy field names (mirrors text_embeding/gaze/models.py)
# ---------------------------------------------------------------------------
LEGACY_FIELDS = frozenset({
    "eye_contact_percentage",
    "gaze_direction_stats",
    "total_frames",
    "analyzed_duration",
    "focusing_duration",
    "attention_to_person_percentage",
    "attention_to_objects_percentage",
    "attention_to_book_percentage",
    "book_focusing_score",
    "detected_objects",
    "detected_books",
    "object_interaction_events",
    "risk_score",
    "focus_timeline",
    "object_focus_stats",
    "pattern_analysis",
    "gaze_wandering_score",
    "gaze_wandering_percentage",
    "wandering_periods",
    "fatigue_score",
    "fatigue_level",
    "fatigue_indicators",
    "focus_level",
    "focus_level_details",
    "object_detection_model",
    "object_detection_available",
})

# Enhanced-only fields added on top of the legacy schema
ENHANCED_FIELDS = frozenset({
    "enhanced_version",
    "overall_quality_score",
    "head_pose_compensation_active",
    "camera_angle_correction_active",
    "ai_model_active",
    "sensor_fusion_active",
    "confidence_scores",
    "quality_metrics",
})


class LegacyResponseAdapter:
    """
    Converts enhanced system outputs to the legacy GazeAnalysisResponse dict.

    The returned dict always contains every legacy field (Requirement 10.3)
    plus additional enhanced fields (Requirement 10.1).
    """

    def __init__(
        self,
        object_detection_model: Optional[str] = None,
        object_detection_available: bool = False,
    ) -> None:
        self.object_detection_model = object_detection_model
        self.object_detection_available = object_detection_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_session_summary(self, summary: SessionSummary) -> Dict[str, Any]:
        """Build a legacy-compatible response from a SessionSummary."""
        total_frames = summary.total_frames
        valid_frames = summary.valid_frames
        duration = summary.end_time - summary.start_time

        # Attention percentages
        attention_stats = summary.attention_statistics or {}
        eye_contact_pct = attention_stats.get("eye_contact_percentage", 0.0)
        person_pct = attention_stats.get("attention_to_person_percentage", 0.0)
        objects_pct = attention_stats.get("attention_to_objects_percentage", 0.0)
        book_pct = attention_stats.get("attention_to_book_percentage", 0.0)

        # Focus events → legacy timeline
        focus_timeline = self._build_focus_timeline(summary.focus_events)
        object_focus_stats = self._build_object_focus_stats(summary.focus_events)
        focusing_duration = sum(e.duration for e in summary.focus_events)

        # Wandering
        wandering_periods = self._build_wandering_periods(summary.wandering_periods)
        wandering_pct = (
            sum(w.duration for w in summary.wandering_periods) / duration * 100
            if duration > 0
            else 0.0
        )
        wandering_score = min(100.0, wandering_pct)

        # Quality / fatigue (from processing_metrics if available)
        overall_quality = summary.overall_quality
        fatigue = summary.fatigue_assessment or {}

        response = self._build_base_response(
            eye_contact_percentage=eye_contact_pct,
            total_frames=total_frames,
            analyzed_duration=duration,
            focusing_duration=focusing_duration,
            attention_to_person_percentage=person_pct,
            attention_to_objects_percentage=objects_pct,
            attention_to_book_percentage=book_pct,
            book_focusing_score=book_pct,
            focus_timeline=focus_timeline,
            object_focus_stats=object_focus_stats,
            gaze_wandering_score=wandering_score,
            gaze_wandering_percentage=wandering_pct,
            wandering_periods=wandering_periods,
            fatigue_score=fatigue.get("fatigue_score", 0.0),
            fatigue_level=fatigue.get("fatigue_level", "low"),
            fatigue_indicators=fatigue.get("fatigue_indicators", {}),
            risk_score=self._calculate_risk_score(
                wandering_score, fatigue.get("fatigue_score", 0.0), overall_quality
            ),
        )

        # Attach enhanced fields
        response.update(
            self._build_enhanced_fields(
                overall_quality=overall_quality,
                session_summary=summary,
            )
        )

        return response

    def from_gaze_estimates(
        self,
        estimates: List[GazeEstimate],
        duration: float,
        detected_objects: Optional[List[Dict[str, Any]]] = None,
        detected_books: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a legacy-compatible response from a list of GazeEstimates."""
        total_frames = len(estimates)
        if total_frames == 0:
            return self._empty_response()

        # Gaze direction stats
        gaze_direction_stats = self._compute_gaze_direction_stats(estimates)

        # Confidence / quality averages
        avg_confidence = sum(e.confidence for e in estimates) / total_frames
        avg_quality = (
            sum(e.quality_metrics.overall_quality for e in estimates) / total_frames
        )

        response = self._build_base_response(
            eye_contact_percentage=avg_confidence * 100,
            total_frames=total_frames,
            analyzed_duration=duration,
            focusing_duration=0.0,
            attention_to_person_percentage=0.0,
            attention_to_objects_percentage=0.0,
            attention_to_book_percentage=0.0,
            book_focusing_score=0.0,
            gaze_direction_stats=gaze_direction_stats,
            detected_objects=detected_objects or [],
            detected_books=detected_books or [],
        )

        response.update(
            self._build_enhanced_fields(
                overall_quality=avg_quality,
                confidence_scores=[e.confidence for e in estimates],
            )
        )

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_base_response(self, **kwargs) -> Dict[str, Any]:
        """Return a dict with all legacy fields populated."""
        defaults: Dict[str, Any] = {
            "eye_contact_percentage": 0.0,
            "gaze_direction_stats": {"left": 0.0, "right": 0.0, "center": 0.0, "up": 0.0, "down": 0.0},
            "total_frames": 0,
            "analyzed_duration": 0.0,
            "focusing_duration": 0.0,
            "attention_to_person_percentage": 0.0,
            "attention_to_objects_percentage": 0.0,
            "attention_to_book_percentage": 0.0,
            "book_focusing_score": 0.0,
            "detected_objects": [],
            "detected_books": [],
            "object_interaction_events": [],
            "risk_score": 0.0,
            "focus_timeline": [],
            "object_focus_stats": {},
            "pattern_analysis": {},
            "gaze_wandering_score": 0.0,
            "gaze_wandering_percentage": 0.0,
            "wandering_periods": [],
            "fatigue_score": 0.0,
            "fatigue_level": "low",
            "fatigue_indicators": {},
            "focus_level": 0.0,
            "focus_level_details": {},
            "object_detection_model": self.object_detection_model,
            "object_detection_available": self.object_detection_available,
        }
        defaults.update(kwargs)
        return defaults

    def _build_enhanced_fields(
        self,
        overall_quality: float = 0.0,
        session_summary: Optional[SessionSummary] = None,
        confidence_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Return the additional enhanced fields."""
        return {
            "enhanced_version": "2.0.0",
            "overall_quality_score": overall_quality,
            "head_pose_compensation_active": True,
            "camera_angle_correction_active": True,
            "ai_model_active": True,
            "sensor_fusion_active": True,
            "confidence_scores": confidence_scores or [],
            "quality_metrics": {
                "overall_quality": overall_quality,
            },
        }

    def _build_focus_timeline(self, focus_events: List[FocusEvent]) -> List[Dict[str, Any]]:
        return [
            {
                "start_time": e.start_time,
                "duration": e.duration,
                "target_object_id": e.target_object_id,
                "focus_type": e.focus_type.value if isinstance(e.focus_type, AttentionType) else str(e.focus_type),
                "stability_score": e.stability_score,
                "confidence": e.confidence,
            }
            for e in focus_events
        ]

    def _build_object_focus_stats(self, focus_events: List[FocusEvent]) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        for event in focus_events:
            obj_id = event.target_object_id or "unknown"
            if obj_id not in stats:
                stats[obj_id] = {"total_duration": 0.0, "event_count": 0, "avg_stability": 0.0}
            stats[obj_id]["total_duration"] += event.duration
            stats[obj_id]["event_count"] += 1
            stats[obj_id]["avg_stability"] = (
                stats[obj_id]["avg_stability"] * (stats[obj_id]["event_count"] - 1)
                + event.stability_score
            ) / stats[obj_id]["event_count"]
        return stats

    def _build_wandering_periods(self, periods: List[WanderingPeriod]) -> List[Dict[str, Any]]:
        return [
            {
                "start_time": p.start_time,
                "end_time": p.end_time,
                "duration": p.duration,
                "stability_score": p.stability_score,
                "average_position": list(p.average_position),
                "variance": p.variance,
            }
            for p in periods
        ]

    def _compute_gaze_direction_stats(self, estimates: List[GazeEstimate]) -> Dict[str, float]:
        """Derive rough gaze direction distribution from 3D gaze vectors."""
        counts: Dict[str, int] = {"left": 0, "right": 0, "center": 0, "up": 0, "down": 0}
        for est in estimates:
            vec = est.gaze_vector_3d
            if vec is None or len(vec) < 3:
                counts["center"] += 1
                continue
            x, y = float(vec[0]), float(vec[1])
            threshold = 0.12
            if abs(x) < threshold and abs(y) < threshold:
                counts["center"] += 1
            elif abs(x) >= abs(y):
                counts["left" if x < 0 else "right"] += 1
            else:
                counts["up" if y < 0 else "down"] += 1

        total = max(1, len(estimates))
        return {k: round(v / total * 100, 2) for k, v in counts.items()}

    def _calculate_risk_score(
        self, wandering_score: float, fatigue_score: float, quality: float
    ) -> float:
        """Heuristic risk score combining wandering, fatigue, and quality."""
        quality_penalty = (1.0 - quality) * 20.0
        return min(100.0, wandering_score * 0.4 + fatigue_score * 0.4 + quality_penalty * 0.2)

    def _empty_response(self) -> Dict[str, Any]:
        response = self._build_base_response()
        response.update(self._build_enhanced_fields())
        return response


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def build_legacy_response(
    estimates: List[GazeEstimate],
    duration: float,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    detected_books: Optional[List[Dict[str, Any]]] = None,
    object_detection_model: Optional[str] = None,
    object_detection_available: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper: build a legacy-compatible response dict from a list
    of GazeEstimates.

    Requirements: 10.1, 10.3
    """
    adapter = LegacyResponseAdapter(
        object_detection_model=object_detection_model,
        object_detection_available=object_detection_available,
    )
    return adapter.from_gaze_estimates(
        estimates=estimates,
        duration=duration,
        detected_objects=detected_objects,
        detected_books=detected_books,
    )
