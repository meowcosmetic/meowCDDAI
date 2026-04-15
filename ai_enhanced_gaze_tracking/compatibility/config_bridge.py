"""
Configuration bridge between legacy and enhanced config parameters.

Allows the enhanced system to accept both legacy parameter names
(from the old gaze tracking config) and new enhanced parameter names.

Requirements: 10.2
"""

from __future__ import annotations

from typing import Any, Dict

from ..config import EnhancedGazeConfig


# Mapping: legacy_key -> enhanced_key
LEGACY_TO_ENHANCED: Dict[str, str] = {
    # Face detection
    "FACE_DETECTION_CONFIDENCE": "face_detection_confidence",
    "face_detection_confidence_threshold": "face_detection_confidence",
    # Head pose
    "HEAD_POSE_ENABLED": "head_pose_compensation",
    "head_pose_enabled": "head_pose_compensation",
    # Camera
    "CAMERA_ANGLE_CORRECTION": "camera_angle_correction",
    "camera_correction_enabled": "camera_angle_correction",
    # Gaze
    "GAZE_METHOD": "gaze_estimation_method",
    "gaze_method": "gaze_estimation_method",
    "AI_MODEL_PATH": "ai_gaze_model_path",
    "ai_model_path": "ai_gaze_model_path",
    # Focus
    "MIN_FOCUS_DURATION": "min_focus_duration",
    "GAZE_STABILITY_THRESHOLD": "focus_stability_threshold",
    "MIN_FOCUSING_DURATION": "min_focus_duration",
    # Object detection
    "OBJECT_DETECTION_MODEL": "object_detection_model",
    "OBJECT_DETECTION_CONFIDENCE": "object_detection_confidence",
    "OBJECT_DETECTION_INTERVAL": "object_detection_interval",
    "CUSTOM_YOLO_WEIGHTS": "custom_yolo_weights",
    # Performance
    "TARGET_FPS": "target_fps",
    "USE_GPU": "gpu_acceleration",
    "MEMORY_LIMIT_MB": "memory_limit_mb",
    # Logging
    "LOG_LEVEL": "log_level",
}


class ConfigBridge:
    """
    Translates legacy configuration dicts into EnhancedGazeConfig instances.

    Supports both legacy parameter names (upper-case constants from the old
    GazeConfig class) and the new lower-case field names used by
    EnhancedGazeConfig.

    Requirements: 10.2
    """

    @staticmethod
    def from_legacy_dict(legacy_params: Dict[str, Any]) -> EnhancedGazeConfig:
        """
        Build an EnhancedGazeConfig from a dict that may contain legacy keys.

        Unknown keys are silently ignored so that old callers don't break.
        """
        translated: Dict[str, Any] = {}

        for key, value in legacy_params.items():
            # Try direct mapping first
            enhanced_key = LEGACY_TO_ENHANCED.get(key)
            if enhanced_key is None:
                # Try the key as-is (already an enhanced key)
                enhanced_key = key

            # Only apply if the target field actually exists on the config
            if hasattr(EnhancedGazeConfig, enhanced_key) or enhanced_key in EnhancedGazeConfig.__dataclass_fields__:
                translated[enhanced_key] = _coerce_value(enhanced_key, value)

        return EnhancedGazeConfig(**translated)

    @staticmethod
    def to_legacy_dict(config: EnhancedGazeConfig) -> Dict[str, Any]:
        """
        Export an EnhancedGazeConfig as a dict using legacy key names where
        a mapping exists, falling back to the enhanced key name.
        """
        # Reverse mapping: enhanced_key -> legacy_key (first match wins)
        enhanced_to_legacy: Dict[str, str] = {}
        for legacy_key, enhanced_key in LEGACY_TO_ENHANCED.items():
            if enhanced_key not in enhanced_to_legacy:
                enhanced_to_legacy[enhanced_key] = legacy_key

        result: Dict[str, Any] = {}
        for field_name, value in config.to_dict().items():
            legacy_key = enhanced_to_legacy.get(field_name, field_name)
            result[legacy_key] = value
        return result

    @staticmethod
    def merge_configs(
        legacy_params: Dict[str, Any],
        enhanced_params: Dict[str, Any],
    ) -> EnhancedGazeConfig:
        """
        Merge legacy and enhanced parameter dicts.

        Enhanced params take precedence over legacy params when both specify
        the same underlying setting.
        """
        # Start from legacy
        base = ConfigBridge.from_legacy_dict(legacy_params)
        base_dict = base.to_dict()

        # Apply enhanced overrides
        for key, value in enhanced_params.items():
            if key in base_dict:
                base_dict[key] = value

        return EnhancedGazeConfig(**base_dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_value(field_name: str, value: Any) -> Any:
    """Coerce a value to the expected type for the given config field."""
    # Boolean fields that may arrive as strings
    bool_fields = {
        "head_pose_compensation", "camera_angle_correction", "gpu_acceleration",
        "real_time_processing", "auto_calibration", "gaze_smoothing",
        "head_pose_smoothing", "adaptive_weights", "focus_detection_enabled",
        "wandering_detection_enabled", "quality_assessment", "legacy_api_support",
        "legacy_output_format", "migration_mode", "structured_logging",
        "performance_monitoring", "metrics_collection",
    }
    float_fields = {
        "face_detection_confidence", "face_quality_threshold",
        "ai_gaze_confidence_threshold", "min_focus_duration",
        "focus_stability_threshold", "max_camera_angle_degrees",
        "recalibration_interval", "temporal_consistency_weight",
        "min_quality_threshold",
    }
    int_fields = {
        "target_fps", "memory_limit_mb", "object_detection_interval",
        "gaze_smoothing_window", "calibration_reference_points",
    }

    if field_name in bool_fields:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    if field_name in float_fields:
        return float(value)

    if field_name in int_fields:
        return int(value)

    return value
