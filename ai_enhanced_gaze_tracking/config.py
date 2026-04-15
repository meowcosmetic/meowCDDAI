"""
Configuration management for the AI-Enhanced Gaze Tracking System.

This module provides centralized configuration management with support for
environment variables, configuration files, and runtime parameter updates.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml


@dataclass
class EnhancedGazeConfig:
    """Enhanced configuration for the AI-Enhanced Gaze Tracking System."""
    
    # ========================================================================
    # FACE DETECTION CONFIGURATION
    # ========================================================================
    face_detection_confidence: float = 0.5
    face_detection_model: str = "mediapipe"  # "mediapipe", "opencv", "custom"
    face_quality_threshold: float = 0.6
    child_face_size_threshold: float = 0.10
    adult_face_size_threshold: float = 0.15
    
    # ========================================================================
    # HEAD POSE ESTIMATION CONFIGURATION  
    # ========================================================================
    head_pose_compensation: bool = True
    head_pose_model: str = "pnp_solver"  # "pnp_solver", "ai_model"
    head_pose_smoothing: bool = True
    head_pose_smoothing_factor: float = 0.7
    max_head_rotation_degrees: float = 45.0
    
    # ========================================================================
    # GAZE ESTIMATION CONFIGURATION
    # ========================================================================
    gaze_estimation_method: str = "multi_modal"  # "2d", "3d", "ai", "multi_modal"
    ai_gaze_model_path: Optional[str] = None
    ai_gaze_confidence_threshold: float = 0.5
    gaze_ensemble_models: list = field(default_factory=list)
    gaze_smoothing: bool = True
    gaze_smoothing_window: int = 5
    
    # ========================================================================
    # CAMERA CALIBRATION CONFIGURATION
    # ========================================================================
    camera_angle_correction: bool = True
    auto_calibration: bool = True
    calibration_reference_points: int = 9
    max_camera_angle_degrees: float = 30.0
    recalibration_interval: float = 300.0  # seconds
    
    # ========================================================================
    # SENSOR FUSION CONFIGURATION
    # ========================================================================
    sensor_fusion_method: str = "kalman"  # "weighted", "kalman", "particle"
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "2d_landmarks": 0.3,
        "3d_pose": 0.4, 
        "ai_model": 0.3
    })
    adaptive_weights: bool = True
    temporal_consistency_weight: float = 0.2
    
    # ========================================================================
    # FOCUS DETECTION CONFIGURATION
    # ========================================================================
    focus_detection_enabled: bool = True
    min_focus_duration: float = 1.0  # seconds
    focus_stability_threshold: float = 0.02
    focus_stability_window_ms: float = 200.0
    require_object_focus: bool = True
    wandering_detection_enabled: bool = True
    
    # ========================================================================
    # OBJECT DETECTION CONFIGURATION
    # ========================================================================
    object_detection_model: str = "yolov8"  # "yolov8", "custom"
    object_detection_confidence: float = 0.5
    object_detection_interval: int = 5  # frames
    custom_yolo_weights: str = ""
    oid_model_size: str = "l"  # "n", "s", "m", "l", "x"
    
    # ========================================================================
    # QUALITY ASSESSMENT CONFIGURATION
    # ========================================================================
    quality_assessment: bool = True
    min_quality_threshold: float = 0.5
    lighting_quality_weight: float = 0.3
    occlusion_quality_weight: float = 0.3
    stability_quality_weight: float = 0.4
    
    # ========================================================================
    # PERFORMANCE CONFIGURATION
    # ========================================================================
    real_time_processing: bool = True
    target_fps: int = 30
    max_processing_latency_ms: float = 33.0  # ~30 FPS
    gpu_acceleration: bool = True
    memory_limit_mb: int = 2048
    
    # ========================================================================
    # LOGGING AND MONITORING CONFIGURATION
    # ========================================================================
    log_level: str = "INFO"
    log_file: Optional[str] = None
    structured_logging: bool = True
    performance_monitoring: bool = True
    metrics_collection: bool = True
    
    # ========================================================================
    # BACKWARD COMPATIBILITY CONFIGURATION
    # ========================================================================
    legacy_api_support: bool = True
    legacy_output_format: bool = True
    migration_mode: bool = False
    
    @classmethod
    def from_file(cls, config_path: str) -> 'EnhancedGazeConfig':
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Configuration instance
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'EnhancedGazeConfig':
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with 'GAZE_'
        
        Returns:
            Configuration instance
        """
        config_data = {}
        
        # Map environment variables to config fields
        env_mapping = {
            'GAZE_FACE_DETECTION_CONFIDENCE': 'face_detection_confidence',
            'GAZE_HEAD_POSE_COMPENSATION': 'head_pose_compensation',
            'GAZE_CAMERA_ANGLE_CORRECTION': 'camera_angle_correction',
            'GAZE_AI_MODEL_PATH': 'ai_gaze_model_path',
            'GAZE_REAL_TIME_PROCESSING': 'real_time_processing',
            'GAZE_LOG_LEVEL': 'log_level',
            'GAZE_LOG_FILE': 'log_file',
            'GAZE_GPU_ACCELERATION': 'gpu_acceleration'
        }
        
        for env_var, config_field in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion based on field type
                if config_field in ['face_detection_confidence', 'head_pose_smoothing_factor']:
                    config_data[config_field] = float(value)
                elif config_field in ['head_pose_compensation', 'camera_angle_correction', 
                                    'real_time_processing', 'gpu_acceleration']:
                    config_data[config_field] = value.lower() in ['true', '1', 'yes']
                else:
                    config_data[config_field] = value
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Validate ranges
        if not 0.0 <= self.face_detection_confidence <= 1.0:
            raise ValueError("face_detection_confidence must be between 0.0 and 1.0")
        
        if not 0.0 <= self.ai_gaze_confidence_threshold <= 1.0:
            raise ValueError("ai_gaze_confidence_threshold must be between 0.0 and 1.0")
        
        if self.min_focus_duration <= 0:
            raise ValueError("min_focus_duration must be positive")
        
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        
        if self.max_processing_latency_ms <= 0:
            raise ValueError("max_processing_latency_ms must be positive")
        
        # Validate model paths
        if self.ai_gaze_model_path and not Path(self.ai_gaze_model_path).exists():
            raise FileNotFoundError(f"AI gaze model not found: {self.ai_gaze_model_path}")
        
        if self.custom_yolo_weights and not Path(self.custom_yolo_weights).exists():
            raise FileNotFoundError(f"Custom YOLO weights not found: {self.custom_yolo_weights}")


# Global configuration instance
_config: Optional[EnhancedGazeConfig] = None


def get_config() -> EnhancedGazeConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = EnhancedGazeConfig()
    return _config


def set_config(config: EnhancedGazeConfig) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    _config = config


def load_config_from_file(config_path: str) -> None:
    """Load configuration from file and set as global."""
    config = EnhancedGazeConfig.from_file(config_path)
    set_config(config)


def load_config_from_env() -> None:
    """Load configuration from environment variables and set as global."""
    config = EnhancedGazeConfig.from_env()
    set_config(config)