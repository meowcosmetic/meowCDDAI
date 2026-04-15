# AI-Enhanced Gaze Tracking System — API Reference

## Overview

The AI-Enhanced Gaze Tracking System exposes a Python API through the `ai_enhanced_gaze_tracking` package. All public interfaces are defined in `ai_enhanced_gaze_tracking/core/interfaces.py` and the data models in `ai_enhanced_gaze_tracking/core/data_models.py`.

---

## Data Models

### `GazeEstimate`

Represents a single gaze estimate with full 3D information and quality metadata.

```python
@dataclass
class GazeEstimate:
    gaze_vector_3d: np.ndarray          # 3D gaze direction in world coordinates
    gaze_point_2d: Tuple[float, float]  # 2D gaze point on screen (normalized 0–1)
    confidence: float                   # Overall confidence score (0–1)
    head_pose: HeadPose                 # Associated head pose
    timestamp: float                    # Frame timestamp (seconds)
    source_confidences: Dict[str, float]  # Per-source confidence scores
    quality_metrics: QualityMetrics     # Data quality assessment
```

### `HeadPose`

3D head orientation and position.

```python
@dataclass
class HeadPose:
    yaw: float              # Left/right rotation (radians)
    pitch: float            # Up/down rotation (radians)
    roll: float             # Tilt rotation (radians)
    translation: np.ndarray # 3D position vector
    rotation_matrix: np.ndarray  # 3×3 rotation matrix
    confidence: float       # Pose estimation confidence (0–1)
```

### `CameraParameters`

Camera calibration and angle correction data.

```python
@dataclass
class CameraParameters:
    intrinsic_matrix: np.ndarray   # 3×3 camera intrinsic matrix
    distortion_coeffs: np.ndarray  # Distortion coefficients
    camera_angle: float            # Camera angle from straight-on (degrees)
    correction_matrix: np.ndarray  # Angle correction transformation matrix
    reference_frame: np.ndarray    # Normalized coordinate system
```

### `FocusEvent`

A detected focus event with target and duration information.

```python
@dataclass
class FocusEvent:
    target_object_id: Optional[str]  # ID of focused object (None if wandering)
    focus_type: str                  # "object" | "person" | "wandering" | "unknown"
    start_time: float                # Focus start timestamp (seconds)
    duration: float                  # Focus duration (seconds)
    stability_score: float           # Gaze stability during focus (0–1)
    confidence: float                # Focus detection confidence (0–1)
```

### `QualityMetrics`

Per-frame data quality assessment.

```python
@dataclass
class QualityMetrics:
    overall_quality: float       # Weighted overall quality (0–1)
    head_pose_quality: float     # Head pose estimation quality (0–1)
    lighting_quality: float      # Lighting condition score (0–1)
    occlusion_level: float       # Face occlusion level (0=none, 1=full)
    motion_blur: float           # Motion blur level (0=sharp, 1=blurry)
    tracking_stability: float    # Temporal tracking stability (0–1)
    eye_visibility: float        # Eye landmark visibility (0–1)
    landmark_quality: float      # Overall landmark quality (0–1)
    temporal_consistency: float  # Consistency with previous frames (0–1)
```

### `FaceDetection`

Result from the face detection component.

```python
@dataclass
class FaceDetection:
    bbox: Tuple[int, int, int, int]  # Bounding box (x, y, w, h)
    landmarks: np.ndarray            # 2D landmark coordinates
    confidence: float                # Detection confidence (0–1)
    quality_score: float             # Face quality score (0–1)
    face_id: Optional[str]           # Tracking ID (if available)
```

---

## Core Interfaces

### `FaceDetector`

```python
class FaceDetector(ABC):
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a video frame."""
```

**Implementations:**
- `HybridFaceDetector` — MediaPipe with OpenCV fallback (recommended)
- `MediaPipeFaceDetector` — MediaPipe Face Mesh only
- `CustomFaceDetector` — OpenCV-based fallback detector

### `HeadPoseEstimator`

```python
class HeadPoseEstimator(ABC):
    def estimate_head_pose(
        self,
        landmarks: np.ndarray,
        camera_matrix: np.ndarray
    ) -> HeadPose:
        """Estimate 3D head pose from 2D landmarks."""
```

**Implementation:** `PnPHeadPoseEstimator`

### `GazeEstimator`

```python
class GazeEstimator(ABC):
    def estimate_gaze(
        self,
        frame: np.ndarray,
        face_detection: FaceDetection,
        head_pose: HeadPose
    ) -> GazeEstimate:
        """Estimate gaze direction from frame and face data."""
```

**Implementations:**
- `CompensatedGazeEstimator` — 3D head-pose-compensated estimation
- `AIGazeEstimator` — CNN-based AI gaze prediction

### `CameraCalibrator`

```python
class CameraCalibrator(ABC):
    def calibrate_camera(
        self,
        reference_points: List[Tuple[float, float]]
    ) -> CameraParameters:
        """Calibrate camera using reference gaze points."""
```

**Implementation:** `CameraCalibrationSystem`

### `SensorFusion`

```python
class SensorFusion(ABC):
    def fuse_estimates(
        self,
        estimates: List[GazeEstimate],
        confidences: List[float]
    ) -> GazeEstimate:
        """Fuse multiple gaze estimates into a single result."""
```

**Implementation:** `MultiModalFusion`

### `FocusDetector`

```python
class FocusDetector(ABC):
    def detect_focus(
        self,
        gaze_vector: np.ndarray,
        objects: List[Dict]
    ) -> Optional[FocusEvent]:
        """Detect focus events from gaze and scene objects."""
```

**Implementation:** `FocusDetectionSystem`

### `QualityAssessor`

```python
class QualityAssessor(ABC):
    def assess_quality(
        self,
        gaze_estimate: GazeEstimate,
        frame: np.ndarray,
        face_detection: FaceDetection
    ) -> QualityMetrics:
        """Assess quality of a gaze estimate."""

    def flag_unreliable_data(
        self,
        quality_metrics: List[QualityMetrics]
    ) -> List[int]:
        """Return indices of unreliable data segments."""
```

**Implementation:** `GazeQualityAssessor`

---

## Error Handling API

### `ErrorHandler`

Central error and degradation manager. Implements Requirement 8.5.

```python
handler = ErrorHandler(recovery_cooldown_s=10.0, max_failures=5)

# Report a component failure
action = handler.report_failure(
    component="face_detection",
    error=exception,
    severity=ErrorSeverity.MEDIUM
)

# Get diagnostic information
diagnostics = handler.get_diagnostics()
# Returns: system_state, component_health, recent_errors, active_fallbacks

# Get user-facing guidance
guidance = handler.get_user_guidance(component="camera")

# Handle resource exhaustion
state = handler.handle_resource_exhaustion(memory_mb=2600.0, cpu_percent=90.0)
```

**`ErrorSeverity` values:** `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`

**`SystemState` values:** `NORMAL`, `DEGRADED`, `MINIMAL`, `RECOVERING`, `FAILED`

**`RecoveryAction` values:** `RETRY`, `USE_FALLBACK`, `REDUCE_COMPLEXITY`, `SKIP_FRAME`, `RESTART_COMPONENT`, `NOTIFY_USER`, `GRACEFUL_SHUTDOWN`

---

## Quality Assessment API

### `GazeQualityAssessor`

```python
assessor = GazeQualityAssessor(
    quality_threshold=0.4,
    alert_threshold=0.3,
    temporal_window=10
)

# Assess a single frame
metrics = assessor.assess_quality(gaze_estimate, frame, face_detection)

# Check for user-facing alerts (Req 9.3)
alerts = assessor.check_for_alerts(metrics)
for alert in alerts:
    print(f"[{alert.severity}] {alert.message}")
    print(f"  Action: {alert.suggested_action}")

# Validate a sequence of estimates
reliable_flags = assessor.validate_data_segment(gaze_sequence)

# Get indices of unreliable segments
bad_indices = assessor.flag_unreliable_data(quality_metrics_list)

# Session summary
summary = assessor.get_session_quality_summary(quality_metrics_list)
# Returns: mean_overall_quality, reliability_percentage, total_frames, etc.
```

---

## Configuration API

### `EnhancedGazeConfig`

```python
from ai_enhanced_gaze_tracking.config import EnhancedGazeConfig, get_config, set_config

# Default configuration
config = EnhancedGazeConfig()

# Load from file
config = EnhancedGazeConfig.from_file("config.yaml")

# Load from environment variables (GAZE_* prefix)
config = EnhancedGazeConfig.from_env()

# Validate and set as global
set_config(config)

# Access global config
config = get_config()

# Save to file
config.save_to_file("config.yaml")
```

See `docs/configuration_guide.md` for all available parameters.

---

## Dependency Injection

```python
from ai_enhanced_gaze_tracking.core.dependency_injection import DIContainer

container = DIContainer()
container.register(FaceDetector, HybridFaceDetector)
container.register(GazeEstimator, CompensatedGazeEstimator)

face_detector = container.resolve(FaceDetector)
```

---

## Backward Compatibility

The system maintains full backward compatibility with the legacy API. Existing calls continue to work unchanged. Enhanced fields are added to responses without breaking existing consumers.

See `ai_enhanced_gaze_tracking/compatibility/legacy_adapter.py` for the adapter implementation.
