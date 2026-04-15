# AI-Enhanced Gaze Tracking System — Setup & Configuration Guide

## Requirements

- Python 3.10 or 3.12
- pip or conda
- (Optional) NVIDIA GPU with CUDA 11.8+ for GPU acceleration

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <project-root>
```

### 2. Create a virtual environment

```bash
python -m venv venv312
# Windows
venv312\Scripts\activate
# Linux / macOS
source venv312/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA-enabled PyTorch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Download model weights

```bash
# YOLOv8 object detection weights
python download_yolo_weights.py

# (Optional) Open Images Dataset detector
python download_oid_model.py
```

---

## Configuration

The system is configured via `EnhancedGazeConfig`. You can supply configuration through:

1. A YAML or JSON file
2. Environment variables (`GAZE_*` prefix)
3. Programmatically in Python

### Configuration file (recommended)

Copy the template and edit as needed:

```bash
cp config_template.yaml config.yaml
```

Then load it:

```python
from ai_enhanced_gaze_tracking.config import load_config_from_file
load_config_from_file("config.yaml")
```

### Key configuration parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `face_detection_model` | `"mediapipe"` | Face detector: `"mediapipe"`, `"opencv"`, `"custom"` |
| `head_pose_compensation` | `True` | Enable 3D head pose compensation (Req 1) |
| `camera_angle_correction` | `True` | Enable camera angle bias correction (Req 2) |
| `gaze_estimation_method` | `"multi_modal"` | `"2d"`, `"3d"`, `"ai"`, `"multi_modal"` |
| `ai_gaze_model_path` | `None` | Path to AI gaze model weights |
| `min_focus_duration` | `1.0` | Minimum seconds for a valid focus event (Req 6.1) |
| `target_fps` | `30` | Target processing frame rate |
| `gpu_acceleration` | `True` | Use GPU for AI inference |
| `log_level` | `"INFO"` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `legacy_api_support` | `True` | Maintain backward-compatible API responses |

### Environment variables

| Variable | Config field |
|----------|-------------|
| `GAZE_FACE_DETECTION_CONFIDENCE` | `face_detection_confidence` |
| `GAZE_HEAD_POSE_COMPENSATION` | `head_pose_compensation` |
| `GAZE_CAMERA_ANGLE_CORRECTION` | `camera_angle_correction` |
| `GAZE_AI_MODEL_PATH` | `ai_gaze_model_path` |
| `GAZE_REAL_TIME_PROCESSING` | `real_time_processing` |
| `GAZE_LOG_LEVEL` | `log_level` |
| `GAZE_LOG_FILE` | `log_file` |
| `GAZE_GPU_ACCELERATION` | `gpu_acceleration` |

---

## Camera Setup

For best results:

- Position the camera at eye level, facing the subject directly.
- The system corrects for camera angles up to ±30° from straight-on (Req 2.1).
- Ensure the subject's face is well-lit with diffuse, even lighting.
- Avoid strong backlighting or direct light sources in the camera's field of view.
- Minimum recommended face size: 10% of frame width for children.

### Calibration

Automatic calibration runs at session start. For improved accuracy, use reference-point calibration:

```python
from ai_enhanced_gaze_tracking.components.calibration.personal_calibration import PersonalCalibrationSystem

calibrator = PersonalCalibrationSystem(config)
# Ask the subject to look at 9 known screen positions
calibrator.add_reference_observation(screen_point=(0.5, 0.5), gaze_estimate=estimate)
calibrator.finalize_calibration()
```

---

## Running the System

### As part of the FastAPI server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Programmatic usage

```python
from ai_enhanced_gaze_tracking import DIContainer
from ai_enhanced_gaze_tracking.config import EnhancedGazeConfig, set_config

config = EnhancedGazeConfig.from_file("config.yaml")
set_config(config)

container = DIContainer()
# Resolve and use components
face_detector = container.resolve(FaceDetector)
```

---

## Running Tests

```bash
# All tests
pytest ai_enhanced_gaze_tracking/tests/ -v

# Property-based tests only
pytest ai_enhanced_gaze_tracking/tests/ -v -k "property"

# Single test file
pytest ai_enhanced_gaze_tracking/tests/test_graceful_degradation.py -v
```
