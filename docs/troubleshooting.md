# AI-Enhanced Gaze Tracking System — Troubleshooting Guide

## Diagnostic Information

When something goes wrong, the `ErrorHandler` provides detailed diagnostics:

```python
from ai_enhanced_gaze_tracking.components.error_handling.error_handler import ErrorHandler

handler = ErrorHandler()
diagnostics = handler.get_diagnostics()

print("System state:", diagnostics["system_state"])
print("Disabled components:", diagnostics["disabled_components"])
print("Active fallbacks:", diagnostics["active_fallbacks"])
print("Recent errors:", diagnostics["recent_errors"])
```

The system state will be one of: `normal`, `degraded`, `minimal`, `recovering`, `failed`.

---

## Common Issues

### Face detection not working

**Symptoms:** No face detected, tracking stops immediately.

**Causes and fixes:**

1. Poor lighting — ensure the subject's face is evenly lit. Avoid strong backlighting.
2. Face too small — the subject may be too far from the camera. Minimum face size is ~10% of frame width.
3. MediaPipe not installed — run `pip install mediapipe>=0.10.0`.
4. Camera not accessible — check that no other application is using the camera.

**System response:** The `HybridFaceDetector` automatically falls back to the OpenCV-based detector. The `ErrorHandler` will log the failure and set the component to `using_fallback=True`.

---

### Gaze estimation inaccurate

**Symptoms:** Gaze point is consistently offset or jumps erratically.

**Causes and fixes:**

1. Camera angle not calibrated — run the reference-point calibration procedure (see `docs/setup_guide.md`).
2. Head pose compensation disabled — ensure `head_pose_compensation=True` in config.
3. AI model not loaded — set `ai_gaze_model_path` to a valid model file, or leave as `None` to use the traditional CV fallback.
4. Low quality score — check `QualityMetrics.overall_quality`. Values below 0.4 indicate unreliable estimates.

---

### System running below 25 FPS

**Symptoms:** Processing speed drops below the 25 FPS minimum (Req 7.1).

**Causes and fixes:**

1. GPU acceleration disabled — set `gpu_acceleration=True` and ensure CUDA is installed.
2. Too many background processes — close other CPU/GPU-intensive applications.
3. Object detection running every frame — increase `object_detection_interval` (default: 5 frames).
4. AI ensemble with many models — reduce `gaze_ensemble_models` list.

**System response:** The `PerformanceOptimizer` automatically reduces processing complexity when load is high (Req 7.2). Check `SystemState` — if it is `DEGRADED` or `MINIMAL`, the system has already applied automatic reductions.

---

### Camera feed interrupted

**Symptoms:** Tracking stops mid-session, error logged for `camera` component.

**Causes and fixes:**

1. USB camera disconnected — reconnect and restart the session.
2. Camera in use by another application — close the other application.
3. Driver issue — reinstall camera drivers.

**System response:** The `ErrorHandler.handle_camera_interruption()` method is called automatically. The system attempts recovery after the cooldown period (default: 10 seconds). User guidance is available via:

```python
handler.get_user_guidance(component="camera")
```

---

### High memory usage / system crashes

**Symptoms:** Memory usage grows over time, system becomes unresponsive.

**Causes and fixes:**

1. Long sessions without cleanup — the system manages memory automatically, but very long sessions (>2 hours) may accumulate data. Restart the session periodically.
2. Memory limit too high — reduce `memory_limit_mb` in config (default: 2048 MB).
3. Large video frames — reduce input resolution if possible.

**System response:** When memory exceeds `MEMORY_HIGH_MB` (2500 MB), the system transitions to `DEGRADED` state. Above `MEMORY_CRITICAL_MB` (3000 MB), it transitions to `MINIMAL` state. The system never crashes — it degrades gracefully (Req 8.4, Property 20).

---

### Quality alerts firing frequently

**Symptoms:** Many `QualityAlert` objects returned, `overall_quality` consistently low.

**Causes and fixes:**

1. Poor lighting — see lighting recommendations in `docs/setup_guide.md`.
2. Subject not facing camera — ask the subject to face the camera more directly.
3. Occlusion — ensure nothing blocks the subject's face (glasses frames, hands, etc.).
4. Motion blur — reduce subject movement or increase camera shutter speed.

**Checking quality:**

```python
from ai_enhanced_gaze_tracking.components.quality_assessment.quality_assessor import GazeQualityAssessor

assessor = GazeQualityAssessor()
alerts = assessor.check_for_alerts(quality_metrics)
for alert in alerts:
    print(f"[{alert.severity.upper()}] {alert.message}")
    print(f"  Suggested action: {alert.suggested_action}")
```

---

### Unreliable data segments flagged

**Symptoms:** `flag_unreliable_data()` returns many indices.

**Causes:** Low quality frames due to any combination of the above issues.

**Handling flagged data:**

```python
unreliable_indices = assessor.flag_unreliable_data(quality_metrics_list)
reliable_estimates = [
    est for i, est in enumerate(gaze_sequence)
    if i not in set(unreliable_indices)
]
```

Adjust `quality_threshold` (default: 0.4) to be more or less strict:

```python
assessor = GazeQualityAssessor(quality_threshold=0.3)  # more permissive
```

---

## Logging

Enable debug logging to get detailed component-level information:

```python
from ai_enhanced_gaze_tracking.core.logging_config import setup_logging
setup_logging(level="DEBUG", log_file="gaze_debug.log")
```

Or via environment variable:

```bash
set GAZE_LOG_LEVEL=DEBUG
set GAZE_LOG_FILE=gaze_debug.log
```

---

## FAQ

**Q: Can I use the system without a GPU?**
A: Yes. Set `gpu_acceleration=False`. Performance will be lower — expect 15–25 FPS on a modern CPU.

**Q: Does the system work with pre-recorded video?**
A: Yes. Pass video frames directly to the face detector and gaze estimator. The real-time performance requirements apply to live video; offline processing has no FPS constraint.

**Q: How do I disable the AI gaze model and use only traditional CV?**
A: Set `gaze_estimation_method="3d"` or `"2d"` in config, or leave `ai_gaze_model_path=None`.

**Q: Will existing API calls break after upgrading?**
A: No. The system maintains full backward compatibility (Req 10.1). Set `legacy_api_support=True` (default) to ensure existing response formats are preserved.

**Q: How do I tune focus detection sensitivity?**
A: Adjust `min_focus_duration` (default: 1.0 second) and `focus_stability_threshold` (default: 0.02) in config. Lower `min_focus_duration` detects shorter focus events; lower `focus_stability_threshold` requires more stable gaze.

**Q: The system reports `system_state: degraded` — is this a problem?**
A: Degraded mode means one or more components are using fallbacks. Core tracking continues. Check `diagnostics["active_fallbacks"]` to see which components are affected and refer to the relevant section above.
