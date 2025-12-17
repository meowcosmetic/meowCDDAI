# Gaze Tracking Refactoring Guide

## ✅ Đã hoàn thành

### 1. Module Structure
```
gaze_tracking/
├── __init__.py          # Module exports
├── config.py            # Constants & Configuration (loại bỏ magic numbers)
├── gpu_utils.py         # GPU Manager (loại bỏ duplication)
├── models.py            # Pydantic models
├── video_processor.py   # Video handling với context managers
├── face_detector.py     # Strategy Pattern cho MediaPipe/OpenCV
├── object_detector.py   # YOLO wrapper class
└── visualizer.py        # Drawing functions
```

### 2. Improvements

#### ✅ Code Duplication - FIXED
- **Before**: GPU detection code duplicate ở lines 30-57 và 58-85
- **After**: `GPUManager` singleton class, chỉ detect một lần

#### ✅ Magic Numbers - FIXED
- **Before**: Hard-coded values rải rác trong code
- **After**: `GazeConfig` class với tất cả constants

#### ✅ Resource Management - IMPROVED
- **Before**: `cap.release()` có thể không được gọi
- **After**: `video_capture()` context manager đảm bảo cleanup

#### ✅ Strategy Pattern - IMPLEMENTED
- **Before**: Nested if-else cho MediaPipe vs OpenCV
- **After**: `FaceDetector` protocol với `MediaPipeFaceDetector` và `OpenCVFaceDetector`

#### ✅ Type Safety - IMPROVED
- **Before**: Không có type hints
- **After**: Protocol, type hints, Pydantic models

## 📋 Cần làm tiếp

### 1. Tạo GazeAnalyzer class
File `gaze_analyzer.py` cần implement:

```python
class GazeAnalyzer:
    def __init__(self, config, gpu_manager, face_detector, object_detector):
        ...
    
    def process_video(self, video_path: str, show_video: bool = False):
        """Main processing logic - tách từ analyze_gaze()"""
        ...
    
    def analyze_frame(self, frame, frame_count, fps):
        """Analyze single frame"""
        ...
    
    def calculate_results(self):
        """Calculate final metrics"""
        ...
```

### 2. Refactor routes_screening_gaze.py

**Before** (1498 lines):
```python
async def analyze_gaze(...):
    # 900+ lines of processing logic
    ...
```

**After** (target ~100 lines):
```python
@router.post("/analyze", response_model=GazeAnalysisResponse)
async def analyze_gaze(...):
    """API endpoint - chỉ xử lý request/response"""
    with safe_file_cleanup(temp_path):
        analyzer = GazeAnalyzer(config, gpu_manager, face_detector, object_detector)
        results = analyzer.process_video(temp_path, show_video)
        return results
```

### 3. Performance Optimizations

- Cache face detection results
- Optimize object detection interval
- Limit memory usage (clear old tracking data)

### 4. Error Handling

- Validate input files
- Better error messages
- Graceful degradation

## 🎯 Metrics Target

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Cyclomatic Complexity | ~65 | <10 | ⏳ In Progress |
| Lines of Code | ~1400 | <500 | ⏳ In Progress |
| Longest Function | ~900 | <50 | ⏳ In Progress |
| Code Duplication | 2.8% | 0% | ✅ Fixed |
| Maintainability Index | ~30 | >60 | ⏳ In Progress |

## 📝 Usage Example

```python
from gaze_tracking import GazeConfig, GPUManager
from gaze_tracking.face_detector import create_face_detector
from gaze_tracking.object_detector import ObjectDetector
from gaze_tracking.gaze_analyzer import GazeAnalyzer

# Initialize
config = GazeConfig()
gpu_manager = GPUManager()
face_detector = create_face_detector(use_mediapipe=True)
object_detector = ObjectDetector(config, gpu_manager)

# Analyze
analyzer = GazeAnalyzer(config, gpu_manager, face_detector, object_detector)
results = analyzer.process_video("video.mp4", show_video=True)
```

## 🔄 Migration Steps

1. ✅ Create module structure
2. ✅ Extract config and constants
3. ✅ Extract GPU utilities
4. ✅ Extract face detector
5. ✅ Extract object detector
6. ✅ Extract visualizer
7. ⏳ Create GazeAnalyzer class
8. ⏳ Refactor main API endpoint
9. ⏳ Update tests
10. ⏳ Performance testing

