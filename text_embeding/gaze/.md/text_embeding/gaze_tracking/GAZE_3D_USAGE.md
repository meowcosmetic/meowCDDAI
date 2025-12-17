# 3D Gaze Estimation - Hướng dẫn sử dụng

## Tổng quan

Module `GazeEstimator3D` tính toán chính xác hướng nhìn sử dụng:
- **Head Pose Estimation (6DoF)**: Tính toán rotation và translation của đầu
- **Eye Gaze Direction**: Tính hướng nhìn từ pupil positions
- **Ray Casting**: Tìm object nào intersect với gaze ray
- **Confidence Score**: Độ chắc chắn của gaze estimation

## Lợi ích

1. **Accuracy**: Tính chính xác hướng nhìn kể cả khi đầu nghiêng/xoay
2. **Reduced False Positives**: Giảm nhầm lẫn (tưởng nhìn object A nhưng thật ra nhìn object B)
3. **Confidence Score**: Có confidence score cho độ chắc chắn (0.0 - 1.0)

## Cách sử dụng

### 1. Khởi tạo

```python
from gaze_tracking.gaze_estimation_3d import GazeEstimator3D

# Khởi tạo với frame size
gaze_estimator = GazeEstimator3D(
    image_width=640,
    image_height=480
)
```

### 2. Estimate 3D Gaze

```python
# Trong vòng lặp xử lý video
for frame_count, frame in enumerate(frames):
    # Detect faces với MediaPipe
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]  # Child face
        
        # Detect và track objects
        tracked_objects = detector.detect(frame, frame_count=frame_count)
        
        # Estimate 3D gaze
        object_id, confidence = gaze_estimator.estimate_3d_gaze(
            face_landmarks=face_landmarks,
            tracked_objects=tracked_objects
        )
        
        if object_id:
            print(f"Looking at {object_id} with confidence {confidence:.2f}")
        else:
            print("Not looking at any tracked object")
```

### 3. Tích hợp với FocusTimeline

```python
from gaze_tracking.focus_timeline import FocusTimeline
from gaze_tracking.gaze_estimation_3d import GazeEstimator3D

timeline = FocusTimeline()
gaze_estimator = GazeEstimator3D(image_width=w, image_height=h)

for frame_count, frame in enumerate(frames):
    current_time = frame_count / fps
    
    # Detect faces và objects
    face_landmarks = detect_face(frame)
    tracked_objects = detector.detect(frame, frame_count=frame_count)
    
    # Estimate 3D gaze
    gaze_3d_result = None
    if face_landmarks:
        object_id, confidence = gaze_estimator.estimate_3d_gaze(
            face_landmarks, tracked_objects
        )
        if object_id and confidence > 0.3:
            gaze_3d_result = (object_id, confidence)
    
    # Update timeline với 3D gaze result
    timeline.update(
        frame_count=frame_count,
        current_time=current_time,
        gaze_pos=None,  # Không cần 2D nếu có 3D
        tracked_objects=tracked_objects,
        fps=fps,
        gaze_3d_result=gaze_3d_result  # ✅ Ưu tiên 3D
    )
```

### 4. Update camera params khi frame size thay đổi

```python
# Khi resize frame
h, w = frame.shape[:2]
gaze_estimator.update_camera_params(width=w, height=h)
```

## API Reference

### `estimate_3d_gaze(face_landmarks, tracked_objects)`

Estimate 3D gaze và tìm object đang được nhìn.

**Args:**
- `face_landmarks`: MediaPipe face landmarks
- `tracked_objects`: List of tracked objects với bbox và track_id

**Returns:**
- `(object_id, confidence)`: 
  - `object_id`: "class_track_id" (e.g., "book_1", "cup_3") hoặc None
  - `confidence`: 0.0 - 1.0 (độ chắc chắn)

### `estimate_head_pose(face_landmarks)`

Estimate head pose (6DoF) sử dụng solvePnP.

**Returns:**
- `(success, rotation_vec, translation_vec)`

### `calculate_eye_direction(face_landmarks)`

Tính eye gaze direction trong head coordinate system.

**Returns:**
- Eye direction vector (normalized) hoặc None

## Technical Details

### Head Pose Estimation

Sử dụng `cv2.solvePnP` với:
- **3D Model Points**: 6 key points từ face model
- **2D Image Points**: Tương ứng từ MediaPipe landmarks
- **Camera Matrix**: Estimated từ frame size

### Eye Gaze Direction

- Sử dụng pupil positions (landmarks 468, 473)
- Tính offset từ eye center
- Transform sang head coordinate system

### Ray Casting

- Gaze ray: `origin + direction * t`
- Intersection với 3D bounding box (slab method)
- Confidence dựa trên distance và object size

### 3D Bounding Box

- Convert 2D bbox sang 3D dựa trên estimated depth
- Project sử dụng camera intrinsics

## Example Output

```python
# Estimate gaze
object_id, confidence = gaze_estimator.estimate_3d_gaze(face_landmarks, tracked_objects)

# Output examples:
# ("book_1", 0.85)  # Looking at book_1 with 85% confidence
# ("cup_3", 0.62)   # Looking at cup_3 with 62% confidence
# (None, 0.0)       # Not looking at any tracked object
```

## Performance Notes

- **Head Pose Estimation**: ~1-2ms per frame
- **Ray Casting**: ~0.5ms per object
- **Total**: ~2-5ms per frame (depends on number of objects)

## Limitations

1. **Camera Calibration**: Sử dụng estimated camera params, có thể không chính xác 100%
2. **Depth Estimation**: Giả định depth dựa trên face size
3. **3D Model**: Sử dụng average face model, có thể không phù hợp với mọi người

## Future Improvements

1. Camera calibration từ video
2. Personalized 3D face model
3. Multi-person gaze tracking
4. Temporal smoothing cho gaze direction

