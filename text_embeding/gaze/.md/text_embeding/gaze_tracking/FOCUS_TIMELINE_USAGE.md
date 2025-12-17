# Focus Timeline Analysis - Hướng dẫn sử dụng

## Tổng quan

Module `FocusTimeline` track các khoảng thời gian trẻ focus vào từng object cụ thể, cho phép:
- Biết chính xác trẻ nhìn object nào, lúc nào, bao lâu
- Tính tổng thời gian focus từng object riêng biệt
- Phát hiện pattern: có quay lại nhìn object cũ không?

## Cách sử dụng

### 1. Khởi tạo

```python
from gaze_tracking.focus_timeline import FocusTimeline

# Khởi tạo với config
timeline = FocusTimeline(
    stability_threshold=0.1,  # Gaze variance threshold
    min_focus_duration=0.5    # Thời gian tối thiểu để coi là focus (giây)
)
```

### 2. Update mỗi frame

```python
for frame_count, frame in enumerate(frames):
    current_time = frame_count / fps
    
    # Estimate gaze position
    gaze_pos = estimate_gaze(frame)  # (x, y) trong frame
    
    # Detect và track objects
    tracked_objects = detector.detect(frame, frame_count=frame_count)
    # tracked_objects có format:
    # [
    #   {
    #     'class': 'book',
    #     'track_id': 1,
    #     'bbox': [x, y, w, h],
    #     'confidence': 0.85,
    #     ...
    #   },
    #   ...
    # ]
    
    # Update timeline
    ended_period = timeline.update(
        frame_count=frame_count,
        current_time=current_time,
        gaze_pos=gaze_pos,
        tracked_objects=tracked_objects,
        fps=fps
    )
    
    # Xử lý period vừa kết thúc (nếu có)
    if ended_period:
        print(f"Focus ended: {ended_period.object_id} "
              f"({ended_period.duration:.2f}s)")
```

### 3. Finalize khi kết thúc video

```python
# Kết thúc focus period cuối cùng (nếu còn active)
final_period = timeline.finalize(
    final_frame=total_frames,
    final_time=total_duration,
    fps=fps
)
```

### 4. Lấy kết quả

```python
# Lấy timeline dưới dạng list
timeline_data = timeline.get_timeline()
# Output:
# [
#   {
#     'object_id': 'book_1',
#     'start_time': 0.5,
#     'end_time': 10.5,
#     'duration': 10.0,
#     'start_frame': 15,
#     'end_frame': 315,
#     'total_frames': 300,
#     'class_name': 'book',
#     'track_id': 1
#   },
#   {
#     'object_id': 'cup_3',
#     'start_time': 12.0,
#     'end_time': 15.0,
#     'duration': 3.0,
#     ...
#   },
#   {
#     'object_id': 'book_1',  # Quay lại nhìn book_1
#     'start_time': 20.0,
#     'end_time': 35.0,
#     'duration': 15.0,
#     ...
#   }
# ]

# Lấy stats cho từng object
stats = timeline.get_object_stats()
# Output:
# {
#   'book_1': {
#     'total_duration': 25.0,  # Tổng thời gian focus
#     'total_frames': 750,
#     'focus_count': 2,  # Số lần focus (2 lần)
#     'periods': [...]
#   },
#   'cup_3': {
#     'total_duration': 3.0,
#     'total_frames': 90,
#     'focus_count': 1,
#     'periods': [...]
#   }
# }

# Lấy stats cho object cụ thể
book_1_stats = timeline.get_object_stats('book_1')

# Phân tích pattern
pattern = timeline.get_pattern_analysis()
# Output:
# {
#   'revisited_objects': [
#     {
#       'object_id': 'book_1',
#       'focus_count': 2,
#       'total_duration': 25.0,
#       'periods': [
#         {'start': 0.5, 'end': 10.5, 'duration': 10.0},
#         {'start': 20.0, 'end': 35.0, 'duration': 15.0}
#       ]
#     }
#   ],
#   'single_focus_objects': ['cup_3'],
#   'revisit_count': 1,
#   'total_unique_objects': 2
# }
```

## Tích hợp vào GazeAnalyzer

```python
from gaze_tracking.focus_timeline import FocusTimeline

class GazeAnalyzer:
    def __init__(self, config, ...):
        ...
        self.focus_timeline = FocusTimeline(
            stability_threshold=config.GAZE_STABILITY_THRESHOLD,
            min_focus_duration=config.MIN_FOCUSING_DURATION
        )
    
    def process_video(self, video_path, show_video=False):
        ...
        for frame_count, frame in enumerate(frames):
            # Detect faces, objects, estimate gaze
            ...
            
            # Update timeline
            self.focus_timeline.update(
                frame_count=frame_count,
                current_time=current_time,
                gaze_pos=gaze_pos,
                tracked_objects=tracked_objects,
                fps=fps
            )
        
        # Finalize
        self.focus_timeline.finalize(...)
        
        # Include trong response
        return {
            ...
            'focus_timeline': self.focus_timeline.get_timeline(),
            'object_focus_stats': self.focus_timeline.get_object_stats(),
            'pattern_analysis': self.focus_timeline.get_pattern_analysis()
        }
```

## Lợi ích

1. **Precision**: Biết chính xác trẻ nhìn object nào (book_1 vs book_2)
2. **Duration Tracking**: Tính tổng thời gian focus từng object
3. **Pattern Detection**: Phát hiện có quay lại nhìn object cũ không
4. **Detailed Analysis**: Timeline chi tiết với start/end time cho mỗi period

## Example Output

```json
{
  "focus_timeline": [
    {
      "object_id": "book_1",
      "start_time": 0.5,
      "end_time": 10.5,
      "duration": 10.0,
      "class_name": "book",
      "track_id": 1
    },
    {
      "object_id": "cup_3",
      "start_time": 12.0,
      "end_time": 15.0,
      "duration": 3.0,
      "class_name": "cup",
      "track_id": 3
    },
    {
      "object_id": "book_1",
      "start_time": 20.0,
      "end_time": 35.0,
      "duration": 15.0,
      "class_name": "book",
      "track_id": 1
    }
  ],
  "object_focus_stats": {
    "book_1": {
      "total_duration": 25.0,
      "total_frames": 750,
      "focus_count": 2
    },
    "cup_3": {
      "total_duration": 3.0,
      "total_frames": 90,
      "focus_count": 1
    }
  },
  "pattern_analysis": {
    "revisited_objects": [
      {
        "object_id": "book_1",
        "focus_count": 2,
        "total_duration": 25.0
      }
    ],
    "revisit_count": 1,
    "total_unique_objects": 2
  }
}
```

