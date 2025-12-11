"""
Focus Timeline Analysis - Track focus periods cho từng object
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FocusPeriod:
    """Một period focus vào một object"""
    object_id: str  # Format: "class_track_id" (e.g., "book_1", "cup_3")
    start_time: float  # Giây
    end_time: float  # Giây
    duration: float  # Giây
    start_frame: int
    end_frame: int
    total_frames: int
    class_name: str
    track_id: Optional[int] = None


class FocusTimeline:
    """Quản lý timeline focus vào các objects"""
    
    def __init__(self, stability_threshold: float = 0.1, min_focus_duration: float = 0.5):
        """
        Args:
            stability_threshold: Threshold để coi gaze là stable (normalized variance)
            min_focus_duration: Thời gian tối thiểu để coi là focus (giây)
        """
        self.stability_threshold = stability_threshold
        self.min_focus_duration = min_focus_duration
        
        # Current state
        self.current_focus: Optional[str] = None  # object_id hiện tại đang focus
        self.focus_start_time: Optional[float] = None
        self.focus_start_frame: Optional[int] = None
        self.gaze_history: List[tuple] = []  # [(x, y), ...] - lịch sử gaze positions
        self.gaze_history_size: int = 30  # Số frames trong history
        
        # Timeline storage
        self.focus_periods: List[FocusPeriod] = []
        self.object_focus_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_duration': 0.0,
            'total_frames': 0,
            'focus_count': 0,
            'periods': []
        })
    
    def update(self, 
               frame_count: int,
               current_time: float,
               gaze_pos: Optional[tuple],
               tracked_objects: List[Dict[str, Any]],
               fps: float = 30.0,
               gaze_3d_result: Optional[Tuple[str, float]] = None) -> Optional[FocusPeriod]:
        """
        Update timeline với frame mới
        
        Args:
            frame_count: Current frame number
            current_time: Current time in seconds
            gaze_pos: Current gaze position (x, y) hoặc None
            tracked_objects: List of tracked objects với track_id
            fps: Frames per second
        
        Returns:
            FocusPeriod nếu có period mới kết thúc, None nếu không
        """
        # Update gaze history
        if gaze_pos:
            self.gaze_history.append(gaze_pos)
            if len(self.gaze_history) > self.gaze_history_size:
                self.gaze_history.pop(0)
        
        # Tìm object đang được focus (ưu tiên 3D gaze nếu có)
        looking_at_object = self._find_focused_object(gaze_pos, tracked_objects, gaze_3d_result)
        
        # Kiểm tra gaze stability
        is_stable = self._is_gaze_stable()
        
        # Determine current focus
        if looking_at_object and is_stable:
            object_id = looking_at_object['object_id']
            
            if self.current_focus != object_id:
                # Kết thúc focus cũ
                ended_period = None
                if self.current_focus is not None:
                    ended_period = self._end_focus_period(
                        self.current_focus,
                        frame_count,
                        current_time,
                        fps
                    )
                
                # Bắt đầu focus mới
                self.current_focus = object_id
                self.focus_start_time = current_time
                self.focus_start_frame = frame_count
                
                logger.debug(f"[FocusTimeline] Started focusing on {object_id} at {current_time:.2f}s")
                
                return ended_period
            else:
                # Đang tiếp tục focus vào cùng object
                return None
        else:
            # Mất focus
            if self.current_focus is not None:
                ended_period = self._end_focus_period(
                    self.current_focus,
                    frame_count,
                    current_time,
                    fps
                )
                self.current_focus = None
                self.focus_start_time = None
                self.focus_start_frame = None
                
                logger.debug(f"[FocusTimeline] Lost focus at {current_time:.2f}s")
                
                return ended_period
        
        return None
    
    def _find_focused_object(self, 
                            gaze_pos: Optional[tuple],
                            tracked_objects: List[Dict[str, Any]],
                            gaze_3d_result: Optional[Tuple[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Tìm object đang được focus dựa trên gaze position hoặc 3D gaze estimation
        
        Args:
            gaze_pos: (x, y) gaze position trong frame (2D fallback)
            tracked_objects: List of tracked objects
            gaze_3d_result: (object_id, confidence) từ 3D gaze estimation (ưu tiên)
        
        Returns:
            Object dict với thêm 'object_id' và 'gaze_confidence' field, hoặc None
        """
        # Ưu tiên 3D gaze estimation nếu có
        if gaze_3d_result is not None:
            object_id, confidence = gaze_3d_result
            
            if confidence > 0.3:  # Threshold cho confidence
                # Tìm object tương ứng
                for obj in tracked_objects:
                    class_name = obj.get('class', 'unknown')
                    track_id = obj.get('track_id')
                    
                    if track_id is not None:
                        obj_id = f"{class_name}_{track_id}"
                    else:
                        obj_id = f"{class_name}_unknown"
                    
                    if obj_id == object_id:
                        result = obj.copy()
                        result['object_id'] = object_id
                        result['gaze_confidence'] = confidence
                        return result
        
        # Fallback: 2D gaze position
        if gaze_pos is None or len(tracked_objects) == 0:
            return None
        
        gaze_x, gaze_y = gaze_pos
        
        # Tìm object gần gaze position nhất
        best_object = None
        min_distance = float('inf')
        
        for obj in tracked_objects:
            bbox = obj.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            x, y, w, h = bbox
            center_x = x + w / 2
            center_y = y + h / 2
            
            # Tính khoảng cách từ gaze đến center của object
            distance = ((gaze_x - center_x)**2 + (gaze_y - center_y)**2)**0.5
            
            # Kiểm tra xem gaze có trong bbox không (với threshold)
            threshold_ratio = 0.6  # 60% của object size
            threshold_x = w * threshold_ratio
            threshold_y = h * threshold_ratio
            
            if (abs(gaze_x - center_x) < threshold_x and 
                abs(gaze_y - center_y) < threshold_y):
                
                if distance < min_distance:
                    min_distance = distance
                    best_object = obj
        
        if best_object:
            # Tạo object_id: "class_track_id"
            class_name = best_object.get('class', 'unknown')
            track_id = best_object.get('track_id')
            
            if track_id is not None:
                object_id = f"{class_name}_{track_id}"
            else:
                # Fallback nếu không có track_id
                object_id = f"{class_name}_unknown"
            
            result = best_object.copy()
            result['object_id'] = object_id
            result['gaze_confidence'] = 0.5  # Default confidence cho 2D method
            return result
        
        return None
    
    def _is_gaze_stable(self) -> bool:
        """
        Kiểm tra xem gaze có stable không dựa trên history
        
        Returns:
            True nếu gaze stable (variance < threshold)
        """
        if len(self.gaze_history) < 5:  # Cần ít nhất 5 frames
            return False
        
        import numpy as np
        
        # Tính variance của gaze positions
        positions_x = [pos[0] for pos in self.gaze_history]
        positions_y = [pos[1] for pos in self.gaze_history]
        
        variance_x = np.var(positions_x) if len(positions_x) > 1 else 0
        variance_y = np.var(positions_y) if len(positions_y) > 1 else 0
        total_variance = variance_x + variance_y
        
        return total_variance < self.stability_threshold
    
    def _end_focus_period(self,
                          object_id: str,
                          end_frame: int,
                          end_time: float,
                          fps: float) -> Optional[FocusPeriod]:
        """
        Kết thúc một focus period và lưu vào timeline
        
        Returns:
            FocusPeriod nếu duration >= min_focus_duration, None nếu không
        """
        if self.focus_start_time is None or self.focus_start_frame is None:
            return None
        
        duration = end_time - self.focus_start_time
        total_frames = end_frame - self.focus_start_frame
        
        # Chỉ lưu nếu duration đủ dài
        if duration < self.min_focus_duration:
            logger.debug(f"[FocusTimeline] Focus period quá ngắn ({duration:.2f}s < {self.min_focus_duration}s), bỏ qua")
            return None
        
        # Parse object_id để lấy class_name và track_id
        parts = object_id.split('_')
        class_name = parts[0] if len(parts) > 0 else 'unknown'
        track_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        
        # Tạo FocusPeriod
        period = FocusPeriod(
            object_id=object_id,
            start_time=self.focus_start_time,
            end_time=end_time,
            duration=duration,
            start_frame=self.focus_start_frame,
            end_frame=end_frame,
            total_frames=total_frames,
            class_name=class_name,
            track_id=track_id
        )
        
        # Lưu vào timeline
        self.focus_periods.append(period)
        
        # Update stats
        stats = self.object_focus_stats[object_id]
        stats['total_duration'] += duration
        stats['total_frames'] += total_frames
        stats['focus_count'] += 1
        stats['periods'].append(period)
        
        logger.info(f"[FocusTimeline] Focus period: {object_id} from {self.focus_start_time:.2f}s to {end_time:.2f}s ({duration:.2f}s)")
        
        return period
    
    def finalize(self, final_frame: int, final_time: float, fps: float) -> Optional[FocusPeriod]:
        """
        Finalize timeline - kết thúc focus period cuối cùng nếu còn đang active
        
        Call khi kết thúc video
        """
        if self.current_focus is not None:
            return self._end_focus_period(
                self.current_focus,
                final_frame,
                final_time,
                fps
            )
        return None
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Get timeline dưới dạng list of dicts (cho JSON response)
        
        Returns:
            [
                {
                    'object_id': 'book_1',
                    'start_time': 0.5,
                    'end_time': 10.5,
                    'duration': 10.0,
                    'start_frame': 15,
                    'end_frame': 315,
                    'total_frames': 300,
                    'class_name': 'book',
                    'track_id': 1
                },
                ...
            ]
        """
        return [
            {
                'object_id': p.object_id,
                'start_time': round(p.start_time, 2),
                'end_time': round(p.end_time, 2),
                'duration': round(p.duration, 2),
                'start_frame': p.start_frame,
                'end_frame': p.end_frame,
                'total_frames': p.total_frames,
                'class_name': p.class_name,
                'track_id': p.track_id
            }
            for p in self.focus_periods
        ]
    
    def get_object_stats(self, object_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics cho object(s)
        
        Args:
            object_id: Specific object_id, hoặc None cho tất cả
        
        Returns:
            Dictionary với stats
        """
        if object_id:
            return self.object_focus_stats.get(object_id, {
                'total_duration': 0.0,
                'total_frames': 0,
                'focus_count': 0,
                'periods': []
            })
        
        return dict(self.object_focus_stats)
    
    def get_pattern_analysis(self) -> Dict[str, Any]:
        """
        Phân tích pattern: có quay lại nhìn object cũ không?
        
        Returns:
            Dictionary với pattern analysis
        """
        # Group periods by object_id
        object_periods = defaultdict(list)
        for period in self.focus_periods:
            object_periods[period.object_id].append(period)
        
        # Phân tích
        revisited_objects = []
        single_focus_objects = []
        
        for object_id, periods in object_periods.items():
            if len(periods) > 1:
                # Object được nhìn nhiều lần
                revisited_objects.append({
                    'object_id': object_id,
                    'focus_count': len(periods),
                    'total_duration': sum(p.duration for p in periods),
                    'periods': [
                        {
                            'start': p.start_time,
                            'end': p.end_time,
                            'duration': p.duration
                        }
                        for p in periods
                    ]
                })
            else:
                single_focus_objects.append(object_id)
        
        return {
            'revisited_objects': revisited_objects,
            'single_focus_objects': single_focus_objects,
            'revisit_count': len(revisited_objects),
            'total_unique_objects': len(object_periods)
        }
    
    def reset(self) -> None:
        """Reset timeline (dùng khi bắt đầu video mới)"""
        self.current_focus = None
        self.focus_start_time = None
        self.focus_start_frame = None
        self.gaze_history.clear()
        self.focus_periods.clear()
        self.object_focus_stats.clear()
        logger.info("[FocusTimeline] Timeline reset")

