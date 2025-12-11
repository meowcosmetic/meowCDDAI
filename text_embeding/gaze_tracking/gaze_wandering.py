"""
Gaze Wandering Detection - Phát hiện "nhìn vô định"
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class WanderingPeriod:
    """Một period "nhìn vô định" """
    start_time: float  # Giây
    end_time: float  # Giây
    duration: float  # Giây
    start_frame: int
    end_frame: int
    total_frames: int
    reason: str  # Lý do: "no_target", "stable_center", etc.


class GazeWanderingDetector:
    """
    Phát hiện "nhìn vô định" (gaze wandering)
    
    Định nghĩa: Trẻ nhìn vào khoảng không, không focus vào object cụ thể nào
    - Gaze ổn định (không di chuyển)
    - Không nhìn vào objects (< 20%)
    - Không nhìn vào adult (< 20%)
    - Nhìn về camera/center (offset < 0.2)
    - Không có target để focus
    """
    
    def __init__(self, config):
        """
        Args:
            config: GazeConfig instance
        """
        self.config = config
        
        # Current state
        self.is_currently_wandering: bool = False
        self.wandering_start_time: Optional[float] = None
        self.wandering_start_frame: Optional[int] = None
        
        # Window tracking
        self.wandering_window: List[Dict[str, Any]] = []  # List of frame states
        self.window_size = config.WANDERING_WINDOW_SIZE
        
        # Timeline storage
        self.wandering_periods: List[WanderingPeriod] = []
        self.total_wandering_frames: int = 0
    
    def update(self,
               frame_count: int,
               current_time: float,
               is_stable: bool,
               looking_at_object_ratio: float,
               looking_at_adult_ratio: float,
               adult_face_exists: bool,
               gaze_offset_x: float,
               gaze_offset_y: float,
               fps: float = 30.0,
               gaze_3d_result: Optional[Tuple[str, float]] = None) -> Optional[WanderingPeriod]:
        """
        Update wandering detection với frame mới
        
        Args:
            frame_count: Current frame number
            current_time: Current time in seconds
            is_stable: Gaze có stable không
            looking_at_object_ratio: Tỷ lệ nhìn vào objects trong window
            looking_at_adult_ratio: Tỷ lệ nhìn vào adult trong window
            adult_face_exists: Có adult face trong frame không
            gaze_offset_x: Gaze offset X (normalized)
            gaze_offset_y: Gaze offset Y (normalized)
            fps: Frames per second
            gaze_3d_result: (object_id, confidence) từ 3D gaze nếu có
        
        Returns:
            WanderingPeriod nếu có period mới kết thúc, None nếu không
        """
        if not self.config.ENABLE_WANDERING_DETECTION:
            return None
        
        # Update window
        frame_state = {
            'is_stable': is_stable,
            'looking_at_object_ratio': looking_at_object_ratio,
            'looking_at_adult_ratio': looking_at_adult_ratio,
            'adult_face_exists': adult_face_exists,
            'gaze_offset_x': gaze_offset_x,
            'gaze_offset_y': gaze_offset_y,
            'gaze_3d_result': gaze_3d_result
        }
        
        self.wandering_window.append(frame_state)
        if len(self.wandering_window) > self.window_size:
            self.wandering_window.pop(0)
        
        # Tính average ratios trong window
        if len(self.wandering_window) >= self.window_size:
            avg_object_ratio = sum(s['looking_at_object_ratio'] for s in self.wandering_window) / len(self.wandering_window)
            avg_adult_ratio = sum(s['looking_at_adult_ratio'] for s in self.wandering_window) / len(self.wandering_window)
            avg_stable = all(s['is_stable'] for s in self.wandering_window[-5:])  # Check last 5 frames
            avg_offset_x = sum(abs(s['gaze_offset_x']) for s in self.wandering_window) / len(self.wandering_window)
            avg_offset_y = sum(abs(s['gaze_offset_y']) for s in self.wandering_window) / len(self.wandering_window)
            
            # Check 3D gaze result (nếu có)
            has_3d_gaze_target = False
            if gaze_3d_result:
                object_id, confidence = gaze_3d_result
                if confidence > self.config.MIN_3D_GAZE_CONFIDENCE:
                    has_3d_gaze_target = True
            
            # Logic "nhìn vô định"
            is_wandering = (
                avg_stable and  # Mắt không di chuyển (gaze ổn định)
                avg_object_ratio < self.config.WANDERING_OBJECT_RATIO_THRESHOLD and  # Hầu như không nhìn object
                avg_adult_ratio < self.config.WANDERING_ADULT_RATIO_THRESHOLD and  # Hầu như không nhìn adult
                (not adult_face_exists or avg_adult_ratio < self.config.WANDERING_ADULT_RATIO_THRESHOLD) and  # Không có adult hoặc không nhìn vào adult
                avg_offset_x < self.config.WANDERING_GAZE_OFFSET_THRESHOLD and  # Nhìn "thẳng vô máy"
                avg_offset_y < self.config.WANDERING_GAZE_OFFSET_THRESHOLD and  # Nhìn "thẳng vô máy"
                not has_3d_gaze_target  # 3D gaze không detect được target cụ thể
            )
            
            if is_wandering:
                if not self.is_currently_wandering:
                    # Bắt đầu wandering period
                    self.is_currently_wandering = True
                    self.wandering_start_time = current_time
                    self.wandering_start_frame = frame_count
                    logger.debug(f"[Wandering] Started wandering at {current_time:.2f}s")
            else:
                if self.is_currently_wandering:
                    # Kết thúc wandering period
                    ended_period = self._end_wandering_period(
                        frame_count,
                        current_time,
                        fps
                    )
                    self.is_currently_wandering = False
                    self.wandering_start_time = None
                    self.wandering_start_frame = None
                    
                    logger.debug(f"[Wandering] Ended wandering at {current_time:.2f}s")
                    
                    return ended_period
        
        return None
    
    def _end_wandering_period(self,
                             end_frame: int,
                             end_time: float,
                             fps: float) -> Optional[WanderingPeriod]:
        """
        Kết thúc một wandering period và lưu vào timeline
        """
        if self.wandering_start_time is None or self.wandering_start_frame is None:
            return None
        
        duration = end_time - self.wandering_start_time
        total_frames = end_frame - self.wandering_start_frame
        
        # Tạo WanderingPeriod
        period = WanderingPeriod(
            start_time=self.wandering_start_time,
            end_time=end_time,
            duration=duration,
            start_frame=self.wandering_start_frame,
            end_frame=end_frame,
            total_frames=total_frames,
            reason="no_target"  # Không có target để focus
        )
        
        # Lưu vào timeline
        self.wandering_periods.append(period)
        self.total_wandering_frames += total_frames
        
        logger.info(f"[Wandering] Wandering period: {duration:.2f}s from {self.wandering_start_time:.2f}s to {end_time:.2f}s")
        
        return period
    
    def finalize(self, final_frame: int, final_time: float, fps: float) -> Optional[WanderingPeriod]:
        """
        Finalize - kết thúc wandering period cuối cùng nếu còn active
        """
        if self.is_currently_wandering:
            return self._end_wandering_period(
                final_frame,
                final_time,
                fps
            )
        return None
    
    def calculate_wandering_score(self, total_frames: int) -> Tuple[float, float]:
        """
        Tính wandering score và percentage
        
        Args:
            total_frames: Tổng số frames trong video
        
        Returns:
            (wandering_score, wandering_percentage)
            - wandering_score: 0-100 (cao hơn = nhìn vô định nhiều hơn)
            - wandering_percentage: 0-100 (% thời gian nhìn vô định)
        """
        if total_frames == 0:
            return 0.0, 0.0
        
        wandering_percentage = (self.total_wandering_frames / total_frames) * 100
        
        # Wandering score: dựa trên percentage và số periods
        # Nhiều periods ngắn = có thể là dấu hiệu ASD
        period_count = len(self.wandering_periods)
        avg_period_duration = sum(p.duration for p in self.wandering_periods) / period_count if period_count > 0 else 0
        
        # Score = percentage + penalty cho nhiều periods
        base_score = wandering_percentage
        period_penalty = min(20, period_count * 2)  # Max 20 points
        
        wandering_score = min(100, base_score + period_penalty)
        
        return wandering_score, wandering_percentage
    
    def get_wandering_timeline(self) -> List[Dict[str, Any]]:
        """
        Get wandering timeline dưới dạng list of dicts
        """
        return [
            {
                'start_time': round(p.start_time, 2),
                'end_time': round(p.end_time, 2),
                'duration': round(p.duration, 2),
                'start_frame': p.start_frame,
                'end_frame': p.end_frame,
                'total_frames': p.total_frames,
                'reason': p.reason
            }
            for p in self.wandering_periods
        ]
    
    def reset(self) -> None:
        """Reset detector (dùng khi bắt đầu video mới)"""
        self.is_currently_wandering = False
        self.wandering_start_time = None
        self.wandering_start_frame = None
        self.wandering_window.clear()
        self.wandering_periods.clear()
        self.total_wandering_frames = 0
        logger.info("[Wandering] Detector reset")

