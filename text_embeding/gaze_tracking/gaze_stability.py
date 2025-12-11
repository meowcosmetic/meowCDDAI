"""
Improved Gaze Stability Calculation
- Normalized by interocular distance
- Head motion compensation
- Outlier removal & smoothing
- RMS distance metric
- Adaptive threshold
"""
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


def calculate_interocular_distance(face_landmarks, w: int, h: int) -> float:
    """
    Tính khoảng cách giữa 2 mắt (interocular distance) để normalize gaze positions.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        w: Frame width
        h: Frame height
    
    Returns:
        Interocular distance in pixels
    """
    try:
        # MediaPipe landmark indices for eye corners
        LEFT_EYE_CORNER = 33   # Left eye outer corner
        RIGHT_EYE_CORNER = 362  # Right eye outer corner
        
        left_eye = face_landmarks.landmark[LEFT_EYE_CORNER]
        right_eye = face_landmarks.landmark[RIGHT_EYE_CORNER]
        
        # Calculate distance in pixels
        dx = (right_eye.x - left_eye.x) * w
        dy = (right_eye.y - left_eye.y) * h
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance
    except Exception as e:
        logger.warning(f"[GazeStability] Error calculating interocular distance: {str(e)}")
        return 1.0  # Fallback


def remove_outliers(values: List[float], z_threshold: float = 2.5) -> List[float]:
    """
    Loại bỏ outliers sử dụng Z-score method.
    
    Args:
        values: List các giá trị
        z_threshold: Ngưỡng Z-score (mặc định 2.5)
    
    Returns:
        List các giá trị sau khi loại bỏ outliers
    """
    if len(values) < 3:
        return values
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array)
    
    if std == 0:
        return values
    
    # Z-score
    z_scores = np.abs((values_array - mean) / std)
    
    # Giữ lại các giá trị có Z-score < threshold
    filtered = [v for i, v in enumerate(values) if z_scores[i] < z_threshold]
    
    return filtered if len(filtered) >= 2 else values


def smooth_values(values: List[float], window_size: int = 3) -> List[float]:
    """
    Làm mượt giá trị bằng moving average.
    
    Args:
        values: List các giá trị
        window_size: Kích thước window (mặc định 3)
    
    Returns:
        List các giá trị đã được làm mượt
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        window = values[start:end]
        smoothed.append(np.mean(window))
    
    return smoothed


def calculate_rms_distance(positions_x: List[float], positions_y: List[float]) -> float:
    """
    Tính RMS (Root Mean Square) distance từ center - metric dễ hiểu hơn variance.
    
    Args:
        positions_x: List các giá trị X
        positions_y: List các giá trị Y
    
    Returns:
        RMS distance (một "bán kính" dispersion)
    """
    if len(positions_x) < 2 or len(positions_y) < 2:
        return 0.0
    
    # Tính center
    center_x = np.mean(positions_x)
    center_y = np.mean(positions_y)
    
    # Tính distances từ center
    distances = []
    for x, y in zip(positions_x, positions_y):
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        distances.append(dist)
    
    # RMS distance
    rms = np.sqrt(np.mean([d**2 for d in distances]))
    
    return rms


def compensate_head_motion(
    gaze_positions: List[Tuple[float, float]],
    head_poses: List[Tuple[float, float, float]]
) -> List[Tuple[float, float]]:
    """
    Bù trừ chuyển động đầu (head motion) khỏi gaze positions.
    
    Args:
        gaze_positions: List các (gaze_x, gaze_y)
        head_poses: List các (yaw, pitch, roll)
    
    Returns:
        List các gaze positions đã được bù trừ head motion
    """
    if len(gaze_positions) != len(head_poses) or len(gaze_positions) < 2:
        return gaze_positions
    
    compensated = []
    
    # Lấy head pose đầu tiên làm reference
    ref_yaw, ref_pitch, ref_roll = head_poses[0]
    
    for (gaze_x, gaze_y), (yaw, pitch, roll) in zip(gaze_positions, head_poses):
        # Tính sự thay đổi head pose
        delta_yaw = yaw - ref_yaw
        delta_pitch = pitch - ref_pitch
        
        # Bù trừ (giả định: head rotation ảnh hưởng đến gaze position)
        # Scale factor có thể điều chỉnh
        scale = 0.5  # Giảm ảnh hưởng của head motion
        comp_x = gaze_x - delta_yaw * scale
        comp_y = gaze_y - delta_pitch * scale
        
        compensated.append((comp_x, comp_y))
    
    return compensated


class ImprovedGazeStabilityCalculator:
    """
    Improved Gaze Stability Calculator với:
    - Normalization by interocular distance
    - Head motion compensation
    - Outlier removal & smoothing
    - RMS distance metric
    - Adaptive threshold
    """
    
    def __init__(
        self,
        window_size_ms: float = 200.0,  # 200ms window (6-7 frames tại 30fps)
        rms_threshold: float = 0.02,     # RMS threshold (normalized)
        z_threshold: float = 2.5,        # Z-score threshold cho outlier removal
        smoothing_window: int = 3,       # Smoothing window size
        use_head_compensation: bool = True,
        use_outlier_removal: bool = True,
        use_smoothing: bool = True,
        adaptive_threshold: bool = False
    ):
        """
        Args:
            window_size_ms: Window size tính bằng milliseconds (100-300ms recommended)
            rms_threshold: RMS distance threshold (normalized by interocular distance)
            z_threshold: Z-score threshold cho outlier removal
            smoothing_window: Window size cho smoothing
            use_head_compensation: Có bù trừ head motion không
            use_outlier_removal: Có loại bỏ outliers không
            use_smoothing: Có làm mượt không
            adaptive_threshold: Có dùng adaptive threshold không
        """
        self.window_size_ms = window_size_ms
        self.rms_threshold = rms_threshold
        self.z_threshold = z_threshold
        self.smoothing_window = smoothing_window
        self.use_head_compensation = use_head_compensation
        self.use_outlier_removal = use_outlier_removal
        self.use_smoothing = use_smoothing
        self.adaptive_threshold = adaptive_threshold
        
        # Storage
        self.gaze_window: deque = deque(maxlen=100)  # Store (gaze_x, gaze_y, interocular_dist, head_pose, timestamp)
        self.rms_history: deque = deque(maxlen=30)   # Store RMS values for adaptive threshold
    
    def calculate_stability(
        self,
        gaze_x: float,
        gaze_y: float,
        interocular_distance: float,
        head_pose: Optional[Tuple[float, float, float]] = None,
        fps: float = 30.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Tính gaze stability với các cải thiện.
        
        Args:
            gaze_x: Gaze X position (normalized)
            gaze_y: Gaze Y position (normalized)
            interocular_distance: Interocular distance in pixels
            head_pose: (yaw, pitch, roll) nếu có
            fps: Frames per second
            timestamp: Timestamp hiện tại (seconds)
        
        Returns:
            Dict với:
            - is_stable: bool
            - rms_distance: float
            - variance: float (legacy)
            - stability_score: float (0-1, cao hơn = ổn định hơn)
            - details: dict với chi tiết
        """
        # Tính window size theo frames
        window_size_frames = max(3, int(self.window_size_ms * fps / 1000.0))
        
        # Thêm vào window
        if timestamp is None:
            timestamp = len(self.gaze_window) / fps if fps > 0 else len(self.gaze_window)
        
        self.gaze_window.append({
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'interocular_dist': interocular_distance,
            'head_pose': head_pose,
            'timestamp': timestamp
        })
        
        # Cần ít nhất window_size_frames để tính
        if len(self.gaze_window) < window_size_frames:
            return {
                'is_stable': False,
                'rms_distance': float('inf'),
                'variance': float('inf'),
                'stability_score': 0.0,
                'details': {'reason': 'insufficient_data', 'window_size': len(self.gaze_window)}
            }
        
        # Lấy dữ liệu trong window
        window_data = list(self.gaze_window)[-window_size_frames:]
        
        # Extract positions
        positions_x = [d['gaze_x'] for d in window_data]
        positions_y = [d['gaze_y'] for d in window_data]
        interocular_dists = [d['interocular_dist'] for d in window_data]
        head_poses = [d['head_pose'] for d in window_data if d['head_pose'] is not None]
        
        # 1. Normalize by interocular distance
        if len(interocular_dists) > 0 and np.mean(interocular_dists) > 0:
            mean_iod = np.mean(interocular_dists)
            # Normalize positions by interocular distance
            positions_x = [x * mean_iod for x in positions_x]
            positions_y = [y * mean_iod for y in positions_y]
        
        # 2. Head motion compensation
        if self.use_head_compensation and len(head_poses) == len(window_data):
            gaze_positions = list(zip(positions_x, positions_y))
            compensated = compensate_head_motion(gaze_positions, head_poses)
            positions_x = [p[0] for p in compensated]
            positions_y = [p[1] for p in compensated]
        
        # 3. Outlier removal
        if self.use_outlier_removal:
            positions_x = remove_outliers(positions_x, self.z_threshold)
            positions_y = remove_outliers(positions_y, self.z_threshold)
        
        # 4. Smoothing
        if self.use_smoothing and len(positions_x) >= self.smoothing_window:
            positions_x = smooth_values(positions_x, self.smoothing_window)
            positions_y = smooth_values(positions_y, self.smoothing_window)
        
        # 5. Calculate RMS distance (metric chính)
        rms_distance = calculate_rms_distance(positions_x, positions_y)
        
        # 6. Calculate variance (legacy, để so sánh)
        variance_x = np.var(positions_x) if len(positions_x) > 1 else 0
        variance_y = np.var(positions_y) if len(positions_y) > 1 else 0
        total_variance = variance_x + variance_y
        
        # 7. Adaptive threshold (nếu bật)
        threshold = self.rms_threshold
        if self.adaptive_threshold and len(self.rms_history) > 5:
            # Tính threshold dựa trên history
            mean_rms = np.mean(self.rms_history)
            std_rms = np.std(self.rms_history)
            threshold = mean_rms + 1.5 * std_rms  # 1.5 sigma
        
        # 8. Check stability
        is_stable = rms_distance < threshold
        
        # 9. Calculate stability score (0-1, cao hơn = ổn định hơn)
        # Inverse relationship: RMS càng thấp → score càng cao
        max_rms = threshold * 3  # Giả định max RMS = 3x threshold
        stability_score = max(0.0, min(1.0, 1.0 - (rms_distance / max_rms)))
        
        # Lưu RMS vào history
        self.rms_history.append(rms_distance)
        
        return {
            'is_stable': is_stable,
            'rms_distance': rms_distance,
            'variance': total_variance,
            'stability_score': stability_score,
            'threshold': threshold,
            'details': {
                'window_size': len(positions_x),
                'mean_iod': np.mean(interocular_dists) if len(interocular_dists) > 0 else 0,
                'head_compensation': self.use_head_compensation and len(head_poses) == len(window_data),
                'outliers_removed': self.use_outlier_removal,
                'smoothed': self.use_smoothing
            }
        }

