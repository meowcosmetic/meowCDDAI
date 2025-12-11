"""
Focus Level Calculator - Tính "focus level" dựa trên mắt + đầu
"""
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class FocusLevelCalculator:
    """
    Tính "focus level" dựa trên mắt + đầu
    
    Focus level = f(gaze_stability, head_stability, gaze_head_alignment, convergence)
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: Frames per second
        """
        self.fps = fps
        
        # History tracking
        self.gaze_history = deque(maxlen=int(fps * 2))  # 2 seconds
        self.head_pose_history = deque(maxlen=int(fps * 2))
        self.eye_landmarks_history = deque(maxlen=int(fps * 1))  # 1 second
        
    def _calculate_alignment(self, 
                            gaze_direction: Tuple[float, float],
                            head_pose: Tuple[float, float, float]) -> float:
        """
        Tính alignment giữa gaze direction và head direction
        
        Args:
            gaze_direction: (x, y) gaze vector (normalized)
            head_pose: (yaw, pitch, roll) head rotation (radians)
        
        Returns:
            Alignment score (0-1, cao hơn = aligned hơn)
        """
        try:
            yaw, pitch, roll = head_pose
            
            # Convert head pose to direction vector
            # Yaw: left-right rotation
            # Pitch: up-down rotation
            head_direction_x = np.sin(yaw) * np.cos(pitch)
            head_direction_y = np.sin(pitch)
            
            # Normalize
            head_magnitude = np.sqrt(head_direction_x**2 + head_direction_y**2)
            if head_magnitude > 0:
                head_direction_x /= head_magnitude
                head_direction_y /= head_magnitude
            
            # Gaze direction (already normalized)
            gaze_x, gaze_y = gaze_direction
            
            # Calculate cosine similarity
            dot_product = gaze_x * head_direction_x + gaze_y * head_direction_y
            alignment_score = (dot_product + 1) / 2  # Normalize to 0-1
            
            return max(0.0, min(1.0, alignment_score))
        except Exception as e:
            logger.debug(f"[FocusLevel] Alignment calculation error: {str(e)}")
            return 0.5  # Default neutral
    
    def _calculate_convergence(self, face_landmarks) -> float:
        """
        Tính convergence của 2 mắt (có đang nhìn vào cùng 1 điểm không)
        
        Args:
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            Convergence score (0-1, cao hơn = converge tốt hơn)
        """
        try:
            # Left eye center (iris)
            left_eye_center = face_landmarks.landmark[468]  # Left iris center
            # Right eye center (iris)
            right_eye_center = face_landmarks.landmark[473]  # Right iris center
            
            # Calculate eye directions (simplified - using iris position relative to eye corners)
            # Left eye corners
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]
            
            # Right eye corners
            right_eye_left = face_landmarks.landmark[362]
            right_eye_right = face_landmarks.landmark[263]
            
            # Calculate iris offset relative to eye center
            def calculate_offset(iris, corner_left, corner_right):
                eye_center_x = (corner_left.x + corner_right.x) / 2
                eye_center_y = (corner_left.y + corner_right.y) / 2
                offset_x = iris.x - eye_center_x
                offset_y = iris.y - eye_center_y
                return offset_x, offset_y
            
            left_offset_x, left_offset_y = calculate_offset(
                left_eye_center, left_eye_left, left_eye_right
            )
            right_offset_x, right_offset_y = calculate_offset(
                right_eye_center, right_eye_left, right_eye_right
            )
            
            # Convergence: 2 mắt nhìn cùng hướng
            # Tính similarity giữa left và right eye offsets
            left_vector = np.array([left_offset_x, left_offset_y])
            right_vector = np.array([right_offset_x, right_offset_y])
            
            # Normalize
            left_norm = np.linalg.norm(left_vector)
            right_norm = np.linalg.norm(right_vector)
            
            if left_norm == 0 or right_norm == 0:
                return 0.5  # Default neutral
            
            left_normalized = left_vector / left_norm
            right_normalized = right_vector / right_norm
            
            # Cosine similarity
            dot_product = np.dot(left_normalized, right_normalized)
            convergence_score = (dot_product + 1) / 2  # Normalize to 0-1
            
            return max(0.0, min(1.0, convergence_score))
        except Exception as e:
            logger.debug(f"[FocusLevel] Convergence calculation error: {str(e)}")
            return 0.5  # Default neutral
    
    def calculate_focus_level(self,
                            gaze_direction: Tuple[float, float],
                            head_pose: Tuple[float, float, float],
                            gaze_stability: float,
                            head_stability: float,
                            face_landmarks=None) -> Tuple[float, Dict[str, Any]]:
        """
        Tính "focus level" dựa trên mắt + đầu
        
        Args:
            gaze_direction: (x, y) gaze vector (normalized)
            head_pose: (yaw, pitch, roll) head rotation (radians)
            gaze_stability: Variance của gaze (thấp = stable)
            head_stability: Variance của head pose (thấp = stable)
            face_landmarks: MediaPipe face landmarks (optional, for convergence)
        
        Returns:
            (focus_level, details)
            - focus_level: 0-100 (cao hơn = focus tốt hơn)
            - details: dict với các chỉ số chi tiết
        """
        # Update history
        self.gaze_history.append(gaze_direction)
        self.head_pose_history.append(head_pose)
        if face_landmarks:
            self.eye_landmarks_history.append(face_landmarks)
        
        # 1. Gaze-Head Alignment (30%)
        alignment_score = self._calculate_alignment(gaze_direction, head_pose)
        
        # 2. Gaze Stability (30%)
        # gaze_stability là variance, convert sang stability score
        stability_score = 1.0 - min(gaze_stability, 1.0)
        
        # 3. Head Stability (20%)
        head_stability_score = 1.0 - min(head_stability, 1.0)
        
        # 4. Convergence (20%) - nếu có face landmarks
        convergence_score = 0.5  # Default neutral
        if face_landmarks and len(self.eye_landmarks_history) > 0:
            convergence_score = self._calculate_convergence(face_landmarks)
        
        # Weighted focus level
        focus_level = (
            alignment_score * 0.3 +
            stability_score * 0.3 +
            head_stability_score * 0.2 +
            convergence_score * 0.2
        ) * 100
        
        details = {
            'alignment_score': round(alignment_score, 3),
            'gaze_stability_score': round(stability_score, 3),
            'head_stability_score': round(head_stability_score, 3),
            'convergence_score': round(convergence_score, 3),
            'gaze_stability_variance': round(gaze_stability, 4),
            'head_stability_variance': round(head_stability, 4)
        }
        
        return focus_level, details
    
    def reset(self):
        """Reset calculator (dùng khi bắt đầu video mới)"""
        self.gaze_history.clear()
        self.head_pose_history.clear()
        self.eye_landmarks_history.clear()
        logger.info("[FocusLevel] Calculator reset")

