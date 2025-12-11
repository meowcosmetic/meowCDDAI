"""
Fatigue Detection - Phát hiện mệt mỏi từ mắt và đầu
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class FatigueDetector:
    """
    Detect fatigue/tiredness từ:
    - Eye closure (PERCLOS - Percentage of Eye Closure)
    - Blink frequency
    - Head nodding
    - Yawning
    - Eye Aspect Ratio (EAR)
    """
    
    def __init__(self, fps: float = 30.0):
        """
        Args:
            fps: Frames per second
        """
        self.fps = fps
        
        # Eye Aspect Ratio (EAR) tracking
        self.ear_history = deque(maxlen=int(fps * 2))  # 2 seconds history
        self.ear_threshold = 0.25  # Threshold để coi là nhắm mắt
        self.blink_threshold = 0.2  # Threshold để detect blink
        
        # Blink detection
        self.blink_count = 0
        self.blink_times = deque(maxlen=100)  # Store blink timestamps
        self.eye_closed_frames = 0
        self.total_frames = 0
        
        # Head nod detection
        self.head_pitch_history = deque(maxlen=int(fps * 2))
        self.head_nod_count = 0
        self.head_nod_times = deque(maxlen=100)
        
        # Yawn detection
        self.mouth_aspect_ratio_history = deque(maxlen=int(fps * 1))
        self.yawn_count = 0
        self.yawn_threshold = 0.5  # MAR threshold cho yawn
        
        # State tracking
        self.eyes_closed = False
        self.consecutive_closed_frames = 0
        
    def calculate_eye_aspect_ratio(self, face_landmarks) -> Optional[float]:
        """
        Tính Eye Aspect Ratio (EAR) từ MediaPipe landmarks
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p1-p6 là các điểm trên mắt
        
        Args:
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            EAR value (thấp hơn = mắt nhắm hơn)
        """
        try:
            # Left eye landmarks (MediaPipe Face Mesh)
            # Top: 159, Bottom: 145, Left: 33, Right: 133
            # Inner: 133, Outer: 33
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            left_eye_left = face_landmarks.landmark[33]
            left_eye_right = face_landmarks.landmark[133]
            
            # Right eye landmarks
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            right_eye_left = face_landmarks.landmark[362]
            right_eye_right = face_landmarks.landmark[263]
            
            # Calculate distances
            def euclidean_distance(p1, p2):
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            # Left eye EAR
            left_vertical_1 = euclidean_distance(left_eye_top, left_eye_bottom)
            left_vertical_2 = euclidean_distance(left_eye_left, left_eye_right)
            left_horizontal = euclidean_distance(left_eye_left, left_eye_right)
            
            if left_horizontal == 0:
                return None
            
            left_ear = (left_vertical_1 + left_vertical_2) / (2.0 * left_horizontal)
            
            # Right eye EAR
            right_vertical_1 = euclidean_distance(right_eye_top, right_eye_bottom)
            right_vertical_2 = euclidean_distance(right_eye_left, right_eye_right)
            right_horizontal = euclidean_distance(right_eye_left, right_eye_right)
            
            if right_horizontal == 0:
                return None
            
            right_ear = (right_vertical_1 + right_vertical_2) / (2.0 * right_horizontal)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            return ear
        except Exception as e:
            logger.debug(f"[Fatigue] EAR calculation error: {str(e)}")
            return None
    
    def detect_blink(self, ear: float, current_time: float) -> bool:
        """
        Detect blink từ EAR
        
        Returns:
            True nếu detect được blink
        """
        if ear is None:
            return False
        
        self.ear_history.append(ear)
        self.total_frames += 1
        
        # Check if eyes are closed
        if ear < self.ear_threshold:
            if not self.eyes_closed:
                self.eyes_closed = True
                self.consecutive_closed_frames = 1
            else:
                self.consecutive_closed_frames += 1
                self.eye_closed_frames += 1
        else:
            # Eyes opened - check if was blinking
            if self.eyes_closed:
                # Was blinking, now opened
                if self.consecutive_closed_frames > 0:
                    # Valid blink (closed for at least 1 frame)
                    self.blink_count += 1
                    self.blink_times.append(current_time)
                    logger.debug(f"[Fatigue] Blink detected at {current_time:.2f}s")
                
                self.eyes_closed = False
                self.consecutive_closed_frames = 0
        
        return False  # Return False, blink is tracked internally
    
    def calculate_perclos(self) -> float:
        """
        Calculate PERCLOS (Percentage of Eye Closure)
        
        PERCLOS = (frames với eyes closed) / (total frames) * 100
        
        Returns:
            PERCLOS percentage (0-100)
        """
        if self.total_frames == 0:
            return 0.0
        
        perclos = (self.eye_closed_frames / self.total_frames) * 100
        return min(100.0, perclos)
    
    def get_blink_frequency(self, window_seconds: float = 60.0) -> float:
        """
        Tính blink frequency (blinks per minute)
        
        Args:
            window_seconds: Time window để tính frequency
        
        Returns:
            Blinks per minute
        """
        if len(self.blink_times) < 2:
            return 0.0
        
        # Get blinks trong window
        current_time = self.blink_times[-1] if self.blink_times else 0
        window_start = current_time - window_seconds
        
        recent_blinks = [t for t in self.blink_times if t >= window_start]
        
        if len(recent_blinks) < 2:
            return 0.0
        
        # Calculate frequency
        time_span = recent_blinks[-1] - recent_blinks[0]
        if time_span == 0:
            return 0.0
        
        blinks_per_second = len(recent_blinks) / time_span
        blinks_per_minute = blinks_per_second * 60
        
        return blinks_per_minute
    
    def detect_head_nod(self, head_pitch: Optional[float], current_time: float) -> bool:
        """
        Detect head nodding từ head pitch
        
        Args:
            head_pitch: Head pitch angle (radians)
            current_time: Current time
        
        Returns:
            True nếu detect được head nod
        """
        if head_pitch is None:
            return False
        
        self.head_pitch_history.append(head_pitch)
        
        if len(self.head_pitch_history) < int(self.fps * 0.5):  # Cần ít nhất 0.5s
            return False
        
        # Calculate pitch variance và pattern
        pitch_array = np.array(self.head_pitch_history)
        pitch_variance = np.var(pitch_array)
        
        # Head nod: pitch thay đổi nhiều (nodding up/down)
        # Normal: 0.01-0.05, Nodding: > 0.1
        if pitch_variance > 0.1:
            # Check if it's a nod pattern (up-down movement)
            pitch_diff = np.diff(pitch_array)
            direction_changes = np.sum(np.diff(np.sign(pitch_diff)) != 0)
            
            if direction_changes > 2:  # Multiple direction changes = nodding
                self.head_nod_count += 1
                self.head_nod_times.append(current_time)
                logger.debug(f"[Fatigue] Head nod detected at {current_time:.2f}s")
                return True
        
        return False
    
    def calculate_mouth_aspect_ratio(self, face_landmarks) -> Optional[float]:
        """
        Tính Mouth Aspect Ratio (MAR) để detect yawning
        
        Args:
            face_landmarks: MediaPipe face landmarks
        
        Returns:
            MAR value (cao hơn = miệng mở rộng hơn)
        """
        try:
            # Mouth landmarks
            # Left: 61, Right: 291, Top: 13, Bottom: 14
            mouth_left = face_landmarks.landmark[61]
            mouth_right = face_landmarks.landmark[291]
            mouth_top = face_landmarks.landmark[13]
            mouth_bottom = face_landmarks.landmark[14]
            
            # Calculate distances
            def euclidean_distance(p1, p2):
                return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            
            # MAR = mouth height / mouth width
            mouth_width = euclidean_distance(mouth_left, mouth_right)
            mouth_height = euclidean_distance(mouth_top, mouth_bottom)
            
            if mouth_width == 0:
                return None
            
            mar = mouth_height / mouth_width
            return mar
        except Exception as e:
            logger.debug(f"[Fatigue] MAR calculation error: {str(e)}")
            return None
    
    def detect_yawn(self, mar: float, current_time: float) -> bool:
        """
        Detect yawning từ Mouth Aspect Ratio
        
        Args:
            mar: Mouth Aspect Ratio
            current_time: Current time
        
        Returns:
            True nếu detect được yawn
        """
        if mar is None:
            return False
        
        self.mouth_aspect_ratio_history.append(mar)
        
        # Yawn: MAR cao và kéo dài
        if mar > self.yawn_threshold:
            # Check if sustained (yawn lasts > 0.5s)
            if len(self.mouth_aspect_ratio_history) >= int(self.fps * 0.5):
                recent_mars = list(self.mouth_aspect_ratio_history)[-int(self.fps * 0.5):]
                avg_mar = np.mean(recent_mars)
                
                if avg_mar > self.yawn_threshold:
                    self.yawn_count += 1
                    logger.debug(f"[Fatigue] Yawn detected at {current_time:.2f}s")
                    return True
        
        return False
    
    def detect_fatigue(self, 
                      face_landmarks,
                      head_pitch: Optional[float],
                      current_time: float) -> Tuple[float, str, Dict[str, Any]]:
        """
        Main function để detect fatigue
        
        Args:
            face_landmarks: MediaPipe face landmarks
            head_pitch: Head pitch angle (radians)
            current_time: Current time in seconds
        
        Returns:
            (fatigue_score, fatigue_level, indicators)
            - fatigue_score: 0-100 (cao hơn = mệt mỏi hơn)
            - fatigue_level: "low", "medium", "high"
            - indicators: dict với các chỉ số chi tiết
        """
        # 1. Calculate EAR và detect blink
        ear = self.calculate_eye_aspect_ratio(face_landmarks)
        self.detect_blink(ear, current_time)
        
        # 2. Calculate PERCLOS
        perclos = self.calculate_perclos()
        
        # 3. Calculate blink frequency
        blink_freq = self.get_blink_frequency(window_seconds=60.0)
        
        # 4. Detect head nod
        head_nod_detected = self.detect_head_nod(head_pitch, current_time)
        
        # 5. Calculate MAR và detect yawn
        mar = self.calculate_mouth_aspect_ratio(face_landmarks)
        yawn_detected = self.detect_yawn(mar, current_time)
        
        # Calculate fatigue score
        # Normal blink frequency: 15-20 blinks/min
        # Low blink frequency (< 10/min) = tired
        blink_score = 0.0
        if blink_freq > 0:
            if blink_freq < 10:  # Too low
                blink_score = 1.0 - (blink_freq / 10)
            elif blink_freq > 30:  # Too high (stress)
                blink_score = min(1.0, (blink_freq - 20) / 20)
            else:
                blink_score = 0.0  # Normal
        
        # PERCLOS: > 20% = tired
        perclos_score = min(1.0, perclos / 20.0)
        
        # Head nod: frequent nodding = tired
        head_nod_score = min(1.0, self.head_nod_count / 10.0)
        
        # Yawn: frequent yawning = tired
        yawn_score = min(1.0, self.yawn_count / 5.0)
        
        # Weighted fatigue score
        fatigue_score = (
            perclos_score * 0.4 +  # 40% weight
            blink_score * 0.3 +  # 30% weight
            head_nod_score * 0.2 +  # 20% weight
            yawn_score * 0.1  # 10% weight
        ) * 100
        
        # Classify fatigue level
        if fatigue_score < 30:
            fatigue_level = "low"
        elif fatigue_score < 60:
            fatigue_level = "medium"
        else:
            fatigue_level = "high"
        
        indicators = {
            'perclos': round(perclos, 2),
            'blink_frequency': round(blink_freq, 2),
            'blink_count': self.blink_count,
            'head_nod_count': self.head_nod_count,
            'yawn_count': self.yawn_count,
            'ear': round(ear, 3) if ear else None,
            'mar': round(mar, 3) if mar else None
        }
        
        return fatigue_score, fatigue_level, indicators
    
    def reset(self):
        """Reset detector (dùng khi bắt đầu video mới)"""
        self.ear_history.clear()
        self.blink_count = 0
        self.blink_times.clear()
        self.eye_closed_frames = 0
        self.total_frames = 0
        self.head_pitch_history.clear()
        self.head_nod_count = 0
        self.head_nod_times.clear()
        self.mouth_aspect_ratio_history.clear()
        self.yawn_count = 0
        self.eyes_closed = False
        self.consecutive_closed_frames = 0
        logger.info("[Fatigue] Detector reset")




