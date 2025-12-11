from typing import Optional, List, Dict, Any
import logging
import os
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import sys

# Import config để sử dụng GPU settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

logger = logging.getLogger(__name__)

# Lazy import MediaPipe
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("[Pose] MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe")
    mp = None
    mp_pose = None
    mp_drawing = None
    mp_drawing_styles = None

# GPU detection
USE_GPU = Config.USE_GPU.lower() if hasattr(Config, 'USE_GPU') else "auto"
GPU_AVAILABLE = False

try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        logger.info(f"[Pose] ✅ OpenCV GPU detected")
except:
    pass

router = APIRouter(prefix="/screening/pose", tags=["Screening - Pose & Movement"])

# MediaPipe Pose landmarks indices
# Left arm
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
# Right arm
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
# Body
LEFT_HIP = 23
RIGHT_HIP = 24
# Legs
LEFT_KNEE = 25
LEFT_ANKLE = 27
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
# Head
NOSE = 0


def extract_pose_keypoints(pose_landmarks, frame_shape):
    """Extract keypoints từ MediaPipe pose landmarks"""
    h, w = frame_shape[:2]
    
    def get_landmark(idx):
        if pose_landmarks and idx < len(pose_landmarks.landmark):
            lm = pose_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])
        return None
    
    keypoints = {
        'left_shoulder': get_landmark(LEFT_SHOULDER),
        'left_elbow': get_landmark(LEFT_ELBOW),
        'left_wrist': get_landmark(LEFT_WRIST),
        'right_shoulder': get_landmark(RIGHT_SHOULDER),
        'right_elbow': get_landmark(RIGHT_ELBOW),
        'right_wrist': get_landmark(RIGHT_WRIST),
        'left_hip': get_landmark(LEFT_HIP),
        'right_hip': get_landmark(RIGHT_HIP),
        'left_knee': get_landmark(LEFT_KNEE),
        'left_ankle': get_landmark(LEFT_ANKLE),
        'right_knee': get_landmark(RIGHT_KNEE),
        'right_ankle': get_landmark(RIGHT_ANKLE),
        'nose': get_landmark(NOSE),
    }
    
    return keypoints


def detect_hand_flapping(keypoints_history, fps):
    """
    Detect hand flapping: tay vẫy nhanh lên xuống
    Phân tích vertical movement của wrists
    """
    if len(keypoints_history) < int(fps * 0.5):  # Cần ít nhất 0.5s
        return False
    
    # Lấy wrist positions gần đây
    recent_wrists = []
    for kp in keypoints_history[-int(fps * 0.5):]:
        if kp['left_wrist'] is not None and kp['right_wrist'] is not None:
            avg_y = (kp['left_wrist'][1] + kp['right_wrist'][1]) / 2
            recent_wrists.append(avg_y)
    
    if len(recent_wrists) < 5:
        return False
    
    # Tính variance và frequency của vertical movement
    wrist_variance = np.var(recent_wrists)
    
    # Tính số lần đổi hướng (peaks và valleys)
    direction_changes = 0
    for i in range(1, len(recent_wrists) - 1):
        if (recent_wrists[i] > recent_wrists[i-1] and recent_wrists[i] > recent_wrists[i+1]) or \
           (recent_wrists[i] < recent_wrists[i-1] and recent_wrists[i] < recent_wrists[i+1]):
            direction_changes += 1
    
    # Hand flapping: high variance + nhiều direction changes
    frequency = direction_changes / (len(recent_wrists) / fps)
    
    return wrist_variance > 500 and frequency > 2.0  # > 2 cycles per second


def detect_rocking(keypoints_history, fps):
    """
    Detect rocking: đung đưa cơ thể qua lại
    Phân tích horizontal movement của body center
    """
    if len(keypoints_history) < int(fps * 1.0):  # Cần ít nhất 1s
        return False
    
    recent_centers = []
    for kp in keypoints_history[-int(fps * 1.0):]:
        if kp['left_hip'] is not None and kp['right_hip'] is not None:
            center_x = (kp['left_hip'][0] + kp['right_hip'][0]) / 2
            recent_centers.append(center_x)
    
    if len(recent_centers) < 5:
        return False
    
    # Tính variance và pattern của horizontal movement
    center_variance = np.var(recent_centers)
    
    # Tính số lần đổi hướng
    direction_changes = 0
    for i in range(1, len(recent_centers) - 1):
        if (recent_centers[i] > recent_centers[i-1] and recent_centers[i] > recent_centers[i+1]) or \
           (recent_centers[i] < recent_centers[i-1] and recent_centers[i] < recent_centers[i+1]):
            direction_changes += 1
    
    frequency = direction_changes / (len(recent_centers) / fps)
    
    # Rocking: moderate variance + rhythmic pattern
    return center_variance > 200 and 0.5 < frequency < 3.0


def detect_toe_walking(keypoints):
    """
    Detect toe walking: đi nhón chân
    Phân tích góc giữa ankle và knee
    """
    if keypoints['left_ankle'] is None or keypoints['left_knee'] is None or \
       keypoints['right_ankle'] is None or keypoints['right_knee'] is None:
        return False
    
    # Tính góc giữa ankle và knee (vertical angle)
    left_angle = np.arctan2(
        abs(keypoints['left_ankle'][0] - keypoints['left_knee'][0]),
        abs(keypoints['left_ankle'][1] - keypoints['left_knee'][1])
    )
    right_angle = np.arctan2(
        abs(keypoints['right_ankle'][0] - keypoints['right_knee'][0]),
        abs(keypoints['right_ankle'][1] - keypoints['right_knee'][1])
    )
    
    # Toe walking: góc nhỏ (ankle cao hơn bình thường)
    avg_angle = (left_angle + right_angle) / 2
    return avg_angle < 0.3  # Góc nhỏ = nhón chân


def detect_spinning(keypoints_history, fps):
    """
    Detect spinning: quay vòng
    Phân tích rotation của body orientation
    """
    if len(keypoints_history) < int(fps * 0.5):
        return False
    
    recent_orientations = []
    for kp in keypoints_history[-int(fps * 0.5):]:
        if kp['left_shoulder'] is not None and kp['right_shoulder'] is not None and \
           kp['nose'] is not None:
            # Tính orientation từ shoulder line
            shoulder_vec = kp['right_shoulder'][:2] - kp['left_shoulder'][:2]
            orientation = np.arctan2(shoulder_vec[1], shoulder_vec[0])
            recent_orientations.append(orientation)
    
    if len(recent_orientations) < 5:
        return False
    
    # Tính tổng rotation
    total_rotation = 0
    for i in range(1, len(recent_orientations)):
        diff = recent_orientations[i] - recent_orientations[i-1]
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        total_rotation += abs(diff)
    
    # Spinning: rotation lớn trong thời gian ngắn
    rotation_rate = total_rotation / (len(recent_orientations) / fps)
    return rotation_rate > 2.0  # > 2 radians per second


def detect_hyperactivity(keypoints_history, fps):
    """
    Detect hyperactivity: di chuyển liên tục
    Phân tích tổng movement distance
    """
    if len(keypoints_history) < int(fps * 0.5):
        return False
    
    total_movement = 0
    for i in range(1, len(keypoints_history[-int(fps * 0.5):])):
        prev_kp = keypoints_history[-int(fps * 0.5) + i - 1]
        curr_kp = keypoints_history[-int(fps * 0.5) + i]
        
        if prev_kp['nose'] is not None and curr_kp['nose'] is not None:
            movement = np.linalg.norm(curr_kp['nose'][:2] - prev_kp['nose'][:2])
            total_movement += movement
    
    # Hyperactivity: movement lớn trong thời gian ngắn
    movement_rate = total_movement / (len(keypoints_history[-int(fps * 0.5):]) / fps)
    return movement_rate > 50  # pixels per second


def classify_behavior(keypoints, keypoints_history, fps):
    """
    Classify behavior từ keypoints và history
    """
    behaviors_detected = []
    
    if detect_hand_flapping(keypoints_history, fps):
        behaviors_detected.append('hand_flapping')
    
    if detect_rocking(keypoints_history, fps):
        behaviors_detected.append('rocking')
    
    if detect_toe_walking(keypoints):
        behaviors_detected.append('toe_walking')
    
    if detect_spinning(keypoints_history, fps):
        behaviors_detected.append('spinning')
    
    if detect_hyperactivity(keypoints_history, fps):
        behaviors_detected.append('hyperactivity')
    
    if not behaviors_detected:
        return 'normal', 1.0
    
    # Return primary behavior (first detected)
    return behaviors_detected[0], 0.8


def calculate_movement_intensity(keypoints_history):
    """Tính cường độ di chuyển"""
    if len(keypoints_history) < 2:
        return 0.0
    
    total_movement = 0
    for i in range(1, len(keypoints_history)):
        prev_kp = keypoints_history[i-1]
        curr_kp = keypoints_history[i]
        
        if prev_kp['nose'] is not None and curr_kp['nose'] is not None:
            movement = np.linalg.norm(curr_kp['nose'][:2] - prev_kp['nose'][:2])
            total_movement += movement
    
    avg_movement = total_movement / (len(keypoints_history) - 1) if len(keypoints_history) > 1 else 0
    return min(100, avg_movement * 2)  # Scale to 0-100


def draw_pose_annotations(frame, pose_landmarks, detected_behavior, frame_count=0, fps=30):
    """Vẽ pose skeleton và annotations"""
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    
    if pose_landmarks:
        # Vẽ pose landmarks và connections với màu sắc rõ ràng
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style()
        )
    
    # Behavior color mapping
    behavior_colors = {
        'hand_flapping': (0, 0, 255),      # Đỏ
        'rocking': (255, 0, 0),            # Xanh dương
        'toe_walking': (0, 255, 255),      # Vàng
        'spinning': (255, 0, 255),         # Magenta
        'hyperactivity': (0, 165, 255),    # Orange
        'normal': (0, 255, 0)              # Xanh lá
    }
    
    behavior_color = behavior_colors.get(detected_behavior, (255, 255, 255))
    
    # Status bar với background đậm hơn
    cv2.rectangle(annotated_frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Behavior: {detected_behavior.upper().replace('_', ' ')}", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, behavior_color, 2)
    
    # Thêm thông tin pose detection
    if pose_landmarks:
        cv2.putText(annotated_frame, "Pose: DETECTED", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(annotated_frame, "Pose: NOT DETECTED", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Frame count và time ở dưới
    if fps > 0:
        time_sec = frame_count / fps
        time_text = f"Frame: {frame_count} | Time: {time_sec:.2f}s | FPS: {fps:.1f}"
        cv2.rectangle(annotated_frame, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated_frame, time_text, (10, h - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame


class PoseAnalysisResponse(BaseModel):
    """Response model cho phân tích pose và movement"""
    detected_behaviors: Dict[str, float] = Field(..., description="Các hành vi được phát hiện và tần suất (%)")
    activity_score: float = Field(..., description="Điểm hoạt động (0-100, cao hơn = hoạt động nhiều hơn)")
    movement_intensity: float = Field(..., description="Cường độ di chuyển (0-100)")
    total_frames: int = Field(..., description="Tổng số frame đã phân tích")
    analyzed_duration: float = Field(..., description="Thời gian video đã phân tích (giây)")
    risk_score: float = Field(..., description="Điểm đánh giá rủi ro (0-100, cao hơn = rủi ro cao hơn)")


@router.post("/analyze", response_model=PoseAnalysisResponse)
async def analyze_pose(
    video: UploadFile = File(..., description="Video file để phân tích"),
    show_video: str = Form("true", description="Hiển thị video trong quá trình xử lý (true/false)")
):
    """
    Phân tích Pose & Movement từ video
    
    - Nhận diện các hành vi: hand flapping, rocking, đi nhón chân, quay vòng, hyperactivity
    - Tính activity score (hữu ích cho ADHD screening)
    - Đánh giá cường độ di chuyển
    
    Args:
        video: File video (mp4, avi, mov, etc.)
    
    Returns:
        PoseAnalysisResponse với các chỉ số phân tích
    """
    temp_path = None
    try:
        # Lưu file tạm
        temp_path = f"temp_{video.filename}"
        with open(temp_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Đọc video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể đọc video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        show_video_bool = show_video.lower() in ("true", "1", "yes", "on")
        
        use_fallback = not MEDIAPIPE_AVAILABLE
        if use_fallback:
            logger.warning("[Pose] MediaPipe không có, sử dụng fallback đơn giản")
        
        behaviors = {
            "hand_flapping": 0,
            "rocking": 0,
            "toe_walking": 0,
            "spinning": 0,
            "hyperactivity": 0,
            "normal": 0
        }
        
        keypoints_history = []  # Lưu lịch sử keypoints để detect behaviors
        movement_sum = 0.0
        
        frame_count = 0
        
        if use_fallback:
            # Fallback: chỉ detect normal behavior
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                behaviors["normal"] += 1
                
                # Visualize ngay cả trong fallback mode
                if show_video_bool:
                    annotated_frame = draw_pose_annotations(
                        frame, None, "normal", frame_count, fps
                    )
                    try:
                        # Resize nếu frame quá lớn
                        h, w = frame.shape[:2]
                        display_frame = annotated_frame.copy()
                        max_width = 1280
                        if w > max_width:
                            scale = max_width / w
                            new_width = max_width
                            new_height = int(h * scale)
                            display_frame = cv2.resize(display_frame, (new_width, new_height))
                        
                        cv2.imshow("Pose Analysis - Press 'q' to quit", display_frame)
                        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                        if key == ord('q'):
                            logger.info("[Pose] User pressed 'q', stopping video display")
                            show_video_bool = False
                    except cv2.error as e:
                        if "No display" in str(e) or "cannot connect" in str(e).lower():
                            logger.warning("[Pose] Không thể hiển thị video (headless server).")
                            show_video_bool = False
                        else:
                            raise
                    except Exception as e:
                        logger.warning(f"[Pose] Lỗi khi hiển thị video: {str(e)}")
                
                frame_count += 1
        else:
            # MediaPipe Pose implementation
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)
                    
                    # Visualize mọi frame (kể cả khi không detect được pose)
                    if show_video_bool:
                        behavior = "normal"
                        pose_landmarks = None
                        
                        if results.pose_landmarks:
                            # Extract keypoints
                            keypoints = extract_pose_keypoints(results.pose_landmarks, frame.shape)
                            keypoints_history.append(keypoints)
                            
                            # Giữ history trong 2 giây
                            max_history = int(fps * 2)
                            if len(keypoints_history) > max_history:
                                keypoints_history.pop(0)
                            
                            # Classify behavior
                            behavior, confidence = classify_behavior(keypoints, keypoints_history, fps)
                            behaviors[behavior] += 1
                            
                            # Tính movement
                            if len(keypoints_history) > 1:
                                prev_kp = keypoints_history[-2]
                                curr_kp = keypoints_history[-1]
                                if prev_kp['nose'] is not None and curr_kp['nose'] is not None:
                                    movement = np.linalg.norm(curr_kp['nose'][:2] - prev_kp['nose'][:2])
                                    movement_sum += movement
                            
                            pose_landmarks = results.pose_landmarks
                        else:
                            # Không detect được pose
                            behaviors["normal"] += 1
                        
                        # Vẽ annotations và hiển thị
                        annotated_frame = draw_pose_annotations(
                            frame, pose_landmarks, behavior, frame_count, fps
                        )
                        try:
                            # Resize nếu frame quá lớn để dễ nhìn
                            display_frame = annotated_frame.copy()
                            max_width = 1280
                            if w > max_width:
                                scale = max_width / w
                                new_width = max_width
                                new_height = int(h * scale)
                                display_frame = cv2.resize(display_frame, (new_width, new_height))
                            
                            cv2.imshow("Pose Analysis - Press 'q' to quit", display_frame)
                            # Đợi ít nhất 1ms để window update
                            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
                            if key == ord('q'):
                                logger.info("[Pose] User pressed 'q', stopping video display")
                                show_video_bool = False
                        except cv2.error as e:
                            if "No display" in str(e) or "cannot connect" in str(e).lower():
                                logger.warning("[Pose] Không thể hiển thị video (headless server).")
                                show_video_bool = False
                            else:
                                raise
                        except Exception as e:
                            logger.warning(f"[Pose] Lỗi khi hiển thị video: {str(e)}")
                            # Tiếp tục xử lý dù có lỗi hiển thị
                    else:
                        # Không hiển thị video nhưng vẫn xử lý
                        if results.pose_landmarks:
                            keypoints = extract_pose_keypoints(results.pose_landmarks, frame.shape)
                            keypoints_history.append(keypoints)
                            
                            max_history = int(fps * 2)
                            if len(keypoints_history) > max_history:
                                keypoints_history.pop(0)
                            
                            behavior, confidence = classify_behavior(keypoints, keypoints_history, fps)
                            behaviors[behavior] += 1
                            
                            if len(keypoints_history) > 1:
                                prev_kp = keypoints_history[-2]
                                curr_kp = keypoints_history[-1]
                                if prev_kp['nose'] is not None and curr_kp['nose'] is not None:
                                    movement = np.linalg.norm(curr_kp['nose'][:2] - prev_kp['nose'][:2])
                                    movement_sum += movement
                        else:
                            behaviors["normal"] += 1
                    
                    frame_count += 1
        
        # Cleanup
        if cap:
            cap.release()
        
        if show_video_bool:
            cv2.destroyAllWindows()
        
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                import time
                time.sleep(0.1)
                try:
                    os.remove(temp_path)
                except:
                    logger.warning(f"[Pose] Không thể xóa file temp: {temp_path}")
        
        # Tính toán kết quả
        total_behaviors = sum(behaviors.values())
        if total_behaviors > 0:
            detected_behaviors = {k: (v / total_behaviors * 100) for k, v in behaviors.items()}
        else:
            detected_behaviors = {k: 0.0 for k in behaviors.keys()}
        
        # Tính activity score (dựa trên hyperactivity và các hành vi bất thường)
        abnormal_behaviors = (
            detected_behaviors.get("hand_flapping", 0) +
            detected_behaviors.get("rocking", 0) +
            detected_behaviors.get("toe_walking", 0) +
            detected_behaviors.get("spinning", 0) +
            detected_behaviors.get("hyperactivity", 0)
        )
        activity_score = min(100, abnormal_behaviors * 2)  # Scale to 0-100
        
        # Tính movement intensity từ keypoints history
        movement_intensity = calculate_movement_intensity(keypoints_history) if keypoints_history else 0
        
        # Tính risk score (nhiều hành vi bất thường = risk cao)
        risk_score = min(100, abnormal_behaviors * 1.5)
        
        analyzed_duration = frame_count / fps if fps > 0 else 0
        
        return PoseAnalysisResponse(
            detected_behaviors=detected_behaviors,
            activity_score=round(activity_score, 2),
            movement_intensity=round(movement_intensity, 2),
            total_frames=frame_count,
            analyzed_duration=round(analyzed_duration, 2),
            risk_score=round(risk_score, 2)
        )
        
    except HTTPException:
        # Cleanup
        if 'cap' in locals() and cap:
            try:
                cap.release()
            except:
                pass
        if show_video_bool:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise
    except Exception as e:
        logger.error(f"[Pose] Lỗi khi phân tích video: {str(e)}")
        # Cleanup
        if 'cap' in locals() and cap:
            try:
                cap.release()
            except:
                pass
        if 'show_video_bool' in locals() and show_video_bool:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích pose: {str(e)}"
        )

