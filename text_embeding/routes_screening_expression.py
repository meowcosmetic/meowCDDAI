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

# Lazy import MediaPipe - chỉ import khi cần
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("[Expression] MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe")
    mp = None
    mp_face_mesh = None
    mp_drawing = None

# GPU detection
USE_GPU = Config.USE_GPU.lower() if hasattr(Config, 'USE_GPU') else "auto"
GPU_AVAILABLE = False

try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        logger.info(f"[Expression] ✅ OpenCV GPU detected")
except:
    pass

router = APIRouter(prefix="/screening/expression", tags=["Screening - Facial Expression"])

# MediaPipe Face Mesh landmarks cho expression analysis
# Mouth landmarks
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_CENTER_TOP = 12
MOUTH_CENTER_BOTTOM = 15

# Eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Eyebrow landmarks
LEFT_EYEBROW_INNER = 107
LEFT_EYEBROW_OUTER = 70
RIGHT_EYEBROW_INNER = 336
RIGHT_EYEBROW_OUTER = 300

# Nose tip
NOSE_TIP = 4


def extract_expression_features(face_landmarks, frame_shape):
    """
    Extract features từ MediaPipe landmarks để classify expression
    """
    h, w = frame_shape[:2]
    
    # Lấy tọa độ các landmarks quan trọng
    def get_landmark(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])
    
    # Mouth features
    mouth_left = get_landmark(MOUTH_LEFT)
    mouth_right = get_landmark(MOUTH_RIGHT)
    mouth_top = get_landmark(MOUTH_TOP)
    mouth_bottom = get_landmark(MOUTH_BOTTOM)
    mouth_center_top = get_landmark(MOUTH_CENTER_TOP)
    mouth_center_bottom = get_landmark(MOUTH_CENTER_BOTTOM)
    
    # Eye features
    left_eye_top = get_landmark(LEFT_EYE_TOP)
    left_eye_bottom = get_landmark(LEFT_EYE_BOTTOM)
    right_eye_top = get_landmark(RIGHT_EYE_TOP)
    right_eye_bottom = get_landmark(RIGHT_EYE_BOTTOM)
    
    # Eyebrow features
    left_eyebrow_inner = get_landmark(LEFT_EYEBROW_INNER)
    left_eyebrow_outer = get_landmark(LEFT_EYEBROW_OUTER)
    right_eyebrow_inner = get_landmark(RIGHT_EYEBROW_INNER)
    right_eyebrow_outer = get_landmark(RIGHT_EYEBROW_OUTER)
    
    # Tính các features
    features = {}
    
    # Mouth opening (khoảng cách giữa top và bottom)
    mouth_height = np.linalg.norm(mouth_center_bottom - mouth_center_top)
    mouth_width = np.linalg.norm(mouth_right - mouth_left)
    features['mouth_aspect_ratio'] = mouth_height / (mouth_width + 1e-6)
    
    # Mouth curvature (smile/frown)
    mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
    mouth_corners_y = (mouth_left[1] + mouth_right[1]) / 2
    features['mouth_curvature'] = mouth_corners_y - mouth_center_y  # Positive = smile
    
    # Eye opening
    left_eye_height = np.linalg.norm(left_eye_bottom - left_eye_top)
    right_eye_height = np.linalg.norm(right_eye_bottom - right_eye_top)
    features['eye_opening'] = (left_eye_height + right_eye_height) / 2
    
    # Eyebrow position (furrowed = angry/sad)
    left_eyebrow_height = left_eyebrow_inner[1] - left_eye_top[1]
    right_eyebrow_height = right_eyebrow_inner[1] - right_eye_top[1]
    features['eyebrow_height'] = (left_eyebrow_height + right_eyebrow_height) / 2
    
    # Eyebrow angle (raised = surprised)
    left_eyebrow_angle = np.arctan2(
        left_eyebrow_outer[1] - left_eyebrow_inner[1],
        left_eyebrow_outer[0] - left_eyebrow_inner[0]
    )
    right_eyebrow_angle = np.arctan2(
        right_eyebrow_outer[1] - right_eyebrow_inner[1],
        right_eyebrow_outer[0] - right_eyebrow_inner[0]
    )
    features['eyebrow_angle'] = (left_eyebrow_angle + right_eyebrow_angle) / 2
    
    return features


def classify_expression(features):
    """
    Classify expression dựa trên features extracted từ landmarks
    Sử dụng rule-based approach với thresholds
    """
    mouth_ratio = features.get('mouth_aspect_ratio', 0)
    mouth_curvature = features.get('mouth_curvature', 0)
    eye_opening = features.get('eye_opening', 0)
    eyebrow_height = features.get('eyebrow_height', 0)
    eyebrow_angle = features.get('eyebrow_angle', 0)
    
    # Normalize features (cần calibration với dataset thực tế)
    # Tạm thời dùng heuristic
    
    # Happy: mouth curved up, mouth open
    if mouth_curvature > 5 and mouth_ratio > 0.15:
        return "happy", 0.8
    
    # Sad: mouth curved down, eyebrows furrowed
    if mouth_curvature < -3 and eyebrow_height < -5:
        return "sad", 0.7
    
    # Angry: eyebrows furrowed, mouth tight
    if eyebrow_height < -8 and mouth_ratio < 0.1:
        return "angry", 0.7
    
    # Surprised: eyebrows raised, mouth open, eyes wide
    if eyebrow_angle > 0.3 and mouth_ratio > 0.2 and eye_opening > 15:
        return "surprised", 0.7
    
    # Fearful: eyebrows raised, mouth slightly open, eyes wide
    if eyebrow_angle > 0.2 and mouth_ratio > 0.12 and eye_opening > 12:
        return "fearful", 0.6
    
    # Disgusted: nose wrinkle (khó detect từ landmarks), mouth slightly open
    if mouth_ratio > 0.1 and eyebrow_height < -3:
        return "disgusted", 0.5
    
    # Neutral: default
    return "neutral", 0.6


def detect_expression_opencv(frame, face_cascade):
    """
    Fallback: Simple expression detection với OpenCV
    Chỉ detect neutral vì không có landmarks chi tiết
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) > 0:
        return "neutral", 0.5, faces[0]
    return None, 0.0, None


def draw_expression_annotations(frame, face_landmarks, expression, confidence, frame_count=0, fps=30):
    """
    Vẽ các annotations lên frame để hiển thị expression analysis
    """
    h, w = frame.shape[:2]
    annotated_frame = frame.copy()
    
    # Tính bounding box của face từ landmarks
    if face_landmarks:
        face_x_coords = [lm.x * w for lm in face_landmarks.landmark]
        face_y_coords = [lm.y * h for lm in face_landmarks.landmark]
        
        face_x_min = int(min(face_x_coords))
        face_y_min = int(min(face_y_coords))
        face_x_max = int(max(face_x_coords))
        face_y_max = int(max(face_y_coords))
        
        # Vẽ bounding box cho face
        face_color = (0, 255, 0)  # Xanh lá
        cv2.rectangle(annotated_frame, (face_x_min, face_y_min), 
                     (face_x_max, face_y_max), face_color, 2)
        cv2.putText(annotated_frame, "Face", (face_x_min, face_y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        # Vẽ landmarks quan trọng với màu khác nhau
        # Mouth landmarks (màu đỏ)
        mouth_indices = [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM, 
                        MOUTH_CENTER_TOP, MOUTH_CENTER_BOTTOM]
        for idx in mouth_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
        
        # Eye landmarks (màu xanh dương)
        eye_indices = [LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
                      RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT]
        for idx in eye_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_frame, (x, y), 3, (255, 0, 0), -1)
        
        # Eyebrow landmarks (màu vàng)
        eyebrow_indices = [LEFT_EYEBROW_INNER, LEFT_EYEBROW_OUTER,
                          RIGHT_EYEBROW_INNER, RIGHT_EYEBROW_OUTER]
        for idx in eyebrow_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated_frame, (x, y), 3, (0, 255, 255), -1)
        
        # Vẽ lines để highlight các vùng quan trọng
        # Mouth outline
        mouth_points = [
            (int(face_landmarks.landmark[MOUTH_LEFT].x * w), 
             int(face_landmarks.landmark[MOUTH_LEFT].y * h)),
            (int(face_landmarks.landmark[MOUTH_TOP].x * w), 
             int(face_landmarks.landmark[MOUTH_TOP].y * h)),
            (int(face_landmarks.landmark[MOUTH_RIGHT].x * w), 
             int(face_landmarks.landmark[MOUTH_RIGHT].y * h)),
            (int(face_landmarks.landmark[MOUTH_BOTTOM].x * w), 
             int(face_landmarks.landmark[MOUTH_BOTTOM].y * h)),
        ]
        cv2.polylines(annotated_frame, [np.array(mouth_points, np.int32)], 
                     True, (0, 0, 255), 2)
        
        # Left eye outline
        left_eye_points = [
            (int(face_landmarks.landmark[LEFT_EYE_LEFT].x * w), 
             int(face_landmarks.landmark[LEFT_EYE_LEFT].y * h)),
            (int(face_landmarks.landmark[LEFT_EYE_TOP].x * w), 
             int(face_landmarks.landmark[LEFT_EYE_TOP].y * h)),
            (int(face_landmarks.landmark[LEFT_EYE_RIGHT].x * w), 
             int(face_landmarks.landmark[LEFT_EYE_RIGHT].y * h)),
            (int(face_landmarks.landmark[LEFT_EYE_BOTTOM].x * w), 
             int(face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h)),
        ]
        cv2.polylines(annotated_frame, [np.array(left_eye_points, np.int32)], 
                     True, (255, 0, 0), 2)
        
        # Right eye outline
        right_eye_points = [
            (int(face_landmarks.landmark[RIGHT_EYE_LEFT].x * w), 
             int(face_landmarks.landmark[RIGHT_EYE_LEFT].y * h)),
            (int(face_landmarks.landmark[RIGHT_EYE_TOP].x * w), 
             int(face_landmarks.landmark[RIGHT_EYE_TOP].y * h)),
            (int(face_landmarks.landmark[RIGHT_EYE_RIGHT].x * w), 
             int(face_landmarks.landmark[RIGHT_EYE_RIGHT].y * h)),
            (int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].x * w), 
             int(face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h)),
        ]
        cv2.polylines(annotated_frame, [np.array(right_eye_points, np.int32)], 
                     True, (255, 0, 0), 2)
    
    # Expression color mapping
    expression_colors = {
        "happy": (0, 255, 0),      # Xanh lá
        "sad": (255, 0, 0),         # Xanh dương
        "angry": (0, 0, 255),       # Đỏ
        "neutral": (128, 128, 128), # Xám
        "surprised": (255, 255, 0), # Cyan
        "fearful": (255, 0, 255),   # Magenta
        "disgusted": (0, 165, 255)  # Orange
    }
    
    expr_color = expression_colors.get(expression, (255, 255, 255))
    
    # Vẽ status bar ở trên cùng
    status_y = 20
    cv2.rectangle(annotated_frame, (10, 5), (w - 10, 50), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Expression: {expression.upper()}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, expr_color, 2)
    cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, expr_color, 2)
    
    # Vẽ frame count và time
    if fps > 0:
        time_sec = frame_count / fps
        time_text = f"Frame: {frame_count} | Time: {time_sec:.2f}s"
        cv2.putText(annotated_frame, time_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Vẽ legend cho landmarks
    legend_y = 80
    cv2.putText(annotated_frame, "Legend:", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.circle(annotated_frame, (80, legend_y + 10), 4, (0, 0, 255), -1)
    cv2.putText(annotated_frame, "Mouth", (95, legend_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(annotated_frame, (150, legend_y + 10), 3, (255, 0, 0), -1)
    cv2.putText(annotated_frame, "Eyes", (165, legend_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(annotated_frame, (220, legend_y + 10), 3, (0, 255, 255), -1)
    cv2.putText(annotated_frame, "Eyebrows", (235, legend_y + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return annotated_frame


def draw_expression_annotations_opencv(frame, face_rect, expression, confidence, frame_count=0, fps=30):
    """
    Vẽ annotations cho OpenCV fallback mode
    """
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    
    if face_rect is not None:
        x, y, w_face, h_face = face_rect
        # Vẽ bounding box
        cv2.rectangle(annotated_frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Face", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Expression color
    expression_colors = {
        "happy": (0, 255, 0),
        "sad": (255, 0, 0),
        "angry": (0, 0, 255),
        "neutral": (128, 128, 128),
    }
    expr_color = expression_colors.get(expression, (255, 255, 255))
    
    # Status bar
    cv2.rectangle(annotated_frame, (10, 5), (w - 10, 50), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Expression: {expression.upper()}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, expr_color, 2)
    cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, expr_color, 2)
    
    # Frame count
    if fps > 0:
        time_sec = frame_count / fps
        time_text = f"Frame: {frame_count} | Time: {time_sec:.2f}s"
        cv2.putText(annotated_frame, time_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame


class ExpressionAnalysisResponse(BaseModel):
    """Response model cho phân tích biểu cảm"""
    expression_distribution: Dict[str, float] = Field(..., description="Phân bố biểu cảm (happy, sad, angry, neutral, etc.)")
    expression_diversity_score: float = Field(..., description="Điểm đa dạng biểu cảm (0-100, cao hơn = đa dạng hơn)")
    neutral_percentage: float = Field(..., description="Phần trăm thời gian biểu cảm neutral (%)")
    total_frames: int = Field(..., description="Tổng số frame đã phân tích")
    analyzed_duration: float = Field(..., description="Thời gian video đã phân tích (giây)")
    risk_score: float = Field(..., description="Điểm đánh giá rủi ro (0-100, cao hơn = rủi ro cao hơn)")


@router.post("/analyze", response_model=ExpressionAnalysisResponse)
async def analyze_expression(
    video: UploadFile = File(..., description="Video file để phân tích"),
    show_video: str = Form("true", description="Hiển thị video trong quá trình xử lý (true/false)")
):
    """
    Phân tích Facial Expression Recognition từ video
    
    - Nhận diện biểu cảm: vui, buồn, khó chịu, neutral, surprised, fearful, disgusted
    - Tính mức đa dạng biểu cảm
    - Đánh giá rủi ro (trẻ ASD thường ít biểu cảm)
    
    Args:
        video: File video (mp4, avi, mov, etc.)
        show_video: "true" để hiển thị video real-time
    
    Returns:
        ExpressionAnalysisResponse với các chỉ số phân tích
    """
    show_video_bool = show_video.lower() in ("true", "1", "yes", "on")
    
    use_fallback = not MEDIAPIPE_AVAILABLE
    if use_fallback:
        logger.warning("[Expression] MediaPipe không có, sử dụng OpenCV fallback (chỉ detect neutral)")
    
    temp_path = None
    cap = None
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
        
        expressions = {
            "happy": 0,
            "sad": 0,
            "angry": 0,
            "neutral": 0,
            "surprised": 0,
            "fearful": 0,
            "disgusted": 0
        }
        
        frame_count = 0
        face_detected_count = 0
        
        if use_fallback:
            # OpenCV fallback
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect expression với OpenCV (chỉ neutral)
                result = detect_expression_opencv(frame, face_cascade)
                expression, confidence, face_rect = result
                
                if expression:
                    expressions[expression] += 1
                    face_detected_count += 1
                
                # Visualize nếu được yêu cầu
                if show_video_bool:
                    annotated_frame = draw_expression_annotations_opencv(
                        frame, face_rect, expression or "unknown", confidence or 0.0, frame_count, fps
                    )
                    try:
                        cv2.imshow("Expression Analysis - Press 'q' to quit", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            show_video_bool = False
                    except cv2.error as e:
                        if "No display" in str(e) or "cannot connect" in str(e).lower():
                            logger.warning("[Expression] Không thể hiển thị video (headless server). Tắt video display.")
                            show_video_bool = False
                        else:
                            raise
                
                frame_count += 1
        else:
            # MediaPipe implementation
            with mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as face_mesh:
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face_detected_count += 1
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Extract features từ landmarks
                        features = extract_expression_features(face_landmarks, frame.shape)
                        
                        # Classify expression
                        expression, confidence = classify_expression(features)
                        expressions[expression] += 1
                        
                        # Visualize nếu được yêu cầu
                        if show_video_bool:
                            annotated_frame = draw_expression_annotations(
                                frame, face_landmarks, expression, confidence, frame_count, fps
                            )
                            try:
                                cv2.imshow("Expression Analysis - Press 'q' to quit", annotated_frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    show_video_bool = False
                            except cv2.error as e:
                                if "No display" in str(e) or "cannot connect" in str(e).lower():
                                    logger.warning("[Expression] Không thể hiển thị video (headless server). Tắt video display.")
                                    show_video_bool = False
                                else:
                                    raise
                    else:
                        # Không detect được face, không tính expression
                        pass
                    
                    frame_count += 1
        
        # Release và cleanup
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
                    logger.warning(f"[Expression] Không thể xóa file temp: {temp_path}")
        
        # Tính toán kết quả
        total_expressions = sum(expressions.values())
        if total_expressions > 0:
            expression_distribution = {k: (v / total_expressions * 100) for k, v in expressions.items()}
        else:
            expression_distribution = {k: 0.0 for k in expressions.keys()}
        
        # Tính expression diversity (số lượng biểu cảm khác nhau được sử dụng)
        unique_expressions = sum(1 for v in expressions.values() if v > 0)
        max_possible = len(expressions)
        expression_diversity_score = (unique_expressions / max_possible * 100) if max_possible > 0 else 0
        
        neutral_percentage = expression_distribution.get("neutral", 0.0)
        
        # Tính risk score (neutral cao + diversity thấp = risk cao)
        risk_score = max(0, min(100, (neutral_percentage * 0.6 + (100 - expression_diversity_score) * 0.4)))
        
        analyzed_duration = frame_count / fps if fps > 0 else 0
        
        return ExpressionAnalysisResponse(
            expression_distribution=expression_distribution,
            expression_diversity_score=round(expression_diversity_score, 2),
            neutral_percentage=round(neutral_percentage, 2),
            total_frames=frame_count,
            analyzed_duration=round(analyzed_duration, 2),
            risk_score=round(risk_score, 2)
        )
        
    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        logger.error(f"[Expression] Lỗi khi phân tích video: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích biểu cảm: {str(e)}"
        )

