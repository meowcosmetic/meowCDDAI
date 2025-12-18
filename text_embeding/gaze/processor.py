"""
Gaze Analysis Processor
Hàm chính để xử lý gaze analysis từ VideoCapture
"""
import logging
import time
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from fastapi import HTTPException
from .models import GazeAnalysisResponse
from .helpers import is_looking_at_object
from .config import (
    GAZE_TRACKING_MODULES_AVAILABLE, MEDIAPIPE_AVAILABLE, mp_face_mesh,
    USE_GPU, GPU_AVAILABLE, draw_annotations,
    GazeConfig, GPUManager, GazeEstimator3D,
    GazeWanderingDetector, FocusTimeline,
    FatigueDetector, FocusLevelCalculator,
    ImprovedGazeStabilityCalculator, calculate_interocular_distance,
    ObjectDetector
)

logger = logging.getLogger(__name__)

# MediaPipe landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CENTER = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CENTER = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def _calculate_gaze_direction(gaze_x: float, gaze_y: float, threshold: float = 0.12) -> str:
    """
    Xác định hướng nhìn từ gaze offset
    
    Args:
        gaze_x: Gaze offset X (normalized -1 to 1)
        gaze_y: Gaze offset Y (normalized -1 to 1)
        threshold: Threshold để phân loại (default 0.1)
    
    Returns:
        str: "left", "right", "up", "down", "center"
    """
    abs_x = abs(gaze_x)
    abs_y = abs(gaze_y)
    
    if abs_x < threshold and abs_y < threshold:
        return "center"
    elif abs_x > abs_y:
        return "left" if gaze_x < 0 else "right"
    else:
        return "up" if gaze_y < 0 else "down"


def _estimate_gaze_from_landmarks(face_landmarks, w: int, h: int) -> Tuple[float, float]:
    """
    Estimate gaze position từ MediaPipe face landmarks
    
    Returns:
        (gaze_x, gaze_y): Normalized gaze offset (-1 to 1)
    """
    try:
        landmark = face_landmarks.landmark

        # Dùng pupil landmarks (refine_landmarks=True) và các điểm ổn định (corners + top/bottom)
        # để tránh trường hợp mí mắt sụp làm "tâm mắt" bị kéo lệch => đảo dấu khi nhìn lên/xuống.
        LEFT_PUPIL = 468
        RIGHT_PUPIL = 473
        # Eye corners
        L_CORNER_OUT = 33
        L_CORNER_IN = 133
        R_CORNER_OUT = 362
        R_CORNER_IN = 263
        # Eyelid top/bottom (khá ổn định cho tỉ lệ)
        L_TOP = 159
        L_BOTTOM = 145
        R_TOP = 386
        R_BOTTOM = 374

        # Left eye center (from corners + lids)
        left_eye_center_x = (landmark[L_CORNER_OUT].x + landmark[L_CORNER_IN].x) / 2
        left_eye_center_y = (landmark[L_TOP].y + landmark[L_BOTTOM].y) / 2
        left_eye_w = max(1e-6, abs(landmark[L_CORNER_IN].x - landmark[L_CORNER_OUT].x))
        left_eye_h = max(1e-6, abs(landmark[L_BOTTOM].y - landmark[L_TOP].y))

        # Right eye center
        right_eye_center_x = (landmark[R_CORNER_OUT].x + landmark[R_CORNER_IN].x) / 2
        right_eye_center_y = (landmark[R_TOP].y + landmark[R_BOTTOM].y) / 2
        right_eye_w = max(1e-6, abs(landmark[R_CORNER_IN].x - landmark[R_CORNER_OUT].x))
        right_eye_h = max(1e-6, abs(landmark[R_BOTTOM].y - landmark[R_TOP].y))

        # Pupil positions
        left_pupil_x = landmark[LEFT_PUPIL].x
        left_pupil_y = landmark[LEFT_PUPIL].y
        right_pupil_x = landmark[RIGHT_PUPIL].x
        right_pupil_y = landmark[RIGHT_PUPIL].y

        # Normalized gaze offsets within eye box
        left_gaze_x = (left_pupil_x - left_eye_center_x) / left_eye_w
        left_gaze_y = (left_pupil_y - left_eye_center_y) / left_eye_h
        right_gaze_x = (right_pupil_x - right_eye_center_x) / right_eye_w
        right_gaze_y = (right_pupil_y - right_eye_center_y) / right_eye_h

        gaze_x = float(np.clip((left_gaze_x + right_gaze_x) / 2, -1.0, 1.0))
        gaze_y = float(np.clip((left_gaze_y + right_gaze_y) / 2, -1.0, 1.0))
        
        return gaze_x, gaze_y
    except Exception as e:
        logger.warning(f"[Gaze] Error estimating gaze: {str(e)}")
        return 0.0, 0.0


def process_gaze_analysis(
    cap: cv2.VideoCapture,
    target_type: str = "camera",
    show_video: bool = False,
    max_duration: float = 0.0,  # 0 = không giới hạn
    is_camera: bool = False  # True nếu là camera, False nếu là video file
) -> GazeAnalysisResponse:
    """
    Hàm helper chung để xử lý gaze analysis từ VideoCapture (có thể là file hoặc camera)
    
    Args:
        cap: cv2.VideoCapture object (từ file hoặc camera)
        target_type: Loại đối tượng nhìn vào
        show_video: Có hiển thị video không
        max_duration: Thời gian tối đa (giây), 0 = không giới hạn
        is_camera: True nếu là camera, False nếu là video file
    
    Returns:
        GazeAnalysisResponse với các chỉ số phân tích
    """
    if not MEDIAPIPE_AVAILABLE or mp_face_mesh is None:
        raise HTTPException(
            status_code=500,
            detail="MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe"
        )
    
    # Initialize tracking modules
    config = GazeConfig() if GAZE_TRACKING_MODULES_AVAILABLE else None
    gpu_manager = None
    gaze_estimator_3d = None
    focus_timeline = None
    wandering_detector = None
    fatigue_detector = None
    focus_level_calc = None
    stability_calc = None
    object_detector = None
    
    # Get FPS trước khi khởi tạo các detector (cần cho FatigueDetector và FocusLevelCalculator)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default FPS
    
    # Giới hạn kích thước màn hình để tăng tốc độ xử lý
    MAX_DISPLAY_WIDTH = 1280
    MAX_DISPLAY_HEIGHT = 720
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Tính scale để fit vào max size
    scale_w = MAX_DISPLAY_WIDTH / original_width if original_width > MAX_DISPLAY_WIDTH else 1.0
    scale_h = MAX_DISPLAY_HEIGHT / original_height if original_height > MAX_DISPLAY_HEIGHT else 1.0
    scale = min(scale_w, scale_h, 1.0)  # Không scale up, chỉ scale down
    
    display_width = int(original_width * scale)
    display_height = int(original_height * scale)
    
    logger.info(f"[Gaze] Frame size: {original_width}x{original_height} -> {display_width}x{display_height} (scale: {scale:.2f})")
    
    if GAZE_TRACKING_MODULES_AVAILABLE and config:
        try:
            # Initialize GPU Manager
            gpu_manager = GPUManager()
            
            # Get frame size for 3D gaze estimator
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            gaze_estimator_3d = GazeEstimator3D(image_width=frame_width, image_height=frame_height)
            focus_timeline = FocusTimeline(
                stability_threshold=config.GAZE_STABILITY_THRESHOLD,
                min_focus_duration=config.MIN_FOCUSING_DURATION
            )
            wandering_detector = GazeWanderingDetector(config)
            # FatigueDetector và FocusLevelCalculator nhận fps, không phải config
            fatigue_detector = FatigueDetector(fps=fps)
            focus_level_calc = FocusLevelCalculator(fps=fps)
            stability_calc = ImprovedGazeStabilityCalculator(config)
            
            # Initialize ObjectDetector với config và gpu_manager
            try:
                object_detector = ObjectDetector(config, gpu_manager)
                if object_detector and hasattr(object_detector, 'oid_detector') and object_detector.oid_detector:
                    logger.info("[Gaze] ✅ ObjectDetector đã được khởi tạo thành công")
                else:
                    logger.warning("[Gaze] ⚠️  ObjectDetector không available (có thể thiếu ultralytics)")
            except Exception as e:
                logger.warning(f"[Gaze] Không thể khởi tạo ObjectDetector: {str(e)}")
                object_detector = None
            
            logger.info("[Gaze] Đã khởi tạo các module gaze tracking")
        except Exception as e:
            logger.warning(f"[Gaze] Không thể khởi tạo một số module: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Statistics tracking
    frame_count = 0
    face_detected_count = 0
    child_face_detected_count = 0
    gaze_directions: Dict[str, int] = {"left": 0, "right": 0, "center": 0, "up": 0, "down": 0}
    attention_to_person_frames = 0
    attention_to_objects_frames = 0
    attention_to_book_frames = 0
    focusing_frames = 0
    
    detected_objects: List[Dict[str, Any]] = []
    detected_books: List[Dict[str, Any]] = []
    object_interaction_events: List[Dict[str, Any]] = []
    
    # FPS đã được lấy ở trên khi khởi tạo detectors
    
    start_time = time.time()
    last_frame_time = start_time
    
    # Video display controls (chỉ áp dụng khi show_video=True)
    paused = False
    last_display_frame: Optional[np.ndarray] = None
    
    def _overlay_pause_hint(img: np.ndarray) -> np.ndarray:
        """Vẽ hint PAUSED lên frame (in-place)"""
        try:
            h_img, w_img = img.shape[:2]
            text = "PAUSED - nhan 'p' hoac Space de tiep tuc | 'q'/ESC de thoat"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = max(10, (w_img - tw) // 2)
            y = max(30, th + 20)
            # nền tối để dễ đọc
            cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
            cv2.putText(img, text, (x, y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
        except Exception:
            pass
        return img
    
    # Initialize MediaPipe Face Mesh
    # Các tham số để cải thiện độ chính xác gaze estimation:
    # - refine_landmarks=True: Bật refinement landmarks (bao gồm iris landmarks) - QUAN TRỌNG cho gaze
    # - min_detection_confidence: Ngưỡng phát hiện khuôn mặt (0.0-1.0), cao hơn = chính xác hơn nhưng có thể miss một số frame
    # - min_tracking_confidence: Ngưỡng tracking (0.0-1.0), cao hơn = ổn định hơn nhưng có thể mất track khi chuyển động nhanh
    # - static_image_mode=False: Video mode (tối ưu cho real-time)
    # - max_num_faces=2: Tối đa 2 khuôn mặt (child + adult)
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,  # ✅ QUAN TRỌNG: Bật để có iris landmarks chính xác
        min_detection_confidence=0.6,  # Tăng từ 0.5 → 0.6 để chính xác hơn
        min_tracking_confidence=0.6  # Tăng từ 0.5 → 0.6 để ổn định hơn
    ) as face_mesh:
        
        while cap.isOpened():
            # Nếu đang pause và có video window, chỉ hiển thị lại frame cũ + bắt phím
            if show_video and paused and last_display_frame is not None:
                try:
                    paused_frame = _overlay_pause_hint(last_display_frame.copy())
                    cv2.imshow('Gaze Analysis', paused_frame)
                    key = cv2.waitKey(30) & 0xFF
                    # Toggle pause
                    if key == ord('p') or key == 32:  # 32 = Space
                        paused = False
                    # Quit
                    elif key == ord('q') or key == 27:
                        logger.info("[Gaze] Người dùng nhấn 'q' hoặc ESC để dừng")
                        break
                except Exception:
                    # Nếu có lỗi UI, thoát pause và tiếp tục xử lý bình thường
                    paused = False
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check max duration
            if max_duration > 0 and elapsed_time >= max_duration:
                logger.info(f"[Gaze] Đã đạt max_duration ({max_duration}s)")
                break
            
            # Key press check sẽ được thực hiện sau khi hiển thị frame
            
            # Resize frame nếu cần để giới hạn kích thước màn hình
            if scale < 1.0:
                frame_resized = cv2.resize(frame, (display_width, display_height))
            else:
                frame_resized = frame.copy()
            
            # Convert BGR to RGB cho MediaPipe
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Sử dụng kích thước đã resize cho processing
            h, w = frame_resized.shape[:2]
            child_face_landmarks = None
            adult_face_landmarks = None
            child_face_bbox = None
            adult_face_bbox = None
            gaze_x = 0.0
            gaze_y = 0.0
            gaze_dir = "center"
            is_looking_at_adult = False
            is_looking_at_obj = False
            head_pose = None
            variance = None
            rms_distance = None
            is_stable = False
            
            # Object detection - chỉ detect mỗi N frames để tăng tốc độ
            tracked_objects = []
            if object_detector:
                try:
                    # Sử dụng OBJECT_DETECTION_INTERVAL từ config để giảm tần suất detection
                    detection_interval = getattr(config, 'OBJECT_DETECTION_INTERVAL', 5) if config else 5
                    if frame_count % detection_interval == 0 or frame_count == 1:
                        # Detect objects mỗi N frames (mặc định mỗi 5 frames)
                        # Sử dụng frame_resized để đảm bảo tọa độ nhất quán với face detection
                        tracked_objects = object_detector.detect(
                            frame=frame_resized, 
                            prioritize_book=True,
                            frame_count=frame_count
                        )
                    else:
                        # Giữ lại tracked objects từ frame trước (tracking sẽ tự cập nhật)
                        tracked_objects = getattr(process_gaze_analysis, '_last_tracked_objects', [])
                    
                    # Lưu tracked objects để dùng cho các frame tiếp theo
                    process_gaze_analysis._last_tracked_objects = tracked_objects
                    
                    # Log số lượng objects detected (chỉ log mỗi 30 frames để tránh spam)
                    if frame_count % 30 == 0 and len(tracked_objects) > 0:
                        logger.info(f"[Gaze] Detected {len(tracked_objects)} objects at frame {frame_count}")
                    
                    # Update detected objects list (chỉ khi có detection mới)
                    if frame_count % detection_interval == 0 or frame_count == 1:
                        for obj in tracked_objects:
                            obj_id = obj.get('object_id', '')
                            class_name = obj.get('class', 'unknown')
                            if obj_id.startswith('book') or class_name == 'book':
                                detected_books.append(obj)
                            detected_objects.append(obj)
                except Exception as e:
                    logger.warning(f"[Gaze] Object detection error: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                # Log warning nếu object_detector không available (chỉ log một lần)
                if frame_count == 1:
                    logger.warning("[Gaze] ⚠️  ObjectDetector không available - không thể detect objects")
                    logger.warning("[Gaze]    Cài đặt: pip install ultralytics>=8.0.0")
            
            if results.multi_face_landmarks:
                face_detected_count += 1
                
                # Calculate bounding boxes from landmarks
                def _get_face_bbox(landmarks, w, h):
                    """Extract bounding box from MediaPipe landmarks"""
                    try:
                        x_coords = [lm.x * w for lm in landmarks.landmark]
                        y_coords = [lm.y * h for lm in landmarks.landmark]
                        x_min, x_max = int(min(x_coords)), int(max(x_coords))
                        y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        return [x_min, y_min, x_max - x_min, y_max - y_min]
                    except:
                        return None
                
                def _get_face_area(bbox):
                    """Calculate face area from bounding box"""
                    if bbox and len(bbox) >= 4:
                        return bbox[2] * bbox[3]  # width * height
                    return 0
                
                # Find child face và adult face dựa trên kích thước
                # Child face thường lớn hơn (ngồi gần camera hơn)
                faces = results.multi_face_landmarks
                if len(faces) > 0:
                    # Tính diện tích của mỗi face
                    face_areas = []
                    for face in faces:
                        bbox = _get_face_bbox(face, w, h)
                        area = _get_face_area(bbox)
                        face_areas.append((face, bbox, area))
                    
                    # Sắp xếp theo diện tích (lớn nhất trước)
                    face_areas.sort(key=lambda x: x[2], reverse=True)
                    
                    # Face lớn nhất = child face (thường ngồi gần camera hơn)
                    child_face_landmarks = face_areas[0][0]
                    child_face_bbox = face_areas[0][1]
                    
                    # Face nhỏ hơn = adult face (nếu có)
                    if len(face_areas) > 1:
                        adult_face_landmarks = face_areas[1][0]
                        adult_face_bbox = face_areas[1][1]
                    else:
                        adult_face_landmarks = None
                        adult_face_bbox = None
                    
                    # Estimate gaze (normalized by eye size)
                    gaze_x, gaze_y = _estimate_gaze_from_landmarks(child_face_landmarks, w, h)
                    gaze_dir = _calculate_gaze_direction(gaze_x, gaze_y)
                    gaze_directions[gaze_dir] += 1
                    child_face_detected_count += 1
                    
                    # Calculate absolute gaze position
                    gaze_abs_x = (gaze_x * w / 2) + (w / 2)
                    gaze_abs_y = (gaze_y * h / 2) + (h / 2)
                    gaze_abs_pos = (gaze_abs_x, gaze_abs_y)
                    
                    # Check if looking at objects (tracked_objects đã được detect ở trên)
                    for obj in tracked_objects:
                        bbox = obj.get('bbox', [])
                        if len(bbox) >= 4 and is_looking_at_object(gaze_abs_pos, bbox):
                            is_looking_at_obj = True
                            attention_to_objects_frames += 1
                            if obj.get('object_id', '').startswith('book'):
                                attention_to_book_frames += 1
                            break
                    
                    # Check if looking at adult
                    if adult_face_landmarks:
                        # Simple check: if gaze is in upper part of frame
                        if gaze_y < -0.1:  # Looking up
                            is_looking_at_adult = True
                            attention_to_person_frames += 1
                    
                    # Update tracking modules
                    if GAZE_TRACKING_MODULES_AVAILABLE:
                        try:
                            # 3D gaze estimation and head pose
                            gaze_3d_result = None
                            if gaze_estimator_3d:
                                try:
                                    # Estimate head pose
                                    success, rotation_vec, translation_vec = gaze_estimator_3d.estimate_head_pose(child_face_landmarks)
                                    if success and rotation_vec is not None:
                                        # Convert rotation vector to Euler angles
                                        rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
                                        # Extract yaw, pitch, roll
                                        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                                        pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
                                        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                                        head_pose = (yaw, pitch, roll)
                                    
                                    # 3D gaze estimation
                                    if tracked_objects:
                                        object_id, confidence = gaze_estimator_3d.estimate_3d_gaze(
                                            face_landmarks=child_face_landmarks,
                                            tracked_objects=tracked_objects
                                        )
                                        if object_id:
                                            gaze_3d_result = (object_id, confidence)
                                except Exception as e:
                                    logger.debug(f"[Gaze] 3D gaze/head pose estimation error: {str(e)}")

                            # (đã bỏ auto-calibration theo yêu cầu)
                            
                            # Calculate stability metrics
                            if stability_calc:
                                try:
                                    interocular_dist = calculate_interocular_distance(child_face_landmarks, w, h)
                                    stability_result = stability_calc.calculate_stability(
                                        gaze_x=gaze_x,
                                        gaze_y=gaze_y,
                                        interocular_distance=interocular_dist,
                                        head_pose=head_pose,
                                        fps=fps,
                                        timestamp=elapsed_time
                                    )
                                    is_stable = stability_result.get('is_stable', False)
                                    variance = stability_result.get('variance')
                                    rms_distance = stability_result.get('rms_distance')
                                except Exception as e:
                                    logger.debug(f"[Gaze] Stability calculation error: {str(e)}")
                                    is_stable = False
                            
                            # Update focus timeline
                            if focus_timeline:
                                focus_timeline.update(
                                    frame_count=frame_count,
                                    current_time=elapsed_time,
                                    gaze_pos=(gaze_abs_x, gaze_abs_y),
                                    tracked_objects=tracked_objects,
                                    fps=fps,
                                    gaze_3d_result=gaze_3d_result
                                )
                            
                            # Update wandering detector
                            if wandering_detector:
                                looking_at_obj_ratio = 1.0 if is_looking_at_obj else 0.0
                                looking_at_adult_ratio = 1.0 if is_looking_at_adult else 0.0
                                
                                # Use stability from stability_calc (already calculated above)
                                # If not calculated, default to False
                                if 'is_stable' not in locals():
                                    is_stable = False
                                
                                wandering_detector.update(
                                    frame_count=frame_count,
                                    current_time=elapsed_time,
                                    is_stable=is_stable,
                                    looking_at_object_ratio=looking_at_obj_ratio,
                                    looking_at_adult_ratio=looking_at_adult_ratio,
                                    adult_face_exists=adult_face_landmarks is not None,
                                    gaze_offset_x=gaze_x,
                                    gaze_offset_y=gaze_y,
                                    fps=fps,
                                    gaze_3d_result=gaze_3d_result
                                )
                            
                            # Update fatigue detector
                            if fatigue_detector:
                                try:
                                    interocular_dist = calculate_interocular_distance(child_face_landmarks, w, h)
                                    fatigue_detector.update(
                                        face_landmarks=child_face_landmarks,
                                        frame_shape=(h, w),
                                        interocular_distance=interocular_dist
                                    )
                                except:
                                    pass
                            
                            # Update focus level calculator
                            if focus_level_calc:
                                try:
                                    focus_level_calc.update(
                                        face_landmarks=child_face_landmarks,
                                        gaze_offset_x=gaze_x,
                                        gaze_offset_y=gaze_y,
                                        frame_shape=(h, w)
                                    )
                                except:
                                    pass
                        except Exception as e:
                            logger.warning(f"[Gaze] Error updating tracking modules: {str(e)}")
                    
                    # Check if focusing (looking at object or adult)
                    if is_looking_at_obj or is_looking_at_adult:
                        focusing_frames += 1
            
            # Draw annotations if show_video
            if show_video:
                try:
                    # Prepare face data for drawing
                    child_face_data = None
                    if child_face_bbox:
                        child_face_data = {'bbox': child_face_bbox}
                    elif child_face_landmarks:
                        # Nếu có landmarks nhưng không có bbox, tạo bbox từ landmarks
                        try:
                            h_frame, w_frame = frame_resized.shape[:2]
                            x_coords = [lm.x * w_frame for lm in child_face_landmarks.landmark]
                            y_coords = [lm.y * h_frame for lm in child_face_landmarks.landmark]
                            x_min, x_max = int(min(x_coords)), int(max(x_coords))
                            y_min, y_max = int(min(y_coords)), int(max(y_coords))
                            child_face_data = {'bbox': [x_min, y_min, x_max - x_min, y_max - y_min]}
                        except:
                            pass
                    
                    adult_face_data = None
                    if adult_face_bbox:
                        adult_face_data = {'bbox': adult_face_bbox}
                    
                    # Vẽ annotations trên frame đã resize
                    # Luôn truyền gaze_x, gaze_y và gaze_dir để đảm bảo mũi tên được vẽ
                    annotated_frame = draw_annotations(
                        frame=frame_resized.copy(),
                        child_face=child_face_data,
                        adult_face=adult_face_data,
                        gaze_dir=gaze_dir,
                        detected_objects=tracked_objects if 'tracked_objects' in locals() else [],
                        is_focusing=(is_looking_at_obj or is_looking_at_adult),
                        is_looking_at_adult=is_looking_at_adult,
                        is_looking_at_object=is_looking_at_obj,
                        frame_count=frame_count,
                        fps=int(fps),
                        gaze_x=gaze_x,
                        gaze_y=gaze_y,
                        head_pose=head_pose,
                        variance=variance,
                        rms_distance=rms_distance,
                        face_landmarks=child_face_landmarks,
                        show_landmarks=True  # Show eye landmarks
                    )
                    cv2.imshow('Gaze Analysis', annotated_frame)
                    last_display_frame = annotated_frame
                    # Wait key để hiển thị frame và check key press để dừng
                    if is_camera:
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        # Video file: delay ngắn hơn để tăng tốc độ (không cần hiển thị đúng tốc độ thực)
                        # Nếu muốn hiển thị đúng tốc độ: delay_ms = int(1000 / fps)
                        # Để tăng tốc: dùng delay nhỏ hơn
                        delay_ms = max(1, int(1000 / fps / 2)) if fps > 0 else 1  # Giảm delay một nửa để nhanh hơn
                        key = cv2.waitKey(delay_ms) & 0xFF
                    
                    # Check for 'q' hoặc ESC để dừng
                    if key == ord('q') or key == 27:
                        logger.info("[Gaze] Người dùng nhấn 'q' hoặc ESC để dừng")
                        break
                    # Toggle pause: 'p' hoặc Space
                    if key == ord('p') or key == 32:
                        paused = not paused
                except Exception as e:
                    logger.warning(f"[Gaze] Error drawing annotations: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    cv2.imshow('Gaze Analysis', frame_resized)
                    last_display_frame = frame_resized
                    # Wait key ngay cả khi có lỗi
                    if is_camera:
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        delay_ms = max(1, int(1000 / fps / 2)) if fps > 0 else 1  # Giảm delay một nửa
                        key = cv2.waitKey(delay_ms) & 0xFF
                    
                    # Check for 'q' hoặc ESC để dừng
                    if key == ord('q') or key == 27:
                        logger.info("[Gaze] Người dùng nhấn 'q' hoặc ESC để dừng")
                        break
                    # Toggle pause: 'p' hoặc Space
                    if key == ord('p') or key == 32:
                        paused = not paused
    
    # Calculate final statistics
    analyzed_duration = time.time() - start_time
    
    # Gaze direction percentages
    gaze_direction_stats = {}
    if child_face_detected_count > 0:
        for direction in gaze_directions:
            gaze_direction_stats[direction] = (gaze_directions[direction] / child_face_detected_count) * 100
    else:
        gaze_direction_stats = {d: 0.0 for d in gaze_directions.keys()}
    
    # Attention percentages
    attention_to_person_percentage = 0.0
    attention_to_objects_percentage = 0.0
    attention_to_book_percentage = 0.0
    
    if child_face_detected_count > 0:
        attention_to_person_percentage = (attention_to_person_frames / child_face_detected_count) * 100
        attention_to_objects_percentage = (attention_to_objects_frames / child_face_detected_count) * 100
        attention_to_book_percentage = (attention_to_book_frames / child_face_detected_count) * 100
    
    # Eye contact percentage (focusing percentage)
    eye_contact_percentage = 0.0
    if child_face_detected_count > 0:
        eye_contact_percentage = (focusing_frames / child_face_detected_count) * 100
    
    # Focusing duration
    focusing_duration = (focusing_frames / fps) if fps > 0 else 0.0
    
    # Book focusing score: bỏ công thức riêng, coi book là object bình thường
    # Giữ field để không phá API: dùng attention_to_book_percentage (0-100) làm score đơn giản.
    book_focusing_score = float(max(0.0, min(100.0, attention_to_book_percentage)))
    
    # Get results from tracking modules
    focus_timeline_data = []
    object_focus_stats = {}
    pattern_analysis = {}
    wandering_periods = []
    gaze_wandering_score = 0.0
    gaze_wandering_percentage = 0.0
    fatigue_score = 0.0
    fatigue_level = "low"
    fatigue_indicators = {}
    focus_level = 0.0
    focus_level_details = {}
    risk_score = 0.0
    
    # Initialize variables để tránh lỗi khi không có face
    if child_face_detected_count == 0:
        # Nếu không có face, set các giá trị mặc định
        gaze_wandering_percentage = 0.0
        gaze_wandering_score = 0.0
    
    if GAZE_TRACKING_MODULES_AVAILABLE:
        try:
            # Focus timeline
            if focus_timeline:
                focus_timeline_data = [
                    {
                        'object_id': p.object_id,
                        'start_time': p.start_time,
                        'end_time': p.end_time,
                        'duration': p.duration,
                        'class_name': p.class_name
                    }
                    for p in focus_timeline.focus_periods
                ]
                object_focus_stats = dict(focus_timeline.object_focus_stats)
                pattern_analysis = focus_timeline.get_pattern_analysis()
            
            # Wandering
            if wandering_detector:
                wandering_periods = [
                    {
                        'start_time': p.start_time,
                        'end_time': p.end_time,
                        'duration': p.duration,
                        'reason': p.reason
                    }
                    for p in wandering_detector.wandering_periods
                ]
                if child_face_detected_count > 0:
                    gaze_wandering_percentage = (wandering_detector.total_wandering_frames / child_face_detected_count) * 100
                    # Sử dụng calculate_wandering_score thay vì get_wandering_score
                    gaze_wandering_score, _ = wandering_detector.calculate_wandering_score(total_frames=child_face_detected_count)
                else:
                    gaze_wandering_percentage = 0.0
                    gaze_wandering_score = 0.0
            
            # Fatigue
            if fatigue_detector:
                fatigue_score, fatigue_level, fatigue_indicators = fatigue_detector.get_fatigue_assessment()
            
            # Focus level
            if focus_level_calc:
                focus_level, focus_level_details = focus_level_calc.get_focus_level()
            
            # Risk score (simple calculation)
            risk_score = (
                (100 - eye_contact_percentage) * 0.3 +
                gaze_wandering_percentage * 0.3 +
                (100 - focus_level) * 0.2 +
                fatigue_score * 0.2
            )
            risk_score = max(0, min(100, risk_score))
        except Exception as e:
            logger.warning(f"[Gaze] Error getting tracking module results: {str(e)}")
    
    # Create response
    response = GazeAnalysisResponse(
        eye_contact_percentage=eye_contact_percentage,
        gaze_direction_stats=gaze_direction_stats,
        total_frames=frame_count,
        analyzed_duration=analyzed_duration,
        focusing_duration=focusing_duration,
        attention_to_person_percentage=attention_to_person_percentage,
        attention_to_objects_percentage=attention_to_objects_percentage,
        attention_to_book_percentage=attention_to_book_percentage,
        book_focusing_score=book_focusing_score,
        detected_objects=detected_objects,
        detected_books=detected_books,
        object_interaction_events=object_interaction_events,
        risk_score=risk_score,
        focus_timeline=focus_timeline_data,
        object_focus_stats=object_focus_stats,
        pattern_analysis=pattern_analysis,
        gaze_wandering_score=gaze_wandering_score,
        gaze_wandering_percentage=gaze_wandering_percentage,
        wandering_periods=wandering_periods,
        fatigue_score=fatigue_score,
        fatigue_level=fatigue_level,
        fatigue_indicators=fatigue_indicators,
        focus_level=focus_level,
        focus_level_details=focus_level_details,
        object_detection_model=object_detector.model_name if object_detector else None,
        object_detection_available=object_detector is not None
    )
    
    logger.info(f"[Gaze] Hoàn thành phân tích: {frame_count} frames, {analyzed_duration:.2f}s")
    
    return response
