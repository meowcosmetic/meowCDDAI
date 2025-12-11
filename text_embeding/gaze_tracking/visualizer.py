"""
Visualization functions - Tách riêng drawing logic cho Gaze Tracking
"""
import cv2
import numpy as np
from typing import Optional, List, Dict, Any, Tuple


def draw_annotations(
    frame: np.ndarray,
    child_face: Optional[Any] = None,
    adult_face: Optional[Any] = None,
    gaze_dir: Optional[str] = None,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    is_focusing: bool = False,
    is_looking_at_adult: bool = False,
    is_looking_at_object: bool = False,
    frame_count: int = 0,
    fps: int = 30,
    gaze_x: Optional[float] = None,
    gaze_y: Optional[float] = None,
    head_pose: Optional[Tuple[float, float, float]] = None,
    variance: Optional[float] = None,
    rms_distance: Optional[float] = None,
    face_landmarks: Optional[Any] = None,
    show_landmarks: bool = False
) -> np.ndarray:
    """
    Vẽ các annotations lên frame để hiển thị
    
    Args:
        frame: Frame cần vẽ
        child_face: Face của trẻ (list/tuple [x, y, width, height] hoặc dict)
        adult_face: Face của người lớn (list/tuple [x, y, width, height] hoặc dict)
        gaze_dir: Hướng nhìn ("left", "right", "center", "up", "down")
        detected_objects: Danh sách objects được detect
        is_focusing: Đang focusing hay không
        is_looking_at_adult: Đang nhìn vào người lớn
        is_looking_at_object: Đang nhìn vào object
        frame_count: Số frame hiện tại
        fps: FPS của video
        gaze_x: Tọa độ X của gaze (normalized offset, -1.0 đến 1.0)
        gaze_y: Tọa độ Y của gaze (normalized offset, -1.0 đến 1.0)
        head_pose: Tuple (yaw, pitch, roll) - hướng quay đầu (radians)
        variance: Variance của gaze (legacy metric)
        rms_distance: RMS distance của gaze (improved metric)
        face_landmarks: MediaPipe face landmarks object
        show_landmarks: Có hiển thị eye landmarks không (default: False)
    """
    h, w = frame.shape[:2]
    annotated_frame = frame.copy()
    
    # Vẽ MediaPipe eye landmarks nếu có
    if show_landmarks and face_landmarks is not None:
        try:
            # MediaPipe Face Mesh landmark indices
            LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            LEFT_EYE_CENTER = 468  # Left iris center
            RIGHT_EYE_CENTER = 473  # Right iris center
            
            # Vẽ left eye landmarks (màu xanh lá)
            for idx in LEFT_EYE_INDICES:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), -1)  # Green dots
            
            # Vẽ right eye landmarks (màu xanh dương)
            for idx in RIGHT_EYE_INDICES:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_frame, (x, y), 2, (255, 0, 0), -1)  # Blue dots
            
            # Vẽ left eye center (iris) - màu vàng, lớn hơn
            if LEFT_EYE_CENTER < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[LEFT_EYE_CENTER]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 4, (0, 255, 255), -1)  # Yellow, larger
                cv2.putText(annotated_frame, "L", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Vẽ right eye center (iris) - màu vàng, lớn hơn
            if RIGHT_EYE_CENTER < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[RIGHT_EYE_CENTER]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 4, (0, 255, 255), -1)  # Yellow, larger
                cv2.putText(annotated_frame, "R", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Vẽ outline của mắt (nối các điểm landmarks)
            # Left eye outline
            if len(LEFT_EYE_INDICES) > 0:
                left_eye_points = []
                for idx in LEFT_EYE_INDICES[:8]:  # Lấy 8 điểm đầu để vẽ outline
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        left_eye_points.append([int(lm.x * w), int(lm.y * h)])
                if len(left_eye_points) > 2:
                    cv2.polylines(annotated_frame, [np.array(left_eye_points, np.int32)], 
                                False, (0, 255, 0), 1)  # Green outline
            
            # Right eye outline
            if len(RIGHT_EYE_INDICES) > 0:
                right_eye_points = []
                for idx in RIGHT_EYE_INDICES[:8]:  # Lấy 8 điểm đầu để vẽ outline
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        right_eye_points.append([int(lm.x * w), int(lm.y * h)])
                if len(right_eye_points) > 2:
                    cv2.polylines(annotated_frame, [np.array(right_eye_points, np.int32)], 
                                False, (255, 0, 0), 1)  # Blue outline
        except (AttributeError, IndexError, TypeError) as e:
            # Nếu face_landmarks không đúng format, bỏ qua
            pass
    
    # Vẽ face của trẻ (màu xanh lá)
    if child_face is not None:
        if isinstance(child_face, (list, tuple)) and len(child_face) >= 4:
            # OpenCV format: [x, y, width, height]
            x, y, w_face, h_face = child_face[:4]
            cv2.rectangle(annotated_frame, (int(x), int(y)), 
                         (int(x + w_face), int(y + h_face)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Child", (int(x), int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif isinstance(child_face, dict):
            bbox = child_face.get('bbox', [])
            if len(bbox) >= 4:
                x, y, w_face, h_face = bbox[:4]
                cv2.rectangle(annotated_frame, (int(x), int(y)), 
                             (int(x + w_face), int(y + h_face)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Child", (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Vẽ face của người lớn (màu xanh dương)
    if adult_face is not None:
        if isinstance(adult_face, (list, tuple)) and len(adult_face) >= 4:
            x, y, w_face, h_face = adult_face[:4]
            cv2.rectangle(annotated_frame, (int(x), int(y)), 
                         (int(x + w_face), int(y + h_face)), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "Adult", (int(x), int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        elif isinstance(adult_face, dict):
            bbox = adult_face.get('bbox', [])
            if len(bbox) >= 4:
                x, y, w_face, h_face = bbox[:4]
                cv2.rectangle(annotated_frame, (int(x), int(y)), 
                             (int(x + w_face), int(y + h_face)), (255, 0, 0), 2)
                cv2.putText(annotated_frame, "Adult", (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Vẽ gaze direction arrow
    if child_face is not None and gaze_dir is not None:
        if isinstance(child_face, (list, tuple)) and len(child_face) >= 4:
            x, y, w_face, h_face = child_face[:4]
            face_center_x = int(x + w_face / 2)
            face_center_y = int(y + h_face / 2)
        elif isinstance(child_face, dict):
            bbox = child_face.get('bbox', [])
            if len(bbox) >= 4:
                x, y, w_face, h_face = bbox[:4]
                face_center_x = int(x + w_face / 2)
                face_center_y = int(y + h_face / 2)
            else:
                face_center_x = w // 2
                face_center_y = h // 2
        else:
            face_center_x = w // 2
            face_center_y = h // 2
        
        # Tính vị trí mũi tên dựa trên gaze direction
        arrow_length = 50
        if gaze_dir == "left":
            end_x = face_center_x - arrow_length
            end_y = face_center_y
        elif gaze_dir == "right":
            end_x = face_center_x + arrow_length
            end_y = face_center_y
        elif gaze_dir == "up":
            end_x = face_center_x
            end_y = face_center_y - arrow_length
        elif gaze_dir == "down":
            end_x = face_center_x
            end_y = face_center_y + arrow_length
        else:  # center
            end_x = face_center_x
            end_y = face_center_y
        
        cv2.arrowedLine(annotated_frame, (face_center_x, face_center_y),
                       (end_x, end_y), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(annotated_frame, f"Gaze: {gaze_dir}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Vẽ detected objects - TẤT CẢ objects, không chỉ sách
    if detected_objects:
        # Vẽ tất cả objects với bounding boxes
        for obj in detected_objects:
            obj_class = obj.get('class', 'unknown')
            bbox = obj.get('bbox', [])
            confidence = obj.get('confidence', 0)
            track_id = obj.get('track_id')  # Track ID nếu có
            
            if len(bbox) >= 4:
                x, y, w_obj, h_obj = bbox[:4]
                
                # Màu khác nhau cho từng loại object
                if obj_class == 'book':
                    color = (0, 255, 255)  # Cyan - highlight sách
                    thickness = 3
                    emoji = "📖"
                elif obj_class == 'person':
                    color = (255, 165, 0)  # Orange
                    thickness = 2
                    emoji = "👤"
                elif obj_class == 'cup':
                    color = (255, 0, 255)  # Magenta
                    thickness = 2
                    emoji = "☕"
                elif obj_class == 'bottle':
                    color = (0, 255, 0)  # Green
                    thickness = 2
                    emoji = "🍼"
                elif obj_class == 'cell phone':
                    color = (255, 255, 0)  # Yellow
                    thickness = 2
                    emoji = "📱"
                elif obj_class == 'laptop':
                    color = (128, 0, 128)  # Purple
                    thickness = 2
                    emoji = "💻"
                elif obj_class in ['pen', 'pencil', 'marker', 'crayon']:  # ✅ OID có pen/pencil!
                    color = (0, 255, 255)  # Cyan
                    thickness = 2
                    emoji = "🖊️"
                elif obj_class in ['scissors', 'knife']:
                    color = (255, 200, 0)  # Orange-yellow
                    thickness = 2
                    emoji = "✂️"
                elif obj_class in ['toothbrush', 'hair drier']:
                    color = (128, 128, 128)  # Gray
                    thickness = 2
                    emoji = "📦"
                elif obj_class == 'remote':
                    color = (128, 128, 0)  # Olive
                    thickness = 2
                    emoji = "📺"
                elif obj_class == 'mouse':
                    color = (0, 128, 255)  # Orange
                    thickness = 2
                    emoji = "🖱️"
                elif obj_class == 'keyboard':
                    color = (255, 128, 0)  # Orange
                    thickness = 2
                    emoji = "⌨️"
                else:
                    # ✅ TẤT CẢ objects khác đều được hiển thị
                    color = (255, 0, 255)  # Magenta - default
                    thickness = 2
                    emoji = "📦"
                
                # Vẽ bounding box
                cv2.rectangle(annotated_frame, (int(x), int(y)), 
                             (int(x + w_obj), int(y + h_obj)), color, thickness)
                
                # Label với emoji, class name, confidence và track_id
                label = f"{emoji} {obj_class}"
                if track_id is not None:
                    label += f" ID:{track_id}"
                label += f" {confidence:.2f}"
                
                # Vẽ label với background để dễ đọc
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness
                )
                cv2.rectangle(annotated_frame, 
                             (int(x), int(y) - text_height - 10),
                             (int(x) + text_width, int(y)),
                             color, -1)  # Filled rectangle
                cv2.putText(annotated_frame, label, (int(x), int(y) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness)  # Black text
    
    # Vẽ status bar ở trên cùng
    status_y = 20
    status_color = (0, 255, 0) if is_focusing else (0, 0, 255)
    status_text = "FOCUSING" if is_focusing else "NOT FOCUSING"
    cv2.rectangle(annotated_frame, (10, 5), (w - 10, 35), (0, 0, 0), -1)
    cv2.putText(annotated_frame, status_text, (20, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Vẽ thông tin attention
    info_y = 60
    if is_looking_at_adult:
        cv2.putText(annotated_frame, "Looking at Adult", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        info_y += 25
    if is_looking_at_object:
        cv2.putText(annotated_frame, "Looking at Object", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        info_y += 25
    
    # Vẽ frame count và time
    if fps > 0:
        time_sec = frame_count / fps
        time_text = f"Frame: {frame_count} | Time: {time_sec:.2f}s"
        cv2.putText(annotated_frame, time_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Vẽ hướng dẫn dừng (ở góc trên bên phải)
    stop_text = "Press 'q' or ESC to stop"
    text_size = cv2.getTextSize(stop_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = w - text_size[0] - 10
    text_y = 25
    # Background cho text
    cv2.rectangle(annotated_frame, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (0, 0, 0), -1)  # Black background
    cv2.putText(annotated_frame, stop_text, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Cyan text
    
    # Vẽ vị trí gaze (gaze_x, gaze_y) nếu có
    if gaze_x is not None and gaze_y is not None:
        # Hiển thị giá trị gaze_x và gaze_y
        gaze_text = f"Gaze: X={gaze_x:.3f}, Y={gaze_y:.3f}"
        text_x = 10
        text_y = h - 40  # Ở trên dòng time
        
        # Vẽ background cho text để dễ đọc
        (text_width, text_height), baseline = cv2.getTextSize(
            gaze_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(annotated_frame, 
                     (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + 5),
                     (0, 0, 0), -1)  # Black background
        
        # Vẽ text
        cv2.putText(annotated_frame, gaze_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Cyan color
        
        # Vẽ crosshair tại vị trí gaze trên frame (nếu có child_face)
        if child_face is not None:
            if isinstance(child_face, (list, tuple)) and len(child_face) >= 4:
                x_face, y_face, w_face, h_face = child_face[:4]
                face_center_x = int(x_face + w_face / 2)
                face_center_y = int(y_face + h_face / 2)
            elif isinstance(child_face, dict):
                bbox = child_face.get('bbox', [])
                if len(bbox) >= 4:
                    x_face, y_face, w_face, h_face = bbox[:4]
                    face_center_x = int(x_face + w_face / 2)
                    face_center_y = int(y_face + h_face / 2)
                else:
                    face_center_x = w // 2
                    face_center_y = h // 2
            else:
                face_center_x = w // 2
                face_center_y = h // 2
            
            # Tính vị trí gaze trên frame (từ normalized offset)
            # gaze_x và gaze_y là offset từ face center, tính theo frame size
            gaze_pixel_x = int(face_center_x + gaze_x * (w / 2))
            gaze_pixel_y = int(face_center_y + gaze_y * (h / 2))
            
            # Đảm bảo trong frame bounds
            gaze_pixel_x = max(0, min(w - 1, gaze_pixel_x))
            gaze_pixel_y = max(0, min(h - 1, gaze_pixel_y))
            
            # Vẽ crosshair (dấu +) tại vị trí gaze
            crosshair_size = 15
            crosshair_color = (0, 255, 255)  # Cyan
            crosshair_thickness = 2
            
            # Vẽ đường ngang
            cv2.line(annotated_frame,
                    (gaze_pixel_x - crosshair_size, gaze_pixel_y),
                    (gaze_pixel_x + crosshair_size, gaze_pixel_y),
                    crosshair_color, crosshair_thickness)
            # Vẽ đường dọc
            cv2.line(annotated_frame,
                    (gaze_pixel_x, gaze_pixel_y - crosshair_size),
                    (gaze_pixel_x, gaze_pixel_y + crosshair_size),
                    crosshair_color, crosshair_thickness)
            
            # Vẽ điểm tròn tại vị trí gaze
            cv2.circle(annotated_frame, (gaze_pixel_x, gaze_pixel_y), 5, crosshair_color, -1)
            
            # Vẽ đường nối từ face center đến gaze position
            cv2.line(annotated_frame,
                    (face_center_x, face_center_y),
                    (gaze_pixel_x, gaze_pixel_y),
                    (255, 255, 0), 2)  # Yellow line
    
    # Vẽ hướng quay đầu (head rotation) nếu có
    if child_face is not None and head_pose is not None:
        if isinstance(child_face, (list, tuple)) and len(child_face) >= 4:
            x, y, w_face, h_face = child_face[:4]
            face_center_x = int(x + w_face / 2)
            face_center_y = int(y + h_face / 2)
        elif isinstance(child_face, dict):
            bbox = child_face.get('bbox', [])
            if len(bbox) >= 4:
                x, y, w_face, h_face = bbox[:4]
                face_center_x = int(x + w_face / 2)
                face_center_y = int(y + h_face / 2)
            else:
                face_center_x = w // 2
                face_center_y = h // 2
        else:
            face_center_x = w // 2
            face_center_y = h // 2
        
        try:
            yaw, pitch, roll = head_pose
            
            # Chuyển đổi từ radians sang degrees để hiển thị
            yaw_deg = np.degrees(yaw)
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)
            
            # Tính hướng quay đầu dựa trên yaw và pitch
            # Arrow length tỷ lệ với góc quay
            max_angle = 30.0  # degrees
            arrow_length_base = 60
            
            # Yaw (left/right rotation)
            yaw_ratio = np.clip(abs(yaw_deg) / max_angle, 0, 1)
            yaw_arrow_length = int(arrow_length_base * yaw_ratio)
            if abs(yaw_deg) > 2:  # Chỉ vẽ nếu quay đáng kể (>2 độ)
                if yaw_deg < 0:  # Quay trái
                    yaw_end_x = face_center_x - yaw_arrow_length
                    yaw_end_y = face_center_y
                else:  # Quay phải
                    yaw_end_x = face_center_x + yaw_arrow_length
                    yaw_end_y = face_center_y
                
                # Vẽ arrow cho yaw (màu đỏ)
                cv2.arrowedLine(annotated_frame, 
                               (face_center_x, face_center_y),
                               (yaw_end_x, yaw_end_y),
                               (0, 0, 255), 2, tipLength=0.3)
            
            # Pitch (up/down rotation)
            pitch_ratio = np.clip(abs(pitch_deg) / max_angle, 0, 1)
            pitch_arrow_length = int(arrow_length_base * pitch_ratio)
            if abs(pitch_deg) > 2:  # Chỉ vẽ nếu quay đáng kể (>2 độ)
                if pitch_deg < 0:  # Quay lên
                    pitch_end_x = face_center_x
                    pitch_end_y = face_center_y - pitch_arrow_length
                else:  # Quay xuống
                    pitch_end_x = face_center_x
                    pitch_end_y = face_center_y + pitch_arrow_length
                
                # Vẽ arrow cho pitch (màu xanh lá)
                cv2.arrowedLine(annotated_frame,
                               (face_center_x, face_center_y),
                               (pitch_end_x, pitch_end_y),
                               (0, 255, 0), 2, tipLength=0.3)
            
            # Hiển thị thông tin head pose
            head_pose_text = f"Head: Yaw={yaw_deg:.1f}° Pitch={pitch_deg:.1f}° Roll={roll_deg:.1f}°"
            text_x = 10
            text_y = 90  # Ở dưới gaze direction text
            
            # Vẽ background cho text
            (text_width, text_height), baseline = cv2.getTextSize(
                head_pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(annotated_frame,
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (0, 0, 0), -1)  # Black background
            
            # Vẽ text
            cv2.putText(annotated_frame, head_pose_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # White color
            
            # Vẽ legend cho head rotation arrows
            legend_y = text_y + 25
            cv2.putText(annotated_frame, "Red: Yaw (L/R)", (text_x, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(annotated_frame, "Green: Pitch (U/D)", (text_x, legend_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        except (ValueError, TypeError) as e:
            # Nếu head_pose không đúng format, bỏ qua
            pass
    
    # Vẽ variance và RMS distance nếu có
    stats_y = h - 70  # Ở trên gaze text
    stats_texts = []
    
    if variance is not None:
        stats_texts.append(f"Variance: {variance:.6f}")
    
    if rms_distance is not None:
        stats_texts.append(f"RMS: {rms_distance:.6f}")
    
    if stats_texts:
        stats_text = " | ".join(stats_texts)
        text_x = 10
        
        # Vẽ background cho text
        (text_width, text_height), baseline = cv2.getTextSize(
            stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(annotated_frame,
                     (text_x - 5, stats_y - text_height - 5),
                     (text_x + text_width + 5, stats_y + 5),
                     (0, 0, 0), -1)  # Black background
        
        # Vẽ text
        cv2.putText(annotated_frame, stats_text, (text_x, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Yellow color
    
    return annotated_frame
