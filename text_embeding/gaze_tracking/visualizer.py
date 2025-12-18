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
            
            # Vẽ left eye landmarks (màu xanh lá) - nhỏ gọn hơn
            for idx in LEFT_EYE_INDICES:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_frame, (x, y), 1, (0, 255, 0), -1)  # Green dots (smaller)
            
            # Vẽ right eye landmarks (màu xanh dương) - nhỏ gọn hơn
            for idx in RIGHT_EYE_INDICES:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated_frame, (x, y), 1, (255, 0, 0), -1)  # Blue dots (smaller)
            
            # Vẽ left eye center (iris) - màu vàng, nhỏ gọn hơn
            if LEFT_EYE_CENTER < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[LEFT_EYE_CENTER]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 2, (0, 255, 255), -1)  # Yellow (smaller)
                cv2.putText(annotated_frame, "L", (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            
            # Vẽ right eye center (iris) - màu vàng, nhỏ gọn hơn
            if RIGHT_EYE_CENTER < len(face_landmarks.landmark):
                lm = face_landmarks.landmark[RIGHT_EYE_CENTER]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated_frame, (x, y), 2, (0, 255, 255), -1)  # Yellow (smaller)
                cv2.putText(annotated_frame, "R", (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            
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
    
    # Vẽ gaze direction vector - sử dụng gaze_x và gaze_y nếu có
    # Luôn vẽ mũi tên nếu có child_face hoặc gaze_dir
    should_draw_gaze = False
    face_center_x = w // 2
    face_center_y = h // 2
    
    if child_face is not None:
        should_draw_gaze = True
        # Tìm tâm của khuôn mặt trẻ
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
    
    # Nếu không có child_face nhưng có gaze_dir, vẫn vẽ mũi tên ở giữa màn hình
    if not should_draw_gaze and gaze_dir is not None:
        should_draw_gaze = True
    
    # Vẽ gaze vector dựa trên gaze_x và gaze_y (nếu có)
    if should_draw_gaze and gaze_x is not None and gaze_y is not None:
        """
        GIẢI THÍCH VỀ GAZE MAGNITUDE:
        
        1. gaze_x và gaze_y là gì?
           - Đây là offset của con ngươi (iris) so với TÂM MẮT
           - Công thức: gaze_x = (iris_x - eye_center_x)
           - Giá trị normalized trong hệ tọa độ MediaPipe (0.0-1.0)
        
        2. Tại sao gaze_magnitude lại NHỎ?
           - Con ngươi chỉ di chuyển một khoảng RẤT NHỎ trong mắt
           - Khi nhìn thẳng: iris ở giữa mắt → offset ≈ 0.0
           - Khi nhìn sang trái/phải: iris chỉ di chuyển ~1-5% chiều rộng mắt
           - Trong hệ normalized (0.0-1.0), offset thường chỉ từ -0.05 đến 0.05
           - Vậy nên gaze_magnitude thường rất nhỏ: 0.001 - 0.05
        
        3. Gaze magnitude có phải hướng nhìn không?
           - CÓ, nhưng chỉ là hướng TƯƠNG ĐỐI trong mắt
           - Không phải hướng nhìn tuyệt đối trong không gian 3D
           - Chỉ cho biết con ngươi đang ở đâu trong mắt:
             * gaze_x > 0: nhìn sang phải (iris ở bên phải tâm mắt)
             * gaze_x < 0: nhìn sang trái (iris ở bên trái tâm mắt)
             * gaze_y < 0: nhìn lên trên (iris ở trên tâm mắt)
             * gaze_y > 0: nhìn xuống dưới (iris ở dưới tâm mắt)
        
        4. Tại sao cần nhân lên?
           - Để hiển thị mũi tên rõ ràng trên màn hình
           - Magnitude nhỏ → mũi tên ngắn → khó nhìn thấy
           - Nhân lên 200 lần để phóng đại và dễ quan sát
        """
        
        # Tính độ dài thực tế của vector gaze (magnitude)
        gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
        
        # Base length cho mũi tên (30% của kích thước nhỏ hơn - nhỏ hơn để tinh tế hơn)
        base_length = min(w, h) * 0.3
        
        # Độ dài tối thiểu để mũi tên luôn nhìn thấy được (10% frame)
        min_arrow_length = min(w, h) * 0.1
        
        if gaze_magnitude < 0.01:
            # Nếu gaze quá nhỏ (< 0.01), nhân lên 200 lần để phóng đại
            # Điều này giúp hiển thị rõ ràng ngay cả khi nhìn thẳng
            amplified_gaze_x = gaze_x * 200
            amplified_gaze_y = gaze_y * 200
            amplified_magnitude = np.sqrt(amplified_gaze_x**2 + amplified_gaze_y**2)
            
            # Normalize để giữ hướng nhưng có độ dài hợp lý
            if amplified_magnitude > 0:
                normalized_gaze_x = amplified_gaze_x / amplified_magnitude
                normalized_gaze_y = amplified_gaze_y / amplified_magnitude
            else:
                # Nếu vẫn bằng 0 sau khi nhân, vẽ mũi tên nhỏ lên trên
                normalized_gaze_x = 0
                normalized_gaze_y = -1
            
            # Độ dài mũi tên từ 15% đến 40% frame
            arrow_length = max(min_arrow_length, base_length * min(1.0, amplified_magnitude / 10))
            
            end_x = int(face_center_x + normalized_gaze_x * arrow_length)
            end_y = int(face_center_y + normalized_gaze_y * arrow_length)
        else:
            # Nếu magnitude đủ lớn (>= 0.01), sử dụng giá trị gốc nhưng scale hợp lý
            # Normalize để giữ hướng
            normalized_gaze_x = gaze_x / gaze_magnitude if gaze_magnitude > 0 else 0
            normalized_gaze_y = gaze_y / gaze_magnitude if gaze_magnitude > 0 else 0
            
            # Độ dài mũi tên tỷ lệ với magnitude nhưng có minimum
            # Scale magnitude để mũi tên có độ dài từ 15% đến 40% frame
            arrow_length = max(min_arrow_length, base_length * min(1.0, gaze_magnitude * 15))
            
            end_x = int(face_center_x + normalized_gaze_x * arrow_length)
            end_y = int(face_center_y + normalized_gaze_y * arrow_length)
        
        # Vẽ mũi tên gaze vector (màu vàng, mỏng hơn để giống các annotation khác)
        cv2.arrowedLine(annotated_frame, (face_center_x, face_center_y),
                       (end_x, end_y), (0, 255, 255), 2, tipLength=0.2, line_type=cv2.LINE_AA)
        
        # Vẽ điểm bắt đầu (mắt) - vòng tròn nhỏ hơn
        cv2.circle(annotated_frame, (face_center_x, face_center_y), 3, (0, 255, 255), -1)
        
        # Hiển thị thông tin gaze (font nhỏ hơn)
        gaze_info = f"Gaze: ({gaze_x:.3f}, {gaze_y:.3f})"
        if gaze_dir:
            gaze_info += f" [{gaze_dir}]"
        cv2.putText(annotated_frame, gaze_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif should_draw_gaze and gaze_dir is not None:
        # Fallback: sử dụng gaze_dir nếu không có gaze_x/gaze_y
        arrow_length = min(w, h) * 0.25  # Tăng độ dài mũi tên
        if gaze_dir == "left":
            end_x = face_center_x - int(arrow_length)
            end_y = face_center_y
        elif gaze_dir == "right":
            end_x = face_center_x + int(arrow_length)
            end_y = face_center_y
        elif gaze_dir == "up":
            end_x = face_center_x
            end_y = face_center_y - int(arrow_length)
        elif gaze_dir == "down":
            end_x = face_center_x
            end_y = face_center_y + int(arrow_length)
        else:  # center
            # Vẽ mũi tên nhỏ lên trên để chỉ ra đang nhìn thẳng
            end_x = face_center_x
            end_y = face_center_y - int(arrow_length * 0.3)
        
        cv2.arrowedLine(annotated_frame, (face_center_x, face_center_y),
                       (end_x, end_y), (0, 255, 255), 2, tipLength=0.2, line_type=cv2.LINE_AA)
        cv2.circle(annotated_frame, (face_center_x, face_center_y), 3, (0, 255, 255), -1)
        cv2.putText(annotated_frame, f"Gaze: {gaze_dir}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
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
        
        # Vẽ crosshair tại vị trí gaze trên frame (chỉ khi có child_face)
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
                    w_face, h_face = w * 0.3, h * 0.3
            else:
                face_center_x = w // 2
                face_center_y = h // 2
                w_face, h_face = w * 0.3, h * 0.3

            # Tính vị trí gaze hiển thị trên frame (từ gaze_x/gaze_y)
            # Lưu ý: gaze_x/gaze_y là offset đã chuẩn hoá theo KÍCH THƯỚC MẮT (không phải theo frame).
            # Vì vậy khi vẽ, scale theo kích thước khuôn mặt để trực quan hơn và tránh “bắn” ra xa.
            scale_x = max(40, int(w_face * 0.9))
            scale_y = max(40, int(h_face * 0.9))

            gaze_pixel_x = int(face_center_x + gaze_x * scale_x)
            gaze_pixel_y = int(face_center_y + gaze_y * scale_y)

            # Đảm bảo trong frame bounds
            gaze_pixel_x = max(0, min(w - 1, gaze_pixel_x))
            gaze_pixel_y = max(0, min(h - 1, gaze_pixel_y))

            # Vẽ crosshair (dấu +) tại vị trí gaze
            crosshair_size = 15
            # Điểm nhìn (crosshair) dùng màu khác để phân biệt với mũi tên gaze (màu vàng)
            crosshair_color = (255, 0, 255)  # Magenta
            crosshair_thickness = 2

            # Vẽ đường ngang
            cv2.line(
                annotated_frame,
                (gaze_pixel_x - crosshair_size, gaze_pixel_y),
                (gaze_pixel_x + crosshair_size, gaze_pixel_y),
                crosshair_color,
                crosshair_thickness,
            )
            # Vẽ đường dọc
            cv2.line(
                annotated_frame,
                (gaze_pixel_x, gaze_pixel_y - crosshair_size),
                (gaze_pixel_x, gaze_pixel_y + crosshair_size),
                crosshair_color,
                crosshair_thickness,
            )

            # Vẽ điểm tròn tại vị trí gaze
            cv2.circle(annotated_frame, (gaze_pixel_x, gaze_pixel_y), 5, crosshair_color, -1)

            # Vẽ đường nối từ face center đến gaze position (cùng màu với điểm nhìn)
            cv2.line(
                annotated_frame,
                (face_center_x, face_center_y),
                (gaze_pixel_x, gaze_pixel_y),
                crosshair_color,
                2,
            )  # Yellow line
    
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
                # Với cách tính pitch hiện tại trong processor, pitch âm thường tương ứng “cúi xuống”.
                if pitch_deg < 0:  # Quay xuống
                    pitch_end_x = face_center_x
                    pitch_end_y = face_center_y + pitch_arrow_length
                else:  # Quay lên
                    pitch_end_x = face_center_x
                    pitch_end_y = face_center_y - pitch_arrow_length
                
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
