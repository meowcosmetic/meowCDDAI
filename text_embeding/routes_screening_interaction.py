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
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("[Interaction] MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe")
    mp = None
    mp_hands = None
    mp_drawing = None

# GPU detection
USE_GPU = Config.USE_GPU.lower() if hasattr(Config, 'USE_GPU') else "auto"
GPU_AVAILABLE = False

try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        GPU_AVAILABLE = True
        logger.info(f"[Interaction] ✅ OpenCV GPU detected")
except:
    pass

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

router = APIRouter(prefix="/screening/interaction", tags=["Screening - Interaction Detection"])


def load_yolo_model():
    """Load YOLO model cho object detection"""
    try:
        yolo_configs = [
            ("yolov3-tiny.cfg", "yolov3-tiny.weights"),
            ("yolov3.cfg", "yolov3.weights"),
            ("yolov4-tiny.cfg", "yolov4-tiny.weights"),
        ]
        
        for yolo_config, yolo_weights in yolo_configs:
            if os.path.exists(yolo_weights) and os.path.exists(yolo_config):
                try:
                    net = cv2.dnn.readNet(yolo_weights, yolo_config)
                    if GPU_AVAILABLE:
                        try:
                            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        except:
                            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    else:
                        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
                    logger.info(f"[Interaction] ✅ Đã load YOLO model: {yolo_config}")
                    return net, output_layers
                except Exception as e:
                    continue
        
        logger.warning("[Interaction] YOLO weights không tìm thấy")
        return None, None
    except Exception as e:
        logger.warning(f"[Interaction] Không thể load YOLO model: {str(e)}")
        return None, None


def detect_objects_yolo(frame, net, output_layers, conf_threshold=0.5):
    """Detect objects trong frame sử dụng YOLO"""
    if net is None:
        return []
    
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
    
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            class_name = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else "unknown"
            detected_objects.append({
                'class': class_name,
                'class_id': int(class_ids[i]),
                'confidence': round(confidences[i], 2),
                'bbox': boxes[i],
                'center': [boxes[i][0] + boxes[i][2] // 2, boxes[i][1] + boxes[i][3] // 2]
            })
    
    return detected_objects


def track_objects(current_objects, previous_tracks, max_distance=100):
    """
    Simple object tracking dựa trên IoU và distance
    """
    tracks = []
    used_indices = set()
    
    for obj in current_objects:
        best_match_idx = None
        best_iou = 0
        
        for idx, prev_track in enumerate(previous_tracks):
            if idx in used_indices:
                continue
            
            # Tính IoU
            bbox1 = obj['bbox']
            bbox2 = prev_track['bbox']
            iou = calculate_iou(bbox1, bbox2)
            
            # Tính distance
            center1 = obj['center']
            center2 = prev_track['center']
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            if iou > best_iou and distance < max_distance:
                best_iou = iou
                best_match_idx = idx
        
        if best_match_idx is not None and best_iou > 0.3:
            # Update existing track
            track = previous_tracks[best_match_idx].copy()
            track['bbox'] = obj['bbox']
            track['center'] = obj['center']
            track['confidence'] = obj['confidence']
            track['age'] = track.get('age', 0) + 1
            used_indices.add(best_match_idx)
        else:
            # New track
            track = obj.copy()
            track['age'] = 1
            track['track_id'] = len(previous_tracks) + len(tracks)
        
        tracks.append(track)
    
    return tracks


def calculate_iou(bbox1, bbox2):
    """Tính Intersection over Union giữa 2 bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Tính intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def detect_pointing_gesture(hand_landmarks, frame_shape):
    """
    Detect pointing gesture từ MediaPipe hand landmarks
    """
    if not hand_landmarks:
        return False, None
    
    h, w = frame_shape[:2]
    
    # Index finger tip (8) và MCP (5)
    index_tip = hand_landmarks.landmark[8]
    index_mcp = hand_landmarks.landmark[5]
    thumb_tip = hand_landmarks.landmark[4]
    
    # Tính vector từ MCP đến tip
    index_vector = np.array([
        (index_tip.x - index_mcp.x) * w,
        (index_tip.y - index_mcp.y) * h
    ])
    
    # Tính khoảng cách từ thumb đến index
    thumb_index_dist = np.sqrt(
        ((index_tip.x - thumb_tip.x) * w)**2 + 
        ((index_tip.y - thumb_tip.y) * h)**2
    )
    
    # Pointing: index finger extended, thumb và other fingers closed
    index_length = np.linalg.norm(index_vector)
    
    # Kiểm tra các fingers khác có closed không
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    middle_mcp = hand_landmarks.landmark[9]
    ring_mcp = hand_landmarks.landmark[13]
    pinky_mcp = hand_landmarks.landmark[17]
    
    middle_closed = hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y
    ring_closed = hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y
    pinky_closed = hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y
    
    # Pointing gesture: index extended, others closed
    is_pointing = (index_length > 30 and 
                   thumb_index_dist < 50 and
                   middle_closed and ring_closed and pinky_closed)
    
    if is_pointing:
        pointing_direction = index_vector / (index_length + 1e-6)
        pointing_pos = np.array([index_tip.x * w, index_tip.y * h])
        return True, {'direction': pointing_direction, 'position': pointing_pos}
    
    return False, None


def detect_giving_gesture(hand_landmarks, object_bbox, frame_shape):
    """
    Detect giving gesture: hand near object và extended
    """
    if not hand_landmarks or not object_bbox:
        return False
    
    h, w = frame_shape[:2]
    
    # Hand center (wrist)
    wrist = hand_landmarks.landmark[0]
    hand_center = np.array([wrist.x * w, wrist.y * h])
    
    # Object center
    obj_x, obj_y, obj_w, obj_h = object_bbox
    obj_center = np.array([obj_x + obj_w / 2, obj_y + obj_h / 2])
    
    # Distance từ hand đến object
    distance = np.linalg.norm(hand_center - obj_center)
    obj_size = np.sqrt(obj_w**2 + obj_h**2)
    
    # Hand phải gần object
    if distance < obj_size * 1.5:
        # Kiểm tra hand có extended không (fingers open)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        
        # Fingers extended
        thumb_extended = thumb_tip.x > hand_landmarks.landmark[3].x
        index_extended = index_tip.y < hand_landmarks.landmark[6].y
        middle_extended = middle_tip.y < hand_landmarks.landmark[10].y
        
        if thumb_extended and index_extended and middle_extended:
            return True
    
    return False


def detect_interaction_events(tracks, hand_gestures, previous_state, frame_count, fps):
    """
    Detect các interaction events từ tracks và gestures
    """
    events = []
    
    # Tìm person và objects
    persons = [t for t in tracks if t['class'] == 'person']
    objects = [t for t in tracks if t['class'] != 'person']
    
    # Phân biệt parent và child (giả định: person lớn hơn = parent)
    if len(persons) >= 2:
        persons_sorted = sorted(persons, key=lambda p: p['bbox'][2] * p['bbox'][3], reverse=True)
        parent = persons_sorted[0]
        child = persons_sorted[1]
    elif len(persons) == 1:
        parent = None
        child = persons[0]
    else:
        parent = None
        child = None
    
    # Detect pointing gestures
    for hand_gesture in hand_gestures:
        if hand_gesture['type'] == 'pointing':
            events.append({
                'type': 'pointing',
                'timestamp': frame_count / fps,
                'frame': frame_count,
                'description': 'Pointing gesture detected'
            })
    
    # Detect object offers (parent near object)
    if parent and objects:
        for obj in objects:
            parent_center = np.array(parent['center'])
            obj_center = np.array(obj['center'])
            distance = np.linalg.norm(parent_center - obj_center)
            obj_size = np.sqrt(obj['bbox'][2]**2 + obj['bbox'][3]**2)
            
            if distance < obj_size * 2:
                # Parent đang cầm/đưa object
                event_key = f"offer_{obj['track_id']}"
                if event_key not in previous_state.get('active_offers', {}):
                    events.append({
                        'type': 'object_offer',
                        'timestamp': frame_count / fps,
                        'frame': frame_count,
                        'object_class': obj['class'],
                        'description': f"Parent offering {obj['class']}"
                    })
    
    # Detect following (child gaze/facing towards object)
    if child and objects:
        for obj in objects:
            child_center = np.array(child['center'])
            obj_center = np.array(obj['center'])
            distance = np.linalg.norm(child_center - obj_center)
            obj_size = np.sqrt(obj['bbox'][2]**2 + obj['bbox'][3]**2)
            
            # Child looking at object
            if distance < obj_size * 3:
                event_key = f"following_{obj['track_id']}"
                if event_key not in previous_state.get('active_following', {}):
                    events.append({
                        'type': 'following',
                        'timestamp': frame_count / fps,
                        'frame': frame_count,
                        'object_class': obj['class'],
                        'description': f"Child following {obj['class']}"
                    })
    
    # Detect object exchange (child near object after parent offer)
    if child and objects:
        for obj in objects:
            child_center = np.array(child['center'])
            obj_center = np.array(obj['center'])
            distance = np.linalg.norm(child_center - obj_center)
            obj_size = np.sqrt(obj['bbox'][2]**2 + obj['bbox'][3]**2)
            
            if distance < obj_size * 1.5:
                # Child receiving object
                event_key = f"exchange_{obj['track_id']}"
                if event_key not in previous_state.get('active_exchanges', {}):
                    events.append({
                        'type': 'object_exchange',
                        'timestamp': frame_count / fps,
                        'frame': frame_count,
                        'object_class': obj['class'],
                        'description': f"Child receiving {obj['class']}"
                    })
    
    return events


def draw_interaction_annotations(frame, tracks, hand_gestures, interaction_events, frame_count=0, fps=30):
    """Vẽ annotations cho interaction detection"""
    annotated_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Vẽ tracked objects
    for track in tracks:
        x, y, w_obj, h_obj = track['bbox']
        class_name = track['class']
        track_id = track.get('track_id', 0)
        age = track.get('age', 0)
        
        # Màu khác nhau cho person và objects
        if class_name == 'person':
            color = (0, 255, 0)  # Xanh lá
            label = f"Person #{track_id}"
        else:
            color = (255, 0, 255)  # Magenta
            label = f"{class_name} #{track_id}"
        
        cv2.rectangle(annotated_frame, (int(x), int(y)), 
                     (int(x + w_obj), int(y + h_obj)), color, 2)
        cv2.putText(annotated_frame, label, (int(x), int(y) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Vẽ track ID
        cv2.putText(annotated_frame, f"ID:{track_id}", (int(x), int(y + h_obj + 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Vẽ hand gestures
    for gesture in hand_gestures:
        if gesture['type'] == 'pointing':
            pos = gesture['position']
            direction = gesture['direction']
            # Vẽ mũi tên chỉ hướng
            end_pos = (int(pos[0] + direction[0] * 50), int(pos[1] + direction[1] * 50))
            cv2.arrowedLine(annotated_frame, 
                           (int(pos[0]), int(pos[1])),
                           end_pos, (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(annotated_frame, "POINTING", (int(pos[0]), int(pos[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Status bar
    cv2.rectangle(annotated_frame, (10, 5), (w - 10, 60), (0, 0, 0), -1)
    cv2.putText(annotated_frame, f"Tracks: {len(tracks)}", (20, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Events: {len(interaction_events)}", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Frame count
    if fps > 0:
        time_sec = frame_count / fps
        time_text = f"Frame: {frame_count} | Time: {time_sec:.2f}s"
        cv2.putText(annotated_frame, time_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame


class InteractionAnalysisResponse(BaseModel):
    """Response model cho phân tích tương tác"""
    interaction_events: List[Dict[str, Any]] = Field(..., description="Danh sách các sự kiện tương tác được phát hiện")
    interaction_score: float = Field(..., description="Điểm tương tác (0-100, cao hơn = tương tác tốt hơn)")
    response_rate: float = Field(..., description="Tỷ lệ phản hồi (%)")
    pointing_gestures: int = Field(..., description="Số lần chỉ tay")
    object_exchanges: int = Field(..., description="Số lần trao đổi đồ vật")
    total_frames: int = Field(..., description="Tổng số frame đã phân tích")
    analyzed_duration: float = Field(..., description="Thời gian video đã phân tích (giây)")
    risk_score: float = Field(..., description="Điểm đánh giá rủi ro (0-100, cao hơn = rủi ro cao hơn)")


@router.post("/analyze", response_model=InteractionAnalysisResponse)
async def analyze_interaction(
    video: UploadFile = File(..., description="Video file để phân tích"),
    show_video: str = Form("true", description="Hiển thị video trong quá trình xử lý (true/false)")
):
    """
    Phân tích Interaction Detection từ video
    
    - Nhận diện khi cha/mẹ đưa đồ vật, trẻ có theo dõi, phản hồi, chỉ tay, đưa đồ
    - Sử dụng object detection + tracking
    - Đánh giá mức độ tương tác xã hội
    
    Args:
        video: File video (mp4, avi, mov, etc.)
    
    Returns:
        InteractionAnalysisResponse với các chỉ số phân tích
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
        
        # Load YOLO model
        yolo_net, yolo_output_layers = load_yolo_model()
        use_object_detection = yolo_net is not None
        
        # Initialize MediaPipe Hands
        use_hands = MEDIAPIPE_AVAILABLE
        hands = None
        if use_hands:
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        interaction_events = []
        pointing_count = 0
        exchange_count = 0
        parent_offers = 0
        child_responses = 0
        
        frame_count = 0
        previous_tracks = []
        previous_state = {
            'active_offers': {},
            'active_following': {},
            'active_exchanges': {}
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Object detection (mỗi 5 frames)
            detected_objects = []
            if use_object_detection and frame_count % 5 == 0:
                detected_objects = detect_objects_yolo(frame, yolo_net, yolo_output_layers)
            
            # Track objects
            tracks = track_objects(detected_objects, previous_tracks)
            previous_tracks = tracks
            
            # Hand gesture detection
            hand_gestures = []
            if use_hands:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        is_pointing, pointing_info = detect_pointing_gesture(hand_landmarks, frame.shape)
                        if is_pointing:
                            hand_gestures.append({
                                'type': 'pointing',
                                'position': pointing_info['position'],
                                'direction': pointing_info['direction']
                            })
                            
                            # Check if pointing at object
                            for obj in tracks:
                                if obj['class'] != 'person':
                                    obj_center = np.array(obj['center'])
                                    pointing_pos = pointing_info['position']
                                    distance = np.linalg.norm(pointing_pos - obj_center)
                                    obj_size = np.sqrt(obj['bbox'][2]**2 + obj['bbox'][3]**2)
                                    
                                    if distance < obj_size * 2:
                                        # Pointing at object
                                        hand_gestures.append({
                                            'type': 'pointing_at_object',
                                            'object_class': obj['class'],
                                            'object_id': obj.get('track_id', 0)
                                        })
            
            # Detect interaction events
            frame_events = detect_interaction_events(
                tracks, hand_gestures, previous_state, frame_count, fps
            )
            interaction_events.extend(frame_events)
            
            # Update counts
            for event in frame_events:
                if event['type'] == 'pointing':
                    pointing_count += 1
                elif event['type'] == 'object_exchange':
                    exchange_count += 1
                    child_responses += 1
                elif event['type'] == 'object_offer':
                    parent_offers += 1
            
            # Update previous state
            for event in frame_events:
                if event['type'] == 'object_offer':
                    obj_class = event.get('object_class', 'unknown')
                    previous_state['active_offers'][f"offer_{obj_class}"] = frame_count
                elif event['type'] == 'following':
                    obj_class = event.get('object_class', 'unknown')
                    previous_state['active_following'][f"following_{obj_class}"] = frame_count
                elif event['type'] == 'object_exchange':
                    obj_class = event.get('object_class', 'unknown')
                    previous_state['active_exchanges'][f"exchange_{obj_class}"] = frame_count
            
            # Clean old states (older than 2 seconds)
            max_age = fps * 2
            previous_state['active_offers'] = {
                k: v for k, v in previous_state['active_offers'].items() 
                if frame_count - v < max_age
            }
            previous_state['active_following'] = {
                k: v for k, v in previous_state['active_following'].items() 
                if frame_count - v < max_age
            }
            previous_state['active_exchanges'] = {
                k: v for k, v in previous_state['active_exchanges'].items() 
                if frame_count - v < max_age
            }
            
            # Visualize
            if show_video_bool:
                annotated_frame = draw_interaction_annotations(
                    frame, tracks, hand_gestures, frame_events, frame_count, fps
                )
                try:
                    cv2.imshow("Interaction Analysis - Press 'q' to quit", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        show_video_bool = False
                except cv2.error as e:
                    if "No display" in str(e) or "cannot connect" in str(e).lower():
                        logger.warning("[Interaction] Không thể hiển thị video (headless server).")
                        show_video_bool = False
                    else:
                        raise
            
            frame_count += 1
        
        # Cleanup
        if cap:
            cap.release()
        
        if use_hands and hands:
            hands.close()
        
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
                    logger.warning(f"[Interaction] Không thể xóa file temp: {temp_path}")
        
        # Tính toán kết quả
        response_rate = (child_responses / parent_offers * 100) if parent_offers > 0 else 0
        
        # Tính interaction score (dựa trên số lượng tương tác và response rate)
        interaction_density = len(interaction_events) / (frame_count / fps) if frame_count > 0 else 0  # events per second
        interaction_score = min(100, (interaction_density * 10 + response_rate * 0.5))
        
        # Tính risk score (tương tác thấp = risk cao)
        risk_score = max(0, min(100, 100 - interaction_score))
        
        analyzed_duration = frame_count / fps if fps > 0 else 0
        
        return InteractionAnalysisResponse(
            interaction_events=interaction_events,
            interaction_score=round(interaction_score, 2),
            response_rate=round(response_rate, 2),
            pointing_gestures=pointing_count,
            object_exchanges=exchange_count,
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
        if 'hands' in locals() and hands:
            try:
                hands.close()
            except:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise
    except Exception as e:
        logger.error(f"[Interaction] Lỗi khi phân tích video: {str(e)}")
        # Cleanup
        if 'cap' in locals() and cap:
            try:
                cap.release()
            except:
                pass
        if 'hands' in locals() and hands:
            try:
                hands.close()
            except:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích tương tác: {str(e)}"
        )

