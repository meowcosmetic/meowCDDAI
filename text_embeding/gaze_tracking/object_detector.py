"""
Object Detector với YOLO và DeepSort Tracking - Wrapper class
Hỗ trợ cả COCO và Open Images Dataset V7 (OID)
"""
import logging
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy import OID detector
try:
    from .oid_detector import create_oid_detector, OIDDetector
    OID_DETECTOR_AVAILABLE = True
except ImportError:
    OID_DETECTOR_AVAILABLE = False
    logger.debug("[ObjectDetector] OID detector không available")

# Lazy import Custom YOLO detector (weights train lại)
try:
    from .custom_yolo_detector import CustomYOLODetector
    CUSTOM_YOLO_AVAILABLE = True
except ImportError:
    CUSTOM_YOLO_AVAILABLE = False

# Lazy import DeepSort
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("[ObjectDetector] DeepSort không được cài đặt. Object tracking sẽ bị tắt.")
    logger.warning("[ObjectDetector] Cài đặt: pip install deep-sort-realtime")

# COCO đã bị xóa - chỉ dùng OID


class ObjectDetector:
    """YOLOv8 OID Object Detector với DeepSort Tracking"""
    
    def __init__(self, config, gpu_manager, enable_tracking: bool = True):
        self.config = config
        self.gpu_manager = gpu_manager
        self.enable_tracking = enable_tracking and DEEPSORT_AVAILABLE
        self.tracker: Optional[Any] = None
        self.object_timeline: Dict[int, Dict[str, Any]] = {}  # Track objects qua frames
        
        # Detector backend (custom weights hoặc OID)
        self.oid_detector = None
        self.custom_detector = None
        self.model_name = None  # Lưu model name để sử dụng sau

        # 1) Ưu tiên custom weights nếu được cấu hình
        custom_weights = getattr(config, 'CUSTOM_YOLO_WEIGHTS', '') if config else ''
        if custom_weights:
            if not CUSTOM_YOLO_AVAILABLE:
                logger.error("[ObjectDetector] ❌ Custom YOLO detector không available!")
                logger.error("[ObjectDetector] Cài đặt: pip install ultralytics>=8.0.0")
            elif not os.path.exists(custom_weights):
                logger.error(f"[ObjectDetector] ❌ CUSTOM_YOLO_WEIGHTS không tồn tại: {custom_weights}")
            else:
                try:
                    self.custom_detector = CustomYOLODetector(
                        weights_path=custom_weights,
                        use_gpu=self.gpu_manager.is_available
                    )
                    if self.custom_detector and self.custom_detector.is_available:
                        self.model_name = self.custom_detector.model_name
                        logger.info("[ObjectDetector] ✅ Using CUSTOM YOLO weights")
                        logger.info(f"[ObjectDetector]    Model: {self.model_name}")
                        logger.info(f"[ObjectDetector]    GPU: {'Enabled' if self.gpu_manager.is_available else 'Disabled (CPU)'}")
                    else:
                        logger.error("[ObjectDetector] ❌ Không thể khởi tạo CustomYOLODetector")
                        self.custom_detector = None
                except Exception as e:
                    logger.error(f"[ObjectDetector] ❌ Lỗi khởi tạo CustomYOLODetector: {str(e)}")
                    self.custom_detector = None

        # 2) Fallback về OID detector nếu không có custom
        if self.custom_detector is None:
            if not OID_DETECTOR_AVAILABLE:
                logger.error("[ObjectDetector] ❌ OID detector không available!")
                logger.error("[ObjectDetector] Cài đặt: pip install ultralytics>=8.0.0")
                return
            try:
                oid_model_size = getattr(config, 'OID_MODEL_SIZE', 'n')
                self.oid_detector = create_oid_detector(
                    model_size=oid_model_size,
                    use_gpu=self.gpu_manager.is_available
                )
                if self.oid_detector and self.oid_detector.is_available:
                    logger.info("[ObjectDetector] ✅ OID detector initialized (có pen/pencil)")
                    self.model_name = f"yolov8{oid_model_size}-oidv7.pt"
                    logger.info(f"[ObjectDetector]    Model: {self.model_name}")
                    logger.info(f"[ObjectDetector]    Dataset: Open Images Dataset V7 (600 classes)")
                    logger.info(f"[ObjectDetector]    GPU: {'Enabled' if self.gpu_manager.is_available else 'Disabled (CPU)'}")
                else:
                    logger.error("[ObjectDetector] ❌ Không thể khởi tạo OID detector")
            except Exception as e:
                logger.error(f"[ObjectDetector] ❌ Lỗi khởi tạo OID detector: {str(e)}")
                self.oid_detector = None
        
        self._initialize_tracker()
    
    # COCO YOLO model loading đã bị xóa - chỉ dùng OID
    
    def _initialize_tracker(self) -> None:
        """Initialize DeepSort tracker"""
        if not self.enable_tracking:
            return
        
        try:
            # DeepSort parameters:
            # max_age: số frames tối đa để giữ track khi mất detection (30 frames)
            # n_init: số detections liên tiếp cần để khởi tạo track (3)
            self.tracker = DeepSort(max_age=30, n_init=3)
            logger.info("[ObjectDetector] ✅ DeepSort tracker initialized")
        except Exception as e:
            logger.warning(f"[ObjectDetector] Không thể khởi tạo DeepSort: {str(e)}")
            self.tracker = None
            self.enable_tracking = False
    
    def detect(self, frame: np.ndarray, prioritize_book: bool = True, frame_count: int = 0) -> List[Dict[str, Any]]:
        """
        Detect objects in frame và track qua các frames
        
        Args:
            frame: Input frame
            prioritize_book: Lower threshold for book detection
            frame_count: Current frame number (for tracking timeline)
        
        Returns:
            List of detected objects với track_id
        """
        # ✅ Ưu tiên custom detector nếu có
        if self.custom_detector and self.custom_detector.is_available:
            try:
                detected_objects = self.custom_detector.detect(
                    frame,
                    conf_threshold=self.config.OBJECT_CONFIDENCE_THRESHOLD
                )
                
                # Update tracker nếu có
                if self.enable_tracking and self.tracker is not None and len(detected_objects) > 0:
                    # Format cho DeepSort: list of tuples ([left, top, w, h], confidence, detection_class)
                    detections_list = []
                    for obj in detected_objects:
                        x, y, w, h = obj['bbox']
                        # DeepSort expects: ([left, top, w, h], confidence, detection_class)
                        detection_tuple = ([x, y, w, h], obj['confidence'], obj.get('class', 'unknown'))
                        detections_list.append(detection_tuple)
                    
                    tracks = self.tracker.update_tracks(detections_list, frame=frame)
                    
                    # Map tracks back to objects bằng cách match bbox gần nhất
                    import math
                    for track in tracks:
                        if track.is_confirmed():
                            track_bbox_ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
                            track_bbox = [int(track_bbox_ltrb[0]), int(track_bbox_ltrb[1]), 
                                         int(track_bbox_ltrb[2] - track_bbox_ltrb[0]), 
                                         int(track_bbox_ltrb[3] - track_bbox_ltrb[1])]
                            
                            # Tìm object gần nhất với track bbox
                            best_match_idx = None
                            min_distance = float('inf')
                            
                            for idx, obj in enumerate(detected_objects):
                                obj_bbox = obj['bbox']
                                # Tính khoảng cách giữa centers
                                obj_center = [obj_bbox[0] + obj_bbox[2] // 2, obj_bbox[1] + obj_bbox[3] // 2]
                                track_center = [track_bbox[0] + track_bbox[2] // 2, track_bbox[1] + track_bbox[3] // 2]
                                distance = math.sqrt((obj_center[0] - track_center[0])**2 + (obj_center[1] - track_center[1])**2)
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match_idx = idx
                            
                            # Nếu tìm thấy match và khoảng cách hợp lý (< 100 pixels)
                            if best_match_idx is not None and min_distance < 100:
                                detected_objects[best_match_idx]['track_id'] = track.track_id
                
                return detected_objects
            except Exception as e:
                logger.error(f"[ObjectDetector] Custom YOLO detection error: {str(e)}")
                return []

        # ✅ Fallback: dùng OID detector
        if self.oid_detector and self.oid_detector.is_available:
            try:
                detected_objects = self.oid_detector.detect(
                    frame,
                    conf_threshold=self.config.OBJECT_CONFIDENCE_THRESHOLD
                )
                
                # Update tracker nếu có
                if self.enable_tracking and self.tracker is not None and len(detected_objects) > 0:
                    # Format cho DeepSort: list of tuples ([left, top, w, h], confidence, detection_class)
                    detections_list = []
                    for obj in detected_objects:
                        x, y, w, h = obj['bbox']
                        # DeepSort expects: ([left, top, w, h], confidence, detection_class)
                        detection_tuple = ([x, y, w, h], obj['confidence'], obj.get('class', 'unknown'))
                        detections_list.append(detection_tuple)
                    
                    tracks = self.tracker.update_tracks(detections_list, frame=frame)
                    
                    # Map tracks back to objects bằng cách match bbox gần nhất
                    import math
                    for track in tracks:
                        if track.is_confirmed():
                            track_bbox_ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
                            track_bbox = [int(track_bbox_ltrb[0]), int(track_bbox_ltrb[1]), 
                                         int(track_bbox_ltrb[2] - track_bbox_ltrb[0]), 
                                         int(track_bbox_ltrb[3] - track_bbox_ltrb[1])]
                            
                            # Tìm object gần nhất với track bbox
                            best_match_idx = None
                            min_distance = float('inf')
                            
                            for idx, obj in enumerate(detected_objects):
                                obj_bbox = obj['bbox']
                                # Tính khoảng cách giữa centers
                                obj_center = [obj_bbox[0] + obj_bbox[2] // 2, obj_bbox[1] + obj_bbox[3] // 2]
                                track_center = [track_bbox[0] + track_bbox[2] // 2, track_bbox[1] + track_bbox[3] // 2]
                                distance = math.sqrt((obj_center[0] - track_center[0])**2 + (obj_center[1] - track_center[1])**2)
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match_idx = idx
                            
                            # Nếu tìm thấy match và khoảng cách hợp lý (< 100 pixels)
                            if best_match_idx is not None and min_distance < 100:
                                detected_objects[best_match_idx]['track_id'] = track.track_id
                
                return []

            except Exception as e:
                logger.error(f"[ObjectDetector] OID detection error: {str(e)}")
                return []
        
        logger.warning("[ObjectDetector] Không có detector available, không thể detect objects")
        return []
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        if self.custom_detector is not None and getattr(self.custom_detector, 'is_available', False):
            return True
        return self.oid_detector is not None and self.oid_detector.is_available
    
    def get_object_timeline(self, track_id: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        Get object tracking timeline
        
        Args:
            track_id: Specific track ID, or None for all tracks
        
        Returns:
            Dictionary of track timelines
        """
        if track_id is not None:
            return {track_id: self.object_timeline.get(track_id, {})}
        return self.object_timeline.copy()
    
    def reset_tracker(self) -> None:
        """Reset tracker và timeline (dùng khi bắt đầu video mới)"""
        if self.tracker is not None:
            self.tracker = DeepSort(max_age=30, n_init=3)
        self.object_timeline.clear()
        logger.info("[ObjectDetector] Tracker reset")

