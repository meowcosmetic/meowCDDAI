"""
Open Images Dataset V7 (OID) Object Detector
Sử dụng YOLOv8 từ Ultralytics - có sẵn weights cho OID với 600 classes
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

# Lazy import YOLOv8
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("[OID] Ultralytics không được cài đặt. Cài: pip install ultralytics")

# Open Images Dataset V7 classes (600 classes)
# Lấy từ: https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
# Chỉ lấy các classes phổ biến liên quan đến screening
OID_CLASSES = [
    'Person', 'Man', 'Woman', 'Boy', 'Girl', 'Child', 'Baby',
    'Book', 'Pen', 'Pencil', 'Marker', 'Crayon',  # ✅ Có Pen và Pencil!
    'Cup', 'Mug', 'Bottle', 'Glass', 'Bowl', 'Plate',
    'Cell phone', 'Mobile phone', 'Laptop', 'Computer', 'Tablet computer',
    'Mouse', 'Keyboard', 'Remote control',
    'Toy', 'Doll', 'Teddy bear', 'Ball', 'Building blocks',
    'Chair', 'Table', 'Desk', 'Bed', 'Couch', 'Sofa',
    'Scissors', 'Knife', 'Fork', 'Spoon',
    'Clock', 'Watch', 'Vase', 'Flower', 'Plant',
    'Food', 'Apple', 'Banana', 'Orange', 'Sandwich', 'Pizza',
    'Clothing', 'Shirt', 'Pants', 'Shoe', 'Hat',
    'Vehicle', 'Car', 'Bicycle', 'Motorcycle',
    'Animal', 'Dog', 'Cat', 'Bird', 'Horse',
    # ... và nhiều classes khác
]

# Map OID class names to simpler names
OID_CLASS_MAP = {
    'Person': 'person',
    'Man': 'person',
    'Woman': 'person',
    'Boy': 'person',
    'Girl': 'person',
    'Child': 'person',
    'Baby': 'person',
    'Book': 'book',
    'Pen': 'pen',  # ✅
    'Pencil': 'pencil',  # ✅
    'Marker': 'pen',
    'Crayon': 'pen',
    'Cup': 'cup',
    'Mug': 'cup',
    'Bottle': 'bottle',
    'Glass': 'cup',
    'Cell phone': 'cell phone',
    'Mobile phone': 'cell phone',
    'Laptop': 'laptop',
    'Computer': 'laptop',
    'Tablet computer': 'laptop',
    'Mouse': 'mouse',
    'Keyboard': 'keyboard',
    'Remote control': 'remote',
    'Toy': 'toy',
    'Doll': 'toy',
    'Teddy bear': 'teddy bear',
    'Scissors': 'scissors',
    'Knife': 'knife',
    'Fork': 'fork',
    'Spoon': 'spoon',
}

# Classes to exclude (person, face, and clothing related)
EXCLUDED_CLASSES = {
    # Person related
    'Person', 'Man', 'Woman', 'Boy', 'Girl', 'Child', 'Baby',
    'Human face', 'Face', 'Head', 'Human head',
    'Human body', 'Human hand', 'Human arm', 'Human leg',
    'Human foot', 'Human hair', 'Human eye', 'Human mouth',
    'Human nose', 'Human ear', 'Human neck', 'Human torso',
    # Clothing related
    'Clothing', 'Shirt', 'Pants', 'Shoe', 'Hat', 'Dress', 'Jacket', 'Coat',
    'Suit', 'Tie', 'Belt', 'Sock', 'Glove', 'Boot', 'Sneaker', 'Sandal',
    'Skirt', 'Shorts', 'Jeans', 'Trousers', 'T-shirt', 'Blouse', 'Sweater',
    'Coat', 'Jacket', 'Vest', 'Scarf', 'Cap', 'Helmet', 'Uniform', 'Costume'
}


class OIDDetector:
    """
    Open Images Dataset V7 Object Detector
    Sử dụng YOLOv8 từ Ultralytics
    """
    
    def __init__(self, model_size: str = 'n', use_gpu: bool = True):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            use_gpu: Sử dụng GPU nếu có
        """
        self.model = None
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.is_available = False
        
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("[OID] Ultralytics không available, không thể sử dụng OID detector")
            return
        
        try:
            # YOLOv8 OID model - CHỈ sử dụng OID, không fallback về COCO
            # Hỗ trợ cả hai tên: oidv7 và oiv7 (Open Images Dataset V7)
            model_name_oidv7 = f"yolov8{model_size}-oidv7.pt"
            model_name_oiv7 = f"yolov8{model_size}-oiv7.pt"
            
            logger.info(f"[OID] Đang load YOLOv8 OID model...")
            logger.info(f"[OID] Thử tên: {model_name_oidv7}")
            
            # Thử load với tên oidv7 trước
            model_loaded = False
            model_name = None
            try:
                self.model = YOLO(model_name_oidv7)
                logger.info(f"[OID] ✅ Loaded OID model: {model_name_oidv7}")
                model_name = model_name_oidv7
                model_loaded = True
            except (FileNotFoundError, Exception) as e1:
                # Thử với tên oiv7 (viết tắt của Open Images V7)
                logger.info(f"[OID] Không tìm thấy {model_name_oidv7}, thử {model_name_oiv7}")
                try:
                    self.model = YOLO(model_name_oiv7)
                    logger.info(f"[OID] ✅ Loaded OID model: {model_name_oiv7}")
                    model_name = model_name_oiv7
                    model_loaded = True
                except (FileNotFoundError, Exception) as e2:
                    # Không fallback về COCO - chỉ báo lỗi và hướng dẫn
                    logger.error(f"[OID] ❌ Không thể load OID model!")
                    logger.error(f"[OID]    Đã thử: {model_name_oidv7} - Lỗi: {str(e1)}")
                    logger.error(f"[OID]    Đã thử: {model_name_oiv7} - Lỗi: {str(e2)}")
                    logger.error(f"[OID]    ⚠️  Không tìm thấy OID model!")
                    logger.error(f"[OID]    Để sử dụng OID detector, bạn cần:")
                    logger.error(f"[OID]    1. Đặt model vào: ~/.ultralytics/weights/")
                    logger.error(f"[OID]    2. Tên file phải là: yolov8{model_size}-oidv7.pt hoặc yolov8{model_size}-oiv7.pt")
                    logger.error(f"[OID]    3. Hoặc train model từ Open Images Dataset V7")
                    raise FileNotFoundError(f"OID model không tồn tại. Đã thử: {model_name_oidv7} và {model_name_oiv7}")
            
            if not model_loaded:
                raise FileNotFoundError("Không thể load OID model")
            
            # Set device
            device = 'cuda' if use_gpu else 'cpu'
            self.model.to(device)
            
            # Get model info
            try:
                model_info = self.model.info() if hasattr(self.model, 'info') else {}
                model_params = getattr(self.model, 'model', None)
                if model_params:
                    total_params = sum(p.numel() for p in model_params.parameters()) if hasattr(model_params, 'parameters') else 0
                else:
                    total_params = 0
            except:
                model_info = {}
                total_params = 0
            
            self.is_available = True
            logger.info(f"[OID] ✅ YOLOv8 OID model loaded")
            logger.info(f"[OID]    Model: {model_name}")
            logger.info(f"[OID]    Device: {device}")
            logger.info(f"[OID]    Size: {model_size.upper()} ({'nano' if model_size == 'n' else 'small' if model_size == 's' else 'medium' if model_size == 'm' else 'large' if model_size == 'l' else 'xlarge'})")
            logger.info(f"[OID]    Dataset: Open Images Dataset V7 (600 classes)")
            logger.info(f"[OID]    Classes: Person, Book, Pen, Pencil, Cup, Bottle, etc.")
            if total_params > 0:
                logger.info(f"[OID]    Parameters: {total_params:,}")
        except ImportError as e:
            logger.error(f"[OID] ❌ Ultralytics chưa được cài đặt!")
            logger.error(f"[OID]    Lỗi: {str(e)}")
            logger.error(f"[OID]    Vui lòng cài đặt: pip install ultralytics>=8.0.0")
            logger.error(f"[OID]    Hoặc chạy: install_oid_detector.bat")
            self.model = None
            self.is_available = False
        except FileNotFoundError as e:
            logger.error(f"[OID] ❌ Không tìm thấy OID model file: {str(e)}")
            logger.error(f"[OID]    ⚠️  Ultralytics không có OID model sẵn!")
            logger.error(f"[OID]    Để sử dụng OID detector, bạn cần:")
            logger.error(f"[OID]    1. Train model từ Open Images Dataset V7")
            logger.error(f"[OID]    2. Hoặc download model đã train từ nguồn khác")
            logger.error(f"[OID]    3. Đặt model vào: ~/.ultralytics/weights/yolov8{model_size}-oidv7.pt")
            self.model = None
            self.is_available = False
        except Exception as e:
            logger.error(f"[OID] ❌ Không thể load YOLOv8 OID model: {str(e)}")
            logger.error(f"[OID]    ⚠️  Ultralytics không có OID model sẵn!")
            logger.error(f"[OID]    Để sử dụng OID detector, bạn cần:")
            logger.error(f"[OID]    1. Train model từ Open Images Dataset V7")
            logger.error(f"[OID]    2. Hoặc download model đã train từ nguồn khác")
            logger.error(f"[OID]    3. Đặt model vào: ~/.ultralytics/weights/yolov8{model_size}-oidv7.pt")
            logger.error(f"[OID]    ")
            logger.error(f"[OID]    Hướng dẫn train OID model:")
            logger.error(f"[OID]    - Sử dụng ultralytics để train trên OID dataset")
            logger.error(f"[OID]    - Hoặc tìm model đã train sẵn từ cộng đồng")
            self.model = None
            self.is_available = False
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Detect objects trong frame
        
        Args:
            frame: Input frame (BGR format)
            conf_threshold: Confidence threshold
        
        Returns:
            List of detected objects với format:
            {
                'class': str,
                'bbox': [x, y, w, h],
                'confidence': float,
                'center': [x, y],
                'track_id': Optional[int]
            }
        """
        if not self.is_available or self.model is None:
            return []
        
        try:
            # YOLOv8 expects RGB
            rgb_frame = frame[:, :, ::-1]  # BGR to RGB
            
            # Run inference
            results = self.model(rgb_frame, conf=conf_threshold, verbose=False)
            
            detected_objects = []
            
            if len(results) > 0:
                result = results[0]  # First result
                
                # Get boxes, scores, class_ids
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Get class names
                class_names = result.names
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = float(scores[i])
                    class_id = int(class_ids[i])
                    
                    # Get class name
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    # Skip person, face, and clothing related classes
                    if class_name in EXCLUDED_CLASSES:
                        continue
                    
                    # Also check if class name contains keywords (case insensitive)
                    class_name_lower = class_name.lower()
                    excluded_keywords = [
                        'person', 'face', 'human', 'head', 'man', 'woman', 'boy', 'girl', 'child', 'baby',
                        'clothing', 'clothes', 'shirt', 'pants', 'shoe', 'hat', 'dress', 'jacket', 'coat',
                        'suit', 'tie', 'belt', 'sock', 'glove', 'boot', 'sneaker', 'sandal', 'skirt', 'shorts',
                        'jeans', 'trousers', 't-shirt', 'blouse', 'sweater', 'vest', 'scarf', 'cap', 'helmet',
                        'uniform', 'costume', 'apparel', 'garment', 'outfit', 'wear'
                    ]
                    if any(keyword in class_name_lower for keyword in excluded_keywords):
                        continue
                    
                    # Map to simpler name
                    simple_name = OID_CLASS_MAP.get(class_name, class_name.lower())
                    
                    # Skip nếu mapped name là 'person'
                    if simple_name == 'person':
                        continue
                    
                    # Convert to [x, y, w, h] format
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    detected_objects.append({
                        'class': simple_name,
                        'class_id': class_id,
                        'original_class': class_name,
                        'confidence': round(confidence, 2),
                        'bbox': [x, y, w, h],
                        'center': [x + w // 2, y + h // 2],
                        'priority': 100 if simple_name == 'book' else (30 if simple_name in ['pen', 'pencil'] else 10)
                    })
            
            return detected_objects
        except Exception as e:
            logger.error(f"[OID] Detection error: {str(e)}")
            return []


def create_oid_detector(model_size: str = 'n', use_gpu: bool = True) -> Optional[OIDDetector]:
    """
    Factory function để tạo OID detector
    
    Args:
        model_size: 'n' (nano - fastest), 's', 'm', 'l', 'x' (most accurate)
        use_gpu: Sử dụng GPU nếu có
    
    Returns:
        OIDDetector instance hoặc None nếu không available
    """
    if not ULTRALYTICS_AVAILABLE:
        return None
    
    try:
        detector = OIDDetector(model_size=model_size, use_gpu=use_gpu)
        if detector.is_available:
            return detector
        return None
    except Exception as e:
        logger.warning(f"[OID] Không thể tạo OID detector: {str(e)}")
        return None

