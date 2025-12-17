"""
Custom YOLO Detector (Ultralytics) cho weights train lại từ dataset của bạn.

Mục tiêu: trả ra output cùng format với OIDDetector.detect() để pipeline gaze dùng lại:
  {
    'class': str,
    'class_id': int,
    'original_class': str,
    'confidence': float,
    'bbox': [x, y, w, h],
    'center': [cx, cy],
    'priority': int
  }
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class CustomYOLODetector:
    def __init__(self, weights_path: str, use_gpu: bool = True):
        self.weights_path = weights_path
        self.use_gpu = use_gpu
        self.model: Optional[Any] = None
        self.is_available: bool = False
        self.model_name: str = Path(weights_path).name

        if not ULTRALYTICS_AVAILABLE:
            logger.error("[CustomYOLO] Ultralytics chưa được cài đặt. Cài: pip install ultralytics>=8.0.0")
            return

        if not weights_path or not os.path.exists(weights_path):
            logger.error(f"[CustomYOLO] Không tìm thấy weights: {weights_path}")
            return

        try:
            self.model = YOLO(weights_path)
            device = "cuda" if use_gpu else "cpu"
            try:
                self.model.to(device)
            except Exception:
                # một số phiên bản ultralytics tự handle device trong predict/train
                pass
            self.is_available = True
            logger.info(f"[CustomYOLO] ✅ Loaded weights: {weights_path} (device={device})")
        except Exception as e:
            logger.error(f"[CustomYOLO] ❌ Không thể load weights: {weights_path} - {e}")
            self.model = None
            self.is_available = False

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        if not self.is_available or self.model is None:
            return []

        try:
            # ultralytics YOLO expects RGB
            rgb_frame = frame[:, :, ::-1]
            results = self.model(rgb_frame, conf=conf_threshold, verbose=False)
            if not results:
                return []

            result = results[0]
            detected_objects: List[Dict[str, Any]] = []

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names or {}

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = float(scores[i])
                class_id = int(class_ids[i])
                original_class = class_names.get(class_id, f"class_{class_id}")

                # chuẩn hoá tên class
                simple_name = str(original_class).strip().lower()

                # filter an toàn (nếu dataset lỡ có person)
                if simple_name in {"person", "human", "people"}:
                    continue

                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                if w <= 1 or h <= 1:
                    continue

                detected_objects.append(
                    {
                        "class": simple_name,
                        "class_id": class_id,
                        "original_class": original_class,
                        "confidence": round(confidence, 2),
                        "bbox": [x, y, w, h],
                        "center": [x + w // 2, y + h // 2],
                        "priority": 100 if simple_name == "book" else (30 if simple_name in ["pen", "pencil"] else 10),
                    }
                )

            return detected_objects
        except Exception as e:
            logger.error(f"[CustomYOLO] Detection error: {e}")
            return []


