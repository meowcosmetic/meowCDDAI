"""
Helper functions cho Gaze Analysis
"""
import numpy as np


def is_looking_at_object(child_gaze_abs_pos, object_bbox, threshold_ratio=0.6):
    """
    Kiểm tra xem trẻ có nhìn vào object không
    
    Args:
        child_gaze_abs_pos: Tuple (x, y) - vị trí gaze tuyệt đối trong frame (pixels)
        object_bbox: List [x, y, width, height] - bounding box của object
        threshold_ratio: Tỷ lệ threshold (0.6 = 60% của object size)
    
    Returns:
        bool: True nếu trẻ đang nhìn vào object
    """
    gaze_x, gaze_y = child_gaze_abs_pos
    
    # Vị trí center của object
    obj_x, obj_y, obj_w, obj_h = object_bbox
    obj_center_x = obj_x + obj_w / 2
    obj_center_y = obj_y + obj_h / 2
    
    # Tính khoảng cách từ gaze đến object center
    distance_x = abs(gaze_x - obj_center_x)
    distance_y = abs(gaze_y - obj_center_y)
    
    # Threshold: gaze phải trong vùng object (mở rộng một chút)
    threshold_x = obj_w * threshold_ratio
    threshold_y = obj_h * threshold_ratio
    
    return distance_x < threshold_x and distance_y < threshold_y


