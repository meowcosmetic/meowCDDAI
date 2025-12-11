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


def calculate_book_focusing_score(gaze_vector, book_bbox, frame_shape):
    """
    Tính focusing score dựa trên khoảng cách giữa gaze vector và book direction
    Theo công thức: if |gaze_vector - book_direction| < threshold: focusing = True
    
    Args:
        gaze_vector: Tuple (x, y) - vị trí gaze tuyệt đối
        book_bbox: List [x, y, width, height] - bounding box của sách
        frame_shape: Tuple (h, w) - kích thước frame
    
    Returns:
        float: Focusing score (0-100), 100 = đang nhìn trực tiếp vào sách
    """
    gaze_x, gaze_y = gaze_vector
    h, w = frame_shape[:2]
    
    # Vị trí center của sách
    book_x, book_y, book_w, book_h = book_bbox
    book_center_x = book_x + book_w / 2
    book_center_y = book_y + book_h / 2
    
    # Tính vector từ gaze đến book center
    dx = book_center_x - gaze_x
    dy = book_center_y - gaze_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Tính kích thước sách (diagonal)
    book_size = np.sqrt(book_w**2 + book_h**2)
    
    # Normalize distance theo kích thước sách
    normalized_distance = distance / book_size if book_size > 0 else 1.0
    
    # Focusing score: 100 nếu distance = 0, giảm dần khi distance tăng
    # Sử dụng exponential decay
    focusing_score = 100 * np.exp(-normalized_distance * 2)
    
    return max(0, min(100, focusing_score))

