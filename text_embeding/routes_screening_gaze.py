"""
Gaze Analysis Routes - DEPRECATED
File này đã được refactor thành text_embeding/gaze/ module
Vui lòng sử dụng text_embeding.gaze.routes thay vì file này
"""
import logging

logger = logging.getLogger(__name__)

# Import router từ gaze module mới
from .gaze.routes import router

# Giữ lại để backward compatibility
# Tất cả logic đã được chuyển sang text_embeding/gaze/

__all__ = ['router']
