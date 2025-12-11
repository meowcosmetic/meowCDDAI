"""
Pydantic Models cho Gaze Analysis
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GazeAnalysisResponse(BaseModel):
    """Response model cho Gaze Analysis"""
    eye_contact_percentage: float = Field(..., description="Phần trăm thời gian có eye contact (%)")
    attention_to_person_percentage: float = Field(..., description="Phần trăm thời gian chú ý đến người (%)")
    attention_to_objects_percentage: float = Field(..., description="Phần trăm thời gian chú ý đến đồ vật (%)")
    attention_to_book_percentage: float = Field(..., description="Phần trăm thời gian chú ý đến sách (%)")
    focusing_duration: float = Field(..., description="Thời gian focusing vào một item cố định (giây)")
    book_focusing_score: float = Field(..., description="Điểm focusing vào sách (0-1)")
    detected_objects: Dict[str, int] = Field(default_factory=dict, description="Số lượng objects được detect")
    detected_books: List[Dict[str, Any]] = Field(default_factory=list, description="Danh sách sách được detect")
    object_interaction_events: List[Dict[str, Any]] = Field(default_factory=list, description="Các sự kiện tương tác với objects")
    risk_score: float = Field(..., description="Điểm đánh giá rủi ro (0-100)")
    # Timeline Analysis
    focus_timeline: List[Dict[str, Any]] = Field(default_factory=list, description="Timeline các focus periods")
    object_focus_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Thống kê focus cho từng object")
    pattern_analysis: Dict[str, Any] = Field(default_factory=dict, description="Phân tích pattern (có quay lại nhìn object cũ không)")
    # Gaze Wandering Detection
    gaze_wandering_score: float = Field(default=0.0, description="Điểm 'nhìn vô định' (0-100, cao hơn = nhìn vô định nhiều hơn)")
    gaze_wandering_percentage: float = Field(default=0.0, description="Phần trăm thời gian 'nhìn vô định' (%)")
    wandering_periods: List[Dict[str, Any]] = Field(default_factory=list, description="Các khoảng thời gian 'nhìn vô định'")
    # Fatigue Detection
    fatigue_score: float = Field(default=0.0, description="Điểm mệt mỏi (0-100, cao hơn = mệt mỏi hơn)")
    fatigue_level: str = Field(default="low", description="Mức độ mệt mỏi: low, medium, high")
    fatigue_indicators: Dict[str, Any] = Field(default_factory=dict, description="Các chỉ số chi tiết về mệt mỏi")
    # Focus Level
    focus_level: float = Field(default=0.0, description="Mức độ tập trung dựa trên mắt + đầu (0-100, cao hơn = tập trung tốt hơn)")
    focus_level_details: Dict[str, Any] = Field(default_factory=dict, description="Chi tiết về focus level")
    # Model information
    object_detection_model: Optional[str] = Field(None, description="Model object detection đang sử dụng (ví dụ: 'YOLOv8 OID')")
    object_detection_available: bool = Field(default=False, description="Object detection có available không")


class FaceInfo(BaseModel):
    """Thông tin về một khuôn mặt"""
    bbox: List[float] = Field(..., description="Bounding box [x, y, w, h]")
    landmarks: Optional[List[List[float]]] = Field(None, description="Facial landmarks")
    is_child: bool = Field(False, description="Có phải là trẻ em không")
    is_adult: bool = Field(False, description="Có phải là người lớn không")
    gaze_vector: Optional[List[float]] = Field(None, description="Gaze direction vector")
    confidence: float = Field(1.0, description="Confidence score")


class DetectedObject(BaseModel):
    """Thông tin về một object được detect"""
    class_name: str = Field(..., description="Tên class")
    bbox: List[float] = Field(..., description="Bounding box [x, y, w, h]")
    confidence: float = Field(..., description="Confidence score")
    center: List[float] = Field(..., description="Center point [x, y]")
    track_id: Optional[int] = Field(None, description="Track ID từ DeepSort")
    gaze_confidence: Optional[float] = Field(None, description="Confidence score từ 3D gaze estimation (0-1)")

