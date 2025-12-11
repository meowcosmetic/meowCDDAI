"""
Response models cho Gaze Analysis
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class GazeAnalysisResponse(BaseModel):
    """Response model cho phân tích gaze"""
    eye_contact_percentage: float = Field(..., description="Phần trăm thời gian focusing vào 1 item cố định (%)")
    gaze_direction_stats: Dict[str, float] = Field(..., description="Thống kê hướng nhìn (left, right, center, up, down)")
    total_frames: int = Field(..., description="Tổng số frame đã phân tích")
    analyzed_duration: float = Field(..., description="Thời gian video đã phân tích (giây)")
    focusing_duration: float = Field(..., description="Tổng thời gian focusing (giây)")
    attention_to_person_percentage: float = Field(..., description="Phần trăm thời gian chú ý vào người tương tác (%)")
    attention_to_objects_percentage: float = Field(..., description="Phần trăm thời gian chú ý vào đồ vật (sách, bút, etc.) (%)")
    attention_to_book_percentage: float = Field(..., description="Phần trăm thời gian chú ý vào sách (%)")
    book_focusing_score: float = Field(..., description="Điểm focusing vào sách (0-100, cao hơn = focusing tốt hơn)")
    detected_objects: List[Dict[str, Any]] = Field(..., description="Danh sách các đồ vật được phát hiện")
    detected_books: List[Dict[str, Any]] = Field(..., description="Danh sách các cuốn sách được phát hiện")
    object_interaction_events: List[Dict[str, Any]] = Field(..., description="Các sự kiện tương tác với đồ vật")
    risk_score: float = Field(..., description="Điểm đánh giá rủi ro (0-100, cao hơn = rủi ro cao hơn)")
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
    object_detection_model: Optional[str] = Field(default=None, description="Tên model object detection")
    object_detection_available: bool = Field(default=False, description="Object detection có available không")

