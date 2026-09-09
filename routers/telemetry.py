from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from services.meltdown_service import evaluate_frustration_telemetry

router = APIRouter(prefix="/api/v1/telemetry", tags=["Touch Telemetry & Meltdown Warning"])

class TelemetryEvaluationRequest(BaseModel):
    child_id: str = Field(..., description="ID của trẻ")
    child_name: Optional[str] = Field(default="Bé", description="Tên của trẻ")
    window_seconds: int = Field(default=30, ge=1, le=300, description="Kích thước cửa sổ trượt (giây)")
    tap_events_count: int = Field(default=0, ge=0, description="Tổng số lần chạm trong cửa sổ")
    consecutive_fails: int = Field(default=0, ge=0, description="Số lần trả lời sai liên tiếp")
    idle_seconds: int = Field(default=0, ge=0, description="Số giây đứng yên không tương tác")
    erratic_touch_count: int = Field(default=0, ge=0, description="Số thao tác vuốt trượt loạn xạ")
    average_latency_ms: int = Field(default=0, ge=0, description="Độ trễ trung bình thao tác (ms)")
    comfort_item: Optional[str] = Field(default=None, description="Vật trấn an từ Memory Harness")

class TelemetryEvaluationResponse(BaseModel):
    frustration_score: float
    risk_level: str
    calm_mode_recommended: bool
    calm_mode_action: str
    breathing_pattern: str
    sub_scores: Dict[str, float]
    de_escalation_message: str
    comfort_item_advice: str

@router.post("/evaluate-frustration", response_model=TelemetryEvaluationResponse)
async def evaluate_frustration(payload: TelemetryEvaluationRequest):
    """
    Đánh giá dữ liệu cảm ứng theo Cửa Sổ Trượt 30 giây
    để tính Frustration Score và quyết định có kích hoạt Calm Mode hay không.
    """
    if not payload.child_id or payload.child_id.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id cannot be empty"
        )

    result = evaluate_frustration_telemetry(
        window_seconds=payload.window_seconds,
        tap_events_count=payload.tap_events_count,
        consecutive_fails=payload.consecutive_fails,
        idle_seconds=payload.idle_seconds,
        erratic_touch_count=payload.erratic_touch_count,
        average_latency_ms=payload.average_latency_ms,
        child_name=payload.child_name or "Bé",
        comfort_item=payload.comfort_item
    )

    return result
