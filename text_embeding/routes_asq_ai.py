"""
ASQ-3 AI FastAPI Endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from asq_ai_service import AsqAiService

router = APIRouter(prefix="/api/v1/ai/asq", tags=["ASQ-3 AI Assessment"])


class AsqPredictRequest(BaseModel):
    child_id: Optional[int] = None
    age_interval_months: int = Field(default=12, ge=2, le=60)
    is_premature: bool = False
    domain_scores: Dict[str, float] = Field(
        ...,
        example={
            "COMMUNICATION": 45.0,
            "GROSS_MOTOR": 50.0,
            "FINE_MOTOR": 35.0,
            "PROBLEM_SOLVING": 40.0,
            "PERSONAL_SOCIAL": 45.0
        }
    )


class QuickScreenRequest(BaseModel):
    age_interval_months: int = Field(default=12, ge=2, le=60)


@router.post("/predict")
def predict_asq_risk(req: AsqPredictRequest):
    """
    Dự đoán xác suất nguy cơ, bách phân vị và mẫu bất thường từ điểm số ASQ-3.
    """
    try:
        result = AsqAiService.predict_asq_risk(req.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quick-screen")
def get_quick_screen_items(req: QuickScreenRequest):
    """
    Lấy danh sách 10 câu hỏi có độ phân biệt cao nhất cho chế độ làm bài test nhanh thích ứng.
    """
    try:
        items = AsqAiService.get_quick_screen_items(req.age_interval_months)
        return {
            "age_interval_months": req.age_interval_months,
            "total_items": len(items),
            "discriminator_items": items
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
def health_check():
    return {"status": "ok", "service": "asq3_ai_engine", "version": "1.0.0"}
