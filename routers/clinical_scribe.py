from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from services.clinical_scribe_service import generate_clinical_medical_summary

router = APIRouter(prefix="/api/v1/clinical", tags=["Clinical AI Scribe & DSM-5 Reports"])

class ClinicalReportRequest(BaseModel):
    child_id: str = Field(..., description="ID của trẻ")
    child_name: Optional[str] = Field(default="Bé", description="Tên của trẻ")
    child_age: Optional[int] = Field(default=5, ge=1, le=18, description="Tuổi của trẻ")
    evaluation_period_days: Optional[int] = Field(default=30, ge=7, le=365, description="Chu kỳ đánh giá (ngày)")
    standard_framework: Optional[str] = Field(default="DSM_5", description="Khung tiêu chuẩn (DSM_5 hoặc ICD_11)")
    test_scores_summary: Optional[str] = Field(default=None, description="Tóm tắt điểm các bài test M-CHAT-R, ASQ-3, CDD")
    recent_meltdowns_count: Optional[int] = Field(default=0, ge=0, description="Số đợt bùng nổ trong chu kỳ")
    top_triggers: Optional[List[str]] = Field(default=None, description="Danh sách yếu tố kích hoạt giác quan")
    comfort_items: Optional[List[str]] = Field(default=None, description="Danh sách vật trấn an")
    target_skills: Optional[List[str]] = Field(default=None, description="Kỹ năng đang can thiệp")
    mastered_skills: Optional[List[str]] = Field(default=None, description="Kỹ năng đã làm chủ")
    recent_episodes: Optional[List[Dict[str, Any]]] = Field(default=None, description="Các buổi học gần nhất")

class CriterionDetail(BaseModel):
    title: str
    severity_level: str
    clinical_evidence: str
    sub_criteria_findings: Dict[str, str]

class ClinicalReportResponse(BaseModel):
    child_id: str
    child_name: str
    child_age: int
    evaluation_period_days: int
    standard_framework: str
    criterion_a: CriterionDetail
    criterion_b: CriterionDetail
    standardized_percentiles: Dict[str, str]
    doctor_recommendation_draft: str
    data_sources_used: Dict[str, Any]

@router.post("/generate-medical-summary", response_model=ClinicalReportResponse)
async def generate_medical_summary(payload: ClinicalReportRequest):
    """
    Sinh bản tóm lược lâm sàng chuẩn DSM-5 dựa trên dữ liệu phân mảnh đa nguồn.
    """
    if not payload.child_id or payload.child_id.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id cannot be empty"
        )

    summary = generate_clinical_medical_summary(
        child_id=payload.child_id,
        child_name=payload.child_name or "Bé",
        child_age=payload.child_age or 5,
        evaluation_period_days=payload.evaluation_period_days or 30,
        standard_framework=payload.standard_framework or "DSM_5",
        test_scores_summary=payload.test_scores_summary,
        recent_meltdowns_count=payload.recent_meltdowns_count or 0,
        top_triggers=payload.top_triggers,
        comfort_items=payload.comfort_items,
        target_skills=payload.target_skills,
        mastered_skills=payload.mastered_skills,
        recent_episodes=payload.recent_episodes
    )

    return summary
