from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.social_story_service import SocialStoryService

router = APIRouter(
    prefix="/api/v1/stories",
    tags=["Social Stories Generator"],
)


class GenerateStoryRequest(BaseModel):
    child_id: Optional[str] = Field(None, description="Mã định danh bé để kết nối với Child Memory Harness")
    child_name: Optional[str] = Field("Bé", description="Tên thường gọi của bé")
    child_age: Optional[int] = Field(5, description="Tuổi của bé")
    situation_category: str = Field(
        ...,
        description="Mã tình huống: DENTAL_VISIT, VACCINATION, HAIRCUT, SHARING_TOYS, SUPERMARKET_CROWD, NEW_SCHOOL, ANGER_REGULATION, WAITING_TURN hoặc CUSTOM",
    )
    custom_context: Optional[str] = Field(None, description="Bối cảnh riêng hoặc lo lắng cụ thể của bé")
    target_issue: Optional[str] = Field(None, description="Bé đang gặp vấn đề gì (thừa nhận cảm xúc)")
    specific_trigger: Optional[str] = Field(None, description="Tình huống nào là trigger kích hoạt cơn hoảng sợ")
    sensory_reason: Optional[str] = Field(None, description="Vì sao bé gặp khó khăn (giải thích giác quan/nhận thức)")
    coping_skill: Optional[str] = Field(None, description="Bé cần học kỹ năng nào để tự điều hòa cảm xúc")
    interests: Optional[List[str]] = Field(default=None, description="Sở thích của bé (dùng làm bạn đồng hành)")
    favorite_characters: Optional[List[str]] = Field(default=None, description="Nhân vật yêu thích")
    comfort_item: Optional[str] = Field(None, description="Đồ vật mang lại cảm giác an tâm cho bé")
    triggers_to_avoid: Optional[List[str]] = Field(default=None, description="Yếu tố gây hoảng sợ cần né tránh")


class EvaluateFormulaRequest(BaseModel):
    pages: List[Dict[str, Any]] = Field(..., description="Danh sách các trang truyện cần thẩm định tỷ lệ Carol Gray")


@router.get("/templates", response_model=List[Dict[str, Any]])
async def get_social_story_templates():
    """
    Lấy danh mục 8 tình huống can thiệp xã hội mẫu chuẩn mực.
    """
    return SocialStoryService.get_templates()


@router.post("/generate", response_model=Dict[str, Any])
async def generate_social_story(req: GenerateStoryRequest):
    """
    Tạo kịch bản sách tranh xã hội cá nhân hóa theo chuẩn Carol Gray (1991),
    tự động tính toán tỷ lệ Ratio >= 2.0, gắn thẻ cảnh AAC Pastel và câu hỏi thấu cảm.
    """
    if not req.situation_category or not req.situation_category.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="situation_category không được để trống",
        )

    # Nếu có child_id, thử tích hợp ngữ cảnh từ child_harness_service
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    if req.child_id:
        try:
            from services.child_harness_service import child_harness_service

            harness_context = child_harness_service.get_context(req.child_id)
            if harness_context:
                if not payload.get("interests") and harness_context.get("dominant_interests"):
                    payload["interests"] = harness_context["dominant_interests"]
                if not payload.get("triggers_to_avoid") and harness_context.get("triggers_to_avoid"):
                    payload["triggers_to_avoid"] = harness_context["triggers_to_avoid"]
                if not payload.get("child_name") or payload["child_name"] == "Bé":
                    if harness_context.get("child_name"):
                        payload["child_name"] = harness_context["child_name"]
        except Exception:
            pass

    return SocialStoryService.generate_social_story(payload)


@router.post("/evaluate-formula", response_model=Dict[str, Any])
async def evaluate_carol_gray_formula(req: EvaluateFormulaRequest):
    """
    Thẩm định kịch bản truyện có tuân thủ tỷ lệ vàng Carol Gray Ratio (>= 2.0) hay không.
    """
    if not req.pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Danh sách trang truyện (pages) không được để trống",
        )
    return SocialStoryService.evaluate_carol_gray_formula(req.pages)
