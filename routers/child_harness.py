from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from services.child_harness_service import child_harness_service, ChildHarnessProfile

router = APIRouter(
    prefix="/api/v1/harness",
    tags=["Child Memory Harness"],
)


class UpdateProfileRequest(BaseModel):
    child_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = None
    dominant_interests: Optional[List[str]] = None
    triggers_to_avoid: Optional[List[str]] = None
    mastered_skills: Optional[List[str]] = None
    target_skills: Optional[List[str]] = None


class AddMemoryRequest(BaseModel):
    memory_type: str = Field(..., description="interest, trigger, target_skill, mastered_skill")
    content: str = Field(..., description="Nội dung chi tiết")


class SyncTestResultRequest(BaseModel):
    test_name: str = Field(..., description="Tên bài test (ví dụ: ASQ-3 48 Tháng, CDD Test)")
    category_scores: List[Dict[str, Any]] = Field(default_factory=list, description="Danh sách điểm các lĩnh vực")
    sensory_triggers: Optional[List[str]] = Field(default=None, description="Các phản ứng nhạy cảm trích xuất được")


@router.get("/{child_id}/context", response_model=Dict[str, Any])
async def get_child_harness_context(child_id: str):
    """
    Lấy context cá nhân hóa của trẻ (bao gồm Prompt Injection Snippet)
    để tiêm vào System Prompt của các AI Agent, Chatbot hoặc Game Generator.
    """
    if not child_id or not child_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id không được để trống",
        )
    return child_harness_service.get_context(child_id)


@router.get("/{child_id}/profile", response_model=ChildHarnessProfile)
async def get_child_harness_profile(child_id: str):
    """
    Lấy toàn bộ hồ sơ trí nhớ 4 tầng của trẻ.
    """
    if not child_id or not child_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id không được để trống",
        )
    return child_harness_service.get_or_create_profile(child_id)


@router.post("/{child_id}/profile", response_model=ChildHarnessProfile)
async def update_child_harness_profile(child_id: str, req: UpdateProfileRequest):
    """
    Khởi tạo hoặc cập nhật các trường thông tin trong hồ sơ trí nhớ của bé.
    """
    if not child_id or not child_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id không được để trống",
        )

    profile = child_harness_service.get_or_create_profile(
        child_id=child_id,
        child_name=req.child_name,
        age=req.age,
        diagnosis=req.diagnosis,
    )

    if req.gender is not None:
        profile.gender = req.gender
    if req.dominant_interests is not None:
        profile.dominant_interests = req.dominant_interests
    if req.triggers_to_avoid is not None:
        profile.triggers_to_avoid = req.triggers_to_avoid
    if req.mastered_skills is not None:
        profile.mastered_skills = req.mastered_skills
    if req.target_skills is not None:
        profile.target_skills = req.target_skills

    profile.prompt_injection_snippet = child_harness_service.generate_prompt_snippet(profile)
    return profile


@router.post("/{child_id}/add-memory", response_model=ChildHarnessProfile)
async def add_child_memory_note(child_id: str, req: AddMemoryRequest):
    """
    Thêm nhanh một mẩu sở thích, trigger hoặc ghi chú mới vào trí nhớ AI của bé.
    """
    if not child_id or not child_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id không được để trống",
        )
    try:
        return child_harness_service.add_memory(
            child_id=child_id,
            memory_type=req.memory_type,
            content=req.content,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{child_id}/sync-test-result", response_model=ChildHarnessProfile)
async def sync_child_test_result(child_id: str, req: SyncTestResultRequest):
    """
    Kích hoạt tự động đồng bộ kết quả bài test (ASQ-3/CDD) vào Child Memory Harness.
    - Lĩnh vực cần can thiệp (FAIL/RED) -> nạp vào target_skills.
    - Lĩnh vực đạt chuẩn (PASS/WHITE) -> nạp vào mastered_skills.
    - Sensory triggers -> nạp vào triggers_to_avoid.
    """
    if not child_id or not child_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="child_id không được để trống",
        )
    return child_harness_service.sync_test_result(
        child_id=child_id,
        test_name=req.test_name,
        category_scores=req.category_scores,
        sensory_triggers=req.sensory_triggers,
    )
