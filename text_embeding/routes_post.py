from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Thêm path để import ai_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_agents import InterventionProcessor


router = APIRouter()


class CreatePostRequest(BaseModel):
    title: str = Field(..., description="Tiêu đề bài viết")
    content: str = Field(..., description="Nội dung bài viết")
    tags: Optional[List[str]] = Field(default=None, description="Danh sách tags")


class InterventionGoalRequest(BaseModel):
    intervention_goal: str = Field(..., description="Mục tiêu can thiệp cho trẻ đặc biệt")
    title: Optional[str] = Field(default=None, description="Tiêu đề tùy chọn cho bài viết")
    book_content: Optional[List[str]] = Field(default=None, description="Mảng nội dung sách liên quan để làm context")


@router.post("/create-post", response_model=dict)
async def create_post(payload: CreatePostRequest):
    """Tạo bài viết mới (stub). Có thể mở rộng để lưu trữ sau."""
    try:
        # Stub: chỉ phản hồi lại dữ liệu đã nhận kèm id tạm thời
        post_id = f"post_{abs(hash(payload.title + payload.content)) % (10 ** 8)}"
        return {
            "message": "Tạo bài viết thành công",
            "post": {
                "id": post_id,
                "title": payload.title,
                "content": payload.content,
                "tags": payload.tags or [],
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo bài viết: {str(exc)}")


@router.post("/process-intervention-goal", response_model=dict)
async def process_intervention_goal(payload: InterventionGoalRequest):
    """
    Xử lý mục tiêu can thiệp cho trẻ đặc biệt qua 4 bước workflow
    
    Workflow:
    1. ExpertAgent: Phân tích, tạo khung lý thuyết
    2. PracticalAgent: Thêm ví dụ, checklist thực tế
    3. VerifierAgent: Kiểm chứng, bổ sung nguồn tham khảo (trên nội dung từ bước 1 + 2)
    4. EditorAgent (duy nhất): Gom tất cả nội dung và biên tập dễ hiểu + format (Markdown/HTML)
    """
    try:
        # Khởi tạo processor
        processor = InterventionProcessor()
        
        # Tạo context từ book_content nếu có
        context = {}
        if payload.book_content and len(payload.book_content) > 0:
            context["book_content"] = payload.book_content
        
        # Xử lý mục tiêu can thiệp qua 4 agents
        result = processor.process_intervention_goal(payload.intervention_goal, context)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý: {result['error']}")
        
        # Tạo post_id
        post_id = f"intervention_{abs(hash(payload.intervention_goal)) % (10 ** 8)}"
        
        # Tạo title nếu không được cung cấp
        title = payload.title or f"Can thiệp: {payload.intervention_goal[:50]}..."
        
        return {
            "message": "Xử lý mục tiêu can thiệp thành công",
            "post_id": post_id,
            "title": title,
            "original_goal": result["original_goal"],
            "processing_results": {
                "expert_analysis": result["expert_analysis"],
                "practical_content": result["practical_content"],
                "verified_content": result["verified_content"],
                "final_content": result.get("final_content", "")
            },
            "workflow_summary": {
                "step_1": "ExpertAgent đã phân tích và tạo khung lý thuyết",
                "step_2": "PracticalAgent đã thêm ví dụ và checklist thực tế",
                "step_3": "VerifierAgent đã kiểm chứng và bổ sung nguồn tham khảo",
                "step_4": "EditorAgent đã gom tất cả nội dung và biên tập + format"
            },
            "final_content": result.get("final_content", "")
        }
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý mục tiêu can thiệp: {str(exc)}")



