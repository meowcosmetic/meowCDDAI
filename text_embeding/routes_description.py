from typing import List, Optional
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Thêm path để import config và langchain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Cấu hình Google AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=Config.GOOGLE_AI_API_KEY,
    temperature=0.7
)

router = APIRouter()


class DescriptionRequest(BaseModel):
    intervention_goal: str = Field(..., description="Mục tiêu can thiệp nhỏ cần tạo mô tả")


class DescriptionResponse(BaseModel):
    description: dict = Field(..., description="Mô tả song ngữ với định dạng vi/en")
    original_goal: str = Field(..., description="Mục tiêu can thiệp gốc")


@router.post("/generate-description", response_model=DescriptionResponse)
async def generate_description(payload: DescriptionRequest):
    """
    Generate mô tả song ngữ từ mục tiêu can thiệp nhỏ
    
    Nhận vào một mục tiêu can thiệp và tạo ra mô tả có định dạng:
    {
        "description": {
            "vi": "Mô tả bằng tiếng Việt",
            "en": "Description in English"
        }
    }
    """
    try:
        # Tạo prompt cho AI để generate mô tả song ngữ
        prompt = f"""
        Hãy tạo một mô tả chi tiết và chuyên nghiệp cho mục tiêu can thiệp sau đây, theo định dạng song ngữ Việt-Anh.

        Mục tiêu can thiệp: "{payload.intervention_goal}"

        Yêu cầu:
        1. Mô tả phải rõ ràng, cụ thể và có thể đo lường được
        2. Phù hợp với trẻ em đặc biệt (autism, ADHD, khuyết tật phát triển)
        3. Sử dụng ngôn ngữ chuyên nghiệp nhưng dễ hiểu
        4. Bao gồm thời gian hoặc điều kiện cụ thể nếu có thể

        Định dạng trả về (JSON):
        {{
            "vi": "Mô tả chi tiết bằng tiếng Việt",
            "en": "Detailed description in English"
        }}

        Ví dụ:
        {{
            "vi": "Trẻ có thể ngồi vững mà không cần hỗ trợ trong ít nhất 30 giây",
            "en": "Child can sit steadily without support for at least 30 seconds"
        }}
        """

        # Gọi AI để generate
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        if not response.content:
            raise HTTPException(status_code=500, detail="Không thể generate mô tả từ AI")
        
        # Parse response từ AI
        ai_response = response.content.strip()
        
        # Xử lý response để extract JSON
        try:
            # Tìm JSON trong response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Không tìm thấy JSON trong response")
            
            json_str = ai_response[start_idx:end_idx]
            
            # Parse JSON
            description_data = json.loads(json_str)
            
            # Validate structure
            if "vi" not in description_data or "en" not in description_data:
                raise ValueError("JSON không có đủ trường vi và en")
            
            return DescriptionResponse(
                description=description_data,
                original_goal=payload.intervention_goal
            )
            
        except (ValueError, json.JSONDecodeError) as e:
            # Fallback: tạo mô tả đơn giản nếu không parse được JSON
            return DescriptionResponse(
                description={
                    "vi": f"Mục tiêu can thiệp: {payload.intervention_goal}",
                    "en": f"Intervention goal: {payload.intervention_goal}"
                },
                original_goal=payload.intervention_goal
            )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi generate mô tả: {str(exc)}")
