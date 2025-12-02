from typing import Dict, Any
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import sys
import os

# Thêm path để import config và langchain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Cấu hình Google AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=Config.GOOGLE_AI_API_KEY,
    temperature=0.7
)

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi cần generate nội dung")


class QuestionTexts(BaseModel):
    vi: str = Field(..., description="Câu hỏi bằng tiếng Việt")
    en: str = Field(..., description="Câu hỏi bằng tiếng Anh")


class Hints(BaseModel):
    vi: str = Field(..., description="Gợi ý bằng tiếng Việt")
    en: str = Field(..., description="Gợi ý bằng tiếng Anh")


class Explanations(BaseModel):
    vi: str = Field(..., description="Giải thích bằng tiếng Việt")
    en: str = Field(..., description="Giải thích bằng tiếng Anh")


class QuestionResponse(BaseModel):
    questionTexts: QuestionTexts = Field(..., description="Nội dung câu hỏi song ngữ")
    hints: Hints = Field(..., description="Gợi ý trả lời song ngữ")
    explanations: Explanations = Field(..., description="Giải thích ý nghĩa câu hỏi song ngữ")


@router.post("/generate-question-content", response_model=QuestionResponse)
async def generate_question_content(payload: QuestionRequest):
    """
    Generate nội dung câu hỏi với format đầy đủ bao gồm:
    - questionTexts: Câu hỏi song ngữ (vi/en)
    - hints: Gợi ý trả lời song ngữ (vi/en)
    - explanations: Giải thích ý nghĩa câu hỏi song ngữ (vi/en)
    
    Nhận vào một câu hỏi và tạo ra nội dung đầy đủ theo format:
    {
        "questionTexts": {
            "vi": "...",
            "en": "..."
        },
        "hints": {
            "vi": "...",
            "en": "..."
        },
        "explanations": {
            "vi": "...",
            "en": "..."
        }
    }
    """
    try:
        # Tạo prompt cho AI để generate nội dung câu hỏi
        prompt = f"""
        Hãy tạo nội dung đầy đủ cho câu hỏi sau đây, theo định dạng song ngữ Việt-Anh.

        Câu hỏi gốc: "{payload.question}"

        Yêu cầu:
        1. **questionTexts**: Tạo lại câu hỏi một cách rõ ràng, chuyên nghiệp bằng cả tiếng Việt và tiếng Anh
        2. **hints**: Cung cấp các gợi ý cụ thể để giúp người trả lời hiểu rõ câu hỏi. Gợi ý nên:
           - Liệt kê các điểm cần xem xét
           - Đưa ra ví dụ cụ thể
           - Sử dụng bullet points với dấu "*" hoặc "-"
        3. **explanations**: Giải thích ý nghĩa và mục đích của câu hỏi, tại sao câu hỏi này quan trọng

        Định dạng trả về (JSON chính xác, không có markdown code block):
        {{
            "questionTexts": {{
                "vi": "Câu hỏi bằng tiếng Việt rõ ràng và chuyên nghiệp",
                "en": "Question in clear and professional English"
            }},
            "hints": {{
                "vi": "Hãy xem xét:\\n     * Gợi ý 1\\n     * Gợi ý 2\\n     * Gợi ý 3",
                "en": "Consider:\\n     * Hint 1\\n     * Hint 2\\n     * Hint 3"
            }},
            "explanations": {{
                "vi": "Giải thích ý nghĩa và mục đích của câu hỏi bằng tiếng Việt",
                "en": "Explanation of the question's meaning and purpose in English"
            }}
        }}

        Lưu ý:
        - Tất cả nội dung phải phù hợp với ngữ cảnh phát triển trẻ em, can thiệp sớm, hoặc giáo dục đặc biệt
        - Sử dụng ngôn ngữ chuyên nghiệp nhưng dễ hiểu
        - hints nên có ít nhất 3 điểm cần xem xét
        - Chỉ trả về JSON, không có text thêm trước hoặc sau
        """

        # Gọi AI để generate
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        if not response.content:
            raise HTTPException(status_code=500, detail="Không thể generate nội dung từ AI")
        
        # Parse response từ AI
        ai_response = response.content.strip()
        
        # Xử lý response để extract JSON
        try:
            # Loại bỏ markdown code block nếu có
            if ai_response.startswith("```json"):
                ai_response = ai_response[7:]
            if ai_response.startswith("```"):
                ai_response = ai_response[3:]
            if ai_response.endswith("```"):
                ai_response = ai_response[:-3]
            ai_response = ai_response.strip()
            
            # Tìm JSON trong response
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("Không tìm thấy JSON trong response")
            
            json_str = ai_response[start_idx:end_idx]
            
            # Parse JSON
            question_data = json.loads(json_str)
            
            # Validate structure
            required_fields = ["questionTexts", "hints", "explanations"]
            for field in required_fields:
                if field not in question_data:
                    raise ValueError(f"JSON thiếu trường: {field}")
                if "vi" not in question_data[field] or "en" not in question_data[field]:
                    raise ValueError(f"Trường {field} thiếu vi hoặc en")
            
            return QuestionResponse(
                questionTexts=QuestionTexts(**question_data["questionTexts"]),
                hints=Hints(**question_data["hints"]),
                explanations=Explanations(**question_data["explanations"])
            )
            
        except (ValueError, json.JSONDecodeError) as e:
            # Fallback: tạo nội dung đơn giản nếu không parse được JSON
            raise HTTPException(
                status_code=500, 
                detail=f"Không thể parse response từ AI: {str(e)}. Response: {ai_response[:200]}"
            )
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi generate nội dung câu hỏi: {str(exc)}")

