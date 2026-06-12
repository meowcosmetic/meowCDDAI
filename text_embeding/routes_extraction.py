"""
Extraction Router - Proxy mỏng tới meowAI.

Ranh giới kiến trúc mới (intervention-pipeline-async):
  - TOÀN BỘ logic gọi model (search Qdrant, batching theo token, vòng lặp
    đồng thuận Extractor↔Validator, tóm tắt 2 tầng) đã được CHUYỂN sang meowAI
    và phơi bày qua endpoint cấp cao `POST /cdd/extract`.
  - meowCDDAI KHÔNG còn tái hiện agent/consensus/batching và KHÔNG gọi raw `/chat`.

Các endpoint dưới đây được giữ lại chỉ để TƯƠNG THÍCH NGƯỢC cho caller cũ còn
gọi `/extract` (Requirement 6.3):
  POST /extract            -> proxy tới meowAI `/cdd/extract`
  POST /validate-extraction -> passthrough (validation nay nằm trong /cdd/extract)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import logging

from config import Config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Extraction"])

# Timeout dài đủ cho pipeline trích lọc của meowAI (batching + consensus).
_MEOW_AI_TIMEOUT_SECONDS = 1800.0


class ExtractionRequest(BaseModel):
    raw_content: List[str] = []  # Optional: nếu rỗng meowAI sẽ tự search từ Qdrant
    intervention_goal: str


class ExtractionResponse(BaseModel):
    extracted_content: str
    confidence: float  # 0.0 - 1.0
    sources: List[dict] = []  # Thông tin nguồn tìm được


class ValidationRequest(BaseModel):
    extracted_content: str
    intervention_goal: str
    raw_content: List[str]


class ValidationResponse(BaseModel):
    is_valid: bool
    issues: List[str] = []
    corrected_content: Optional[str] = None


@router.post("/extract", response_model=ExtractionResponse)
async def extract_content(request: ExtractionRequest):
    """
    Proxy mỏng: chuyển tiếp yêu cầu trích lọc sang endpoint cấp cao
    `POST /cdd/extract` của meowAI và map kết quả về `ExtractionResponse`.

    Giữ lại để tương thích ngược cho caller cũ còn gọi `/extract`
    (Requirement 6.3). Mọi logic model (search Qdrant, batching, consensus,
    tóm tắt 2 tầng) sống trong meowAI.
    """
    if not request.intervention_goal.strip():
        raise HTTPException(status_code=400, detail="intervention_goal must not be empty")

    url = f"{Config.MEOW_AI_URL}/cdd/extract"
    payload = {
        "goal": request.intervention_goal,
        "raw_content": request.raw_content,
    }

    try:
        async with httpx.AsyncClient(timeout=_MEOW_AI_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"[EXTRACTION] meowAI /cdd/extract returned error: {e}")
        raise HTTPException(status_code=502, detail=f"meowAI extract error: {e}")
    except Exception as e:
        logger.error(f"[EXTRACTION] Failed to reach meowAI /cdd/extract: {e}")
        raise HTTPException(status_code=502, detail=f"meowAI unreachable: {e}")

    return ExtractionResponse(
        extracted_content=data.get("extracted_content", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=data.get("sources", []),
    )


@router.post("/validate-extraction", response_model=ValidationResponse)
async def validate_extraction(request: ValidationRequest):
    """
    Endpoint deprecated/passthrough.

    Việc kiểm định (validation) nay diễn ra BÊN TRONG meowAI `/cdd/extract`
    (vòng lặp Extractor↔Validator), nên không còn bước validation độc lập
    có ý nghĩa ở meowCDDAI. meowAI cũng không phơi bày endpoint validation
    riêng để proxy tới. Giữ endpoint này chỉ để tương thích ngược, luôn trả
    về kết quả hợp lệ trung tính.
    """
    logger.warning(
        "[EXTRACTION] /validate-extraction is deprecated; validation now happens "
        "inside meowAI /cdd/extract. Returning passthrough is_valid=True."
    )
    return ValidationResponse(is_valid=True, issues=[], corrected_content=None)
