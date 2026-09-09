import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from services.worksheet_analysis_service import WorksheetAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/vision",
    tags=["Computer-Assisted Worksheet Analysis"]
)

worksheet_service = WorksheetAnalysisService()

@router.post("/analyze-worksheet")
async def analyze_worksheet(
    file: UploadFile = File(...),
    worksheet_type: str = Form("TRACING"),
    child_id: Optional[str] = Form(None)
):
    """
    POST /api/v1/vision/analyze-worksheet
    
    Hỗ trợ phân tích kỹ thuật số bài tập và nét vẽ của trẻ em (Worksheet & Drawing Analysis).
    Đo lường các thuộc tính vật lý khách quan phục vụ rèn luyện vận động tinh:
    - Độ ổn định nét vẽ (Stroke Stability / Tremor Level)
    - Độ tuân thủ đường viền (Boundary Compliance)
    - Tỷ lệ hoàn thành bài tập (Worksheet Completion Rate)
    - Bảng màu sắc & gợi ý sư phạm rèn luyện cơ tay
    
    Lưu ý: Không đưa ra chẩn đoán cảm xúc hay tâm lý võ đoán.
    """
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="File hinh anh bai tap khong duoc de trong")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="File anh rong (0 bytes)")

        result = worksheet_service.analyze(image_bytes, worksheet_type=worksheet_type)
        if child_id:
            result["child_id"] = child_id

        return result
    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"Loi du lieu anh bai tap: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Loi he thong khi phan tich bai tap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Loi he thong khi phan tich anh: {str(e)}")
