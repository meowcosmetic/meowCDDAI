"""
Speech Assessment Router for meowAI.
Provides endpoints for pronunciation scoring, phoneme analysis, and speech therapy feedback.
"""

import logging
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.pronunciation_service import PronunciationResult, pronunciation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/speech", tags=["Speech Assessment"])


@router.post("/assess-pronunciation", response_model=PronunciationResult)
async def assess_child_pronunciation(
    file: UploadFile = File(..., description="File âm thanh bé đọc (wav, webm, mp3, m4a, ogg)"),
    reference_text: str = Form(..., description="Từ hoặc câu mẫu chuẩn cần đọc (ví dụ: 'con cá', 'quả táo')"),
    language: str = Form("vi", description="Mã ngôn ngữ ISO (mặc định: vi)"),
) -> PronunciationResult:
    """
    Đánh giá và chấm điểm phát âm tiếng Việt cho trẻ em.
    Phân tích âm vị học, thanh điệu (sắc/huyền/hỏi/ngã/nặng), phụ âm đầu, phát hiện lỗi ngọng và đưa ra nhận xét sư phạm.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=400,
            detail={"error": "NO_FILE_PROVIDED", "message": "Vui lòng cung cấp file âm thanh"}
        )

    if not reference_text or reference_text.strip() == "":
        raise HTTPException(
            status_code=400,
            detail={"error": "EMPTY_REFERENCE_TEXT", "message": "Văn bản mẫu (reference_text) không được để trống"}
        )

    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"[SPEECH_ASSESS] Lỗi khi đọc file upload: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": "READ_ERROR", "message": f"Không thể đọc file âm thanh: {e}"}
        )

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "EMPTY_AUDIO_FILE", "message": "File âm thanh rỗng (0 bytes)"}
        )

    try:
        result = await pronunciation_service.assess(
            file_bytes=content,
            filename=file.filename,
            reference_text=reference_text,
            language=language
        )
        return result
    except ValueError as val_err:
        err_msg = str(val_err)
        if err_msg == "FILE_TOO_LARGE":
            raise HTTPException(
                status_code=413,
                detail={"error": "FILE_TOO_LARGE", "message": "File âm thanh vượt quá giới hạn 15MB"}
            )
        elif err_msg in ("EMPTY_AUDIO_FILE", "INVALID_AUDIO_FORMAT"):
            raise HTTPException(
                status_code=400,
                detail={"error": err_msg, "message": "File âm thanh không hợp lệ hoặc bị lỗi định dạng"}
            )
        elif err_msg == "EMPTY_REFERENCE_TEXT":
            raise HTTPException(
                status_code=400,
                detail={"error": "EMPTY_REFERENCE_TEXT", "message": "Văn bản mẫu không được để trống"}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={"error": "VALIDATION_ERROR", "message": err_msg}
            )
    except Exception as e:
        logger.error(f"[SPEECH_ASSESS] Lỗi hệ thống khi chấm phát âm: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "ASSESSMENT_FAILED", "message": f"Lỗi hệ thống khi chấm phát âm: {e}"}
        )
