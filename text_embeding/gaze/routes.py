"""
FastAPI routes cho Gaze Analysis
"""
import logging
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from .models import GazeAnalysisResponse
# from .processor import process_gaze_analysis  # Tạm thời comment vì đang refactor
from .config import MEDIAPIPE_AVAILABLE, mp_face_mesh

logger = logging.getLogger(__name__)

# Lazy import để tránh circular dependency
# Import sẽ được thực hiện khi hàm được gọi, không phải ở module level
def _get_process_gaze_analysis():
    """Lazy import helper để tránh circular dependency"""
    try:
        from .processor import process_gaze_analysis
        return process_gaze_analysis
    except (ImportError, NotImplementedError) as e:
        logger.error(f"[Gaze] Không thể import process_gaze_analysis từ processor: {e}")
        raise NotImplementedError("process_gaze_analysis chưa được implement trong processor.py")

router = APIRouter(prefix="/screening/gaze", tags=["Screening - Gaze Tracking"])


@router.post("/analyze_camera", response_model=GazeAnalysisResponse)
async def analyze_gaze_camera(
    target_type: str = Form("camera", description="Loại đối tượng nhìn vào"),
    show_video: str = Form("true", description="Hiển thị video trong quá trình xử lý (true/false)"),
    camera_id: int = Form(0, description="Camera ID (0 = default camera)"),
    max_duration: float = Form(60.0, description="Thời gian tối đa phân tích (giây), 0 = không giới hạn")
):
    """
    Phân tích gaze từ camera real-time
    """
    show_video_bool = show_video.lower() in ("true", "1", "yes", "on")
    
    # Kiểm tra MediaPipe
    if not MEDIAPIPE_AVAILABLE or mp_face_mesh is None:
        raise HTTPException(
            status_code=500,
            detail="MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe"
        )
    
    cap = None
    try:
        # Mở camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Không thể mở camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"[Gaze] Bắt đầu phân tích camera (ID: {camera_id}, max_duration: {max_duration}s)")
        logger.info("[Gaze] Điều khiển cửa sổ video: 'q'/ESC = dừng, 'p'/Space = tạm dừng/tiếp tục")
        
        # Lazy import để tránh circular dependency
        process_gaze_analysis = _get_process_gaze_analysis()
        
        # Sử dụng hàm helper chung để xử lý
        result = process_gaze_analysis(
            cap=cap,
            target_type=target_type,
            show_video=show_video_bool,
            max_duration=max_duration,
            is_camera=True
        )
        logger.info(f"[Gaze] Hoàn thành phân tích camera: {result.total_frames} frames")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Gaze] Error in camera mode: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý camera: {str(e)}")
    finally:
        # Cleanup camera
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        try:
            cv2.destroyAllWindows()
        except:
            pass


@router.post("/analyze", response_model=GazeAnalysisResponse)
async def analyze_gaze(
    video: UploadFile = File(..., description="Video file để phân tích"),
    target_type: str = Form("camera", description="Loại đối tượng nhìn vào"),
    show_video: str = Form("false", description="Hiển thị video trong quá trình xử lý (true/false)")
):
    """
    Phân tích gaze từ video file
    """
    show_video_bool = show_video.lower() in ("true", "1", "yes", "on")
    
    # Kiểm tra MediaPipe
    if not MEDIAPIPE_AVAILABLE or mp_face_mesh is None:
        raise HTTPException(
            status_code=500,
            detail="MediaPipe không được cài đặt. Vui lòng cài: pip install mediapipe"
        )
    
    temp_path = None
    cap = None
    try:
        # Lưu file tạm
        temp_path = f"temp_{video.filename}"
        with open(temp_path, "wb") as f:
            content = await video.read()
            f.write(content)
        
        # Đọc video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể đọc video file")
        
        logger.info("[Gaze] Bắt đầu phân tích video file")
        if show_video_bool:
            logger.info("[Gaze] Điều khiển cửa sổ video: 'q'/ESC = dừng, 'p'/Space = tạm dừng/tiếp tục")
        
        # Lazy import để tránh circular dependency
        process_gaze_analysis = _get_process_gaze_analysis()
        
        # Sử dụng hàm helper chung để xử lý
        result = process_gaze_analysis(
            cap=cap,
            target_type=target_type,
            show_video=show_video_bool,
            max_duration=0.0,  # Không giới hạn cho video file
            is_camera=False
        )
        logger.info(f"[Gaze] Hoàn thành phân tích video: {result.total_frames} frames")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Gaze] Lỗi khi xử lý video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý video: {str(e)}")
    finally:
        # Cleanup
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        # Xóa file tạm
        if temp_path:
            try:
                import os
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

