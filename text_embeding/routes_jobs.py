"""
Job API Router — endpoint async cho pipeline xử lý nội dung can thiệp.

Theo ranh giới kiến trúc (design B.3): meowCDDAI là orchestrator async mỏng.
Các endpoint ở đây chỉ:
  1. Validate input + tạo job (status=pending) qua `job_repository`.
  2. Schedule background task gọi `job_runner.run_extraction/run_description`
     (FastAPI BackgroundTasks) — KHÔNG chạy pipeline đồng bộ.
  3. Trả `job_id` ngay (HTTP 202) để frontend poll.

Endpoints:
  POST /extract-async   -> tạo job extraction, trả {job_id} (202)
  POST /describe-async  -> tạo job description, trả {job_id} (202)
  GET  /jobs/{job_id}   -> trả {status, progress, result, error_message}; 404 nếu không tồn tại

Requirements: 1.1, 1.2, 1.5, 1.6, 4.1, 5.4.
"""

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

from job_repository import job_repository
from job_runner import job_runner

router = APIRouter(tags=["Jobs"])


# --------------------------------------------------------------------------- #
# Request / Response models
# --------------------------------------------------------------------------- #
class ExtractAsyncRequest(BaseModel):
    intervention_goal: str
    raw_content: List[str] = []


class DescribeAsyncRequest(BaseModel):
    confirmed_content: str
    tone: str = "giáo viên"
    context: Optional[dict] = None
    skip_extraction: bool = False


class JobCreatedResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: str
    progress: Optional[dict] = None
    result: Optional[dict] = None
    error_message: Optional[str] = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post(
    "/extract-async",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def extract_async(req: ExtractAsyncRequest, background_tasks: BackgroundTasks):
    """Tạo job trích lọc và chạy nền; trả `job_id` ngay (202). (Requirements 1.1, 1.2)"""
    if not req.intervention_goal or not req.intervention_goal.strip():
        raise HTTPException(status_code=400, detail="intervention_goal must not be empty")

    job_id = job_repository.create_job(
        "extraction",
        {
            "intervention_goal": req.intervention_goal,
            "raw_content": req.raw_content,
        },
    )

    # FastAPI BackgroundTasks hỗ trợ schedule coroutine function: truyền hàm
    # coroutine + job_id, KHÔNG gọi/await tại đây để trả về ngay.
    background_tasks.add_task(job_runner.run_extraction, job_id)

    return JobCreatedResponse(job_id=job_id)


@router.post(
    "/describe-async",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def describe_async(req: DescribeAsyncRequest, background_tasks: BackgroundTasks):
    """Tạo job sinh mô tả chi tiết và chạy nền; trả `job_id` ngay (202). (Requirement 4.1)"""
    if not req.confirmed_content or not req.confirmed_content.strip():
        raise HTTPException(status_code=400, detail="confirmed_content must not be empty")

    job_id = job_repository.create_job(
        "description",
        {
            "confirmed_content": req.confirmed_content,
            "tone": req.tone,
            "context": req.context,
            "skip_extraction": req.skip_extraction,
        },
    )

    background_tasks.add_task(job_runner.run_description, job_id)

    return JobCreatedResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Trả trạng thái hiện tại của job theo `job_id`. (Requirements 1.5, 1.6, 5.4)"""
    job = job_repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        status=job.get("status"),
        progress=job.get("progress"),
        result=job.get("result"),
        error_message=job.get("error_message"),
    )
