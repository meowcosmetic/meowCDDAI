"""
JobRunner — background runner mỏng cho pipeline xử lý nội dung can thiệp (async).

Theo ranh giới kiến trúc (design B.2 + Property 8): meowCDDAI CHỈ điều phối job:
  1. Đặt status=processing + progress.phase (coarse-grained).
  2. Gọi đúng MỘT endpoint cấp cao của meowAI (`/cdd/extract` hoặc
     `/cdd/process-intervention`) qua `call_meow_ai_endpoint`.
  3. Lưu kết quả/lỗi vào job qua `job_repository`.

Runner KHÔNG chứa logic agent/consensus/batching và KHÔNG gọi raw `/chat`
(toàn bộ logic gọi model sống trong meowAI). Status chỉ chuyển tiến theo
`pending → processing → (completed | failed)` (Property 2).

Requirements: 1.2, 1.3, 4.3, 6.3.
"""

import logging

import httpx

from config import Config
from job_repository import job_repository

logger = logging.getLogger(__name__)

# Base URL của meowAI (Central AI Hub), ví dụ http://meow-ai:8003.
# Dùng chung nguồn với llm_client.py / routes_extraction.py.
MEOW_AI_URL = Config.MEOW_AI_URL

# Pipeline (batching + consensus + 4 agents) chạy bên trong meowAI nên là một
# lệnh gọi HTTP dài — cần timeout đủ lớn (~1800s = 30 phút).
MEOW_AI_PIPELINE_TIMEOUT = 1800.0


async def call_meow_ai_endpoint(path: str, payload: dict) -> dict:
    """HTTP client mỏng POST tới endpoint cấp cao của meowAI.

    Đây là chỗ DUY NHẤT meowCDDAI nói chuyện với meowAI cho pipeline này.
    KHÔNG gọi `/chat` — chỉ các endpoint cấp cao (`/cdd/extract`,
    `/cdd/process-intervention`).

    Args:
        path: Đường dẫn endpoint cấp cao, ví dụ "/cdd/extract".
        payload: Body JSON gửi đi.

    Returns:
        Dict JSON phản hồi từ meowAI.

    Raises:
        httpx.HTTPStatusError: khi meowAI trả mã lỗi HTTP.
        httpx.RequestError: khi không kết nối được / timeout.
    """
    url = f"{MEOW_AI_URL}{path}"
    logger.info(f"[JOB_RUNNER] → POST {url}")
    async with httpx.AsyncClient(timeout=MEOW_AI_PIPELINE_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


class JobRunner:
    """Background runner điều phối job (không chứa logic model)."""

    def __init__(self, repo=job_repository):
        self.repo = repo

    async def run_extraction(self, job_id: str):
        """Chạy job trích lọc: gọi `POST /cdd/extract` của meowAI.

        Progress: pending → extracting → completed (hoặc failed).
        Requirements: 1.2, 1.3, 6.3.
        """
        job = self.repo.get_job(job_id)
        if job is None:
            logger.error(f"[JOB_RUNNER] ❌ Job {job_id} not found for extraction")
            return

        self.repo.update_status(job_id, "processing")
        self.repo.update_progress(
            job_id,
            {"phase": "extracting", "message": "Đang trích lọc nội dung"},
        )

        try:
            job_input = job.get("input") or {}
            resp = await call_meow_ai_endpoint(
                "/cdd/extract",
                {
                    "goal": job_input.get("intervention_goal"),
                    "raw_content": job_input.get("raw_content", []),
                },
            )
            # resp -> {extracted_content, confidence, sources, low_confidence,
            #          batch_count?, rounds_used?}
            self.repo.set_result(job_id, resp)
            self.repo.update_progress(
                job_id,
                {
                    "phase": "completed",
                    "batch_count": resp.get("batch_count"),
                    "rounds_used": resp.get("rounds_used"),
                    "low_confidence": resp.get("low_confidence", False),
                },
            )
            self.repo.update_status(job_id, "completed")
            logger.info(f"[JOB_RUNNER] ✅ Extraction job {job_id} completed")
        except Exception as e:
            logger.error(f"[JOB_RUNNER] ❌ Extraction job {job_id} failed: {str(e)}")
            self.repo.set_error(job_id, str(e))
            self.repo.update_status(job_id, "failed")

    async def run_description(self, job_id: str):
        """Chạy job sinh mô tả chi tiết: gọi `POST /cdd/process-intervention`.

        Progress: pending → describing → completed (hoặc failed).
        Requirements: 1.2, 1.3, 4.3, 6.3.
        """
        job = self.repo.get_job(job_id)
        if job is None:
            logger.error(f"[JOB_RUNNER] ❌ Job {job_id} not found for description")
            return

        self.repo.update_status(job_id, "processing")
        self.repo.update_progress(
            job_id,
            {"phase": "describing", "message": "Đang sinh mô tả chi tiết"},
        )

        try:
            job_input = job.get("input") or {}
            resp = await call_meow_ai_endpoint(
                "/cdd/process-intervention",
                {
                    "goal": job_input.get("confirmed_content"),
                    "context": job_input.get("context"),
                    "tone": job_input.get("tone", "giáo viên"),
                },
            )
            # resp -> {expert_analysis, practical_content, verified_content,
            #          final_content, workflow_summary, low_confidence}
            self.repo.set_result(job_id, resp)
            self.repo.update_progress(
                job_id,
                {
                    "phase": "completed",
                    "low_confidence": resp.get("low_confidence", False),
                },
            )
            self.repo.update_status(job_id, "completed")
            logger.info(f"[JOB_RUNNER] ✅ Description job {job_id} completed")
        except Exception as e:
            logger.error(f"[JOB_RUNNER] ❌ Description job {job_id} failed: {str(e)}")
            self.repo.set_error(job_id, str(e))
            self.repo.update_status(job_id, "failed")


# Module-level singleton — Job API (task 6.1) sẽ schedule các method này
# qua FastAPI BackgroundTasks.
job_runner = JobRunner()
