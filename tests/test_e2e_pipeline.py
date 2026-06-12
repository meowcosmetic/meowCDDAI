"""
End-to-end integration test xuyên ranh giới meowCDDAI (orchestrator) ↔ meowAI.

Mục tiêu (task 12.1): chứng minh toàn bộ vòng đời job async hoạt động end-to-end
trong meowCDDAI mà KHÔNG cần DB sống hay meowAI sống — chỉ mock đúng hai ranh
giới (DB + lệnh gọi HTTP tới meowAI), còn lại chạy logic thật của
`JobRunner` + logic tiêu thụ `JobRepository` (qua một fake in-memory đầy đủ
trạng thái).

Khác với `test_job_runner.py` (FakeRepo chỉ ghi lại trình tự lệnh gọi), ở đây
`InMemoryJobRepository` lưu trạng thái thật của job (status/progress/result/
error) để `get_job` phản ánh đúng trạng thái sau khi runner chạy — mô phỏng
luồng "resume/poll" của frontend (Property 7).

Các property được kiểm chứng end-to-end:
- **Property 2 — Status Monotonic Progression**: pending → processing → completed.
- **Property 6 — Result Completeness**: description job completed có đủ 5 khóa.
- **Property 8 — Orchestrator Has No Model Logic**: runner gọi đúng MỘT endpoint
  cấp cao (`/cdd/extract` | `/cdd/process-intervention`), KHÔNG gọi `/chat`.
- **Property 7 — Job Resumability**: GET-style `get_job` trả trạng thái hiện tại
  (completed + result) cho bất kỳ client nào hỏi sau khi job xong.

Validates: Requirements 1, 2, 3, 4, 5, 6
"""

import datetime
import os
import sys
import uuid
from unittest.mock import AsyncMock

import pytest

# Đảm bảo meowCDDAI root (chứa job_runner.py, job_repository.py) nằm trên
# sys.path — conftest.py (task 4.2) cũng làm việc này; lặp lại để độc lập rootdir.
_MEOWCDDAI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MEOWCDDAI_DIR not in sys.path:
    sys.path.insert(0, _MEOWCDDAI_DIR)

import job_runner  # noqa: E402
from job_runner import JobRunner  # noqa: E402


# --------------------------------------------------------------------------- #
# In-memory JobRepository — fake DB đầy đủ trạng thái (boundary #1: DB)
# --------------------------------------------------------------------------- #
class InMemoryJobRepository:
    """Bản sao in-memory của JobRepository (Requirement 5) — không chạm Postgres.

    Lưu job trong dict; mọi lệnh ghi cập nhật `updated_at`. `get_job` trả bản
    sao để mô phỏng đọc qua API (frontend poll). `status_history` ghi lại thứ tự
    chuyển trạng thái để kiểm tra tính đơn điệu (Property 2).
    """

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.status_history: dict[str, list[str]] = {}
        self._tick = 0

    def _now(self):
        self._tick += 1
        return datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=self._tick)

    def create_job(self, job_type: str, input_data: dict) -> str:
        job_id = str(uuid.uuid4())
        now = self._now()
        self.jobs[job_id] = {
            "id": job_id,
            "job_type": job_type,
            "status": "pending",
            "input": input_data or {},
            "progress": None,
            "result": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
        self.status_history[job_id] = ["pending"]
        return job_id

    def get_job(self, job_id: str):
        job = self.jobs.get(job_id)
        return dict(job) if job is not None else None

    def update_status(self, job_id: str, status: str):
        self.jobs[job_id]["status"] = status
        self.jobs[job_id]["updated_at"] = self._now()
        self.status_history[job_id].append(status)

    def update_progress(self, job_id: str, progress: dict):
        self.jobs[job_id]["progress"] = progress or {}
        self.jobs[job_id]["updated_at"] = self._now()

    def set_result(self, job_id: str, result: dict):
        self.jobs[job_id]["result"] = result or {}
        self.jobs[job_id]["updated_at"] = self._now()

    def set_error(self, job_id: str, message: str):
        self.jobs[job_id]["error_message"] = message
        self.jobs[job_id]["updated_at"] = self._now()


# Thứ hạng để kiểm tra status tiến lên đơn điệu (Property 2).
_STATUS_RANK = {"pending": 0, "processing": 1, "completed": 2, "failed": 2}


def assert_monotonic(history: list[str]):
    ranks = [_STATUS_RANK[s] for s in history]
    assert ranks == sorted(ranks), f"Status không đơn điệu: {history}"
    assert history[0] == "pending"
    assert history[-1] in ("completed", "failed")


# --------------------------------------------------------------------------- #
# Mock meowAI responses (boundary #2: HTTP call tới meowAI)
# --------------------------------------------------------------------------- #
# Mô phỏng outcome "nội dung dài → nhiều batch → consolidate 2 tầng" của
# /cdd/extract: batch_count>1 và rounds_used>1 (Requirements 2, 3).
EXTRACT_RESPONSE = {
    "extracted_content": "Nội dung đã trích lọc và tổng hợp từ nhiều batch.",
    "confidence": 0.88,
    "sources": [
        {"book_name": "Sách Can Thiệp A", "chapter": "3", "page": 42, "score": 0.91},
        {"book_name": "Sách Can Thiệp B", "chapter": "1", "page": 8, "score": 0.84},
    ],
    "low_confidence": False,
    "batch_count": 4,   # >1 -> đã đi qua bước consolidate tầng 2
    "rounds_used": 2,   # >1 -> consensus Extractor↔Validator chạy nhiều vòng
}

# Mô phỏng /cdd/process-intervention: 5 khóa nội dung + low_confidence
# (Requirement 4.5, Property 6).
DESCRIBE_RESPONSE = {
    "expert_analysis": "Phân tích chuyên gia về mục tiêu can thiệp.",
    "practical_content": "Nội dung thực hành cụ thể cho trẻ.",
    "verified_content": "Nội dung đã được kiểm chứng tính chính xác.",
    "final_content": "<p>Mô tả chi tiết cuối cùng dạng HTML.</p>",
    "workflow_summary": {
        "step_1": "Expert: phân tích",
        "step_2": "Practical: triển khai",
        "step_3": "Verifier: kiểm chứng",
        "step_4": "Editor: hoàn thiện",
    },
    "low_confidence": False,
}

_DESCRIPTION_CONTENT_KEYS = {
    "expert_analysis",
    "practical_content",
    "verified_content",
    "final_content",
    "workflow_summary",
}


# --------------------------------------------------------------------------- #
# 1. Extraction lifecycle end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_extraction_lifecycle_end_to_end(monkeypatch):
    """create → run_extraction → completed + result persisted + poll-ready.

    Chứng minh: nội dung dài (nhiều batch, consolidate) đi qua /cdd/extract,
    job chuyển pending→processing→completed (Property 2), result lưu lại với
    extracted_content, và path gọi là /cdd/extract chứ KHÔNG phải /chat
    (Property 8). Sau đó get_job trả completed + result (Property 7).
    """
    mock_call = AsyncMock(return_value=EXTRACT_RESPONSE)
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = InMemoryJobRepository()
    runner = JobRunner(repo=repo)

    # --- create job (pending) ---
    job_id = repo.create_job(
        "extraction",
        {
            "intervention_goal": "Cải thiện kỹ năng giao tiếp cho trẻ tự kỷ",
            "raw_content": ["đoạn nội dung dài 1", "đoạn nội dung dài 2"],
        },
    )
    assert repo.get_job(job_id)["status"] == "pending"

    # --- chạy runner thật (chỉ boundary HTTP bị mock) ---
    await runner.run_extraction(job_id)

    # Property 8: đúng MỘT lệnh gọi tới endpoint cấp cao /cdd/extract, không /chat.
    assert mock_call.await_count == 1
    called_path = mock_call.await_args.args[0]
    assert called_path == "/cdd/extract"
    assert "/chat" not in called_path
    # Payload mang đúng input của job.
    called_payload = mock_call.await_args.args[1]
    assert called_payload["goal"] == "Cải thiện kỹ năng giao tiếp cho trẻ tự kỷ"
    assert called_payload["raw_content"] == ["đoạn nội dung dài 1", "đoạn nội dung dài 2"]

    # Property 2: pending → processing → completed (không quay lui).
    assert repo.status_history[job_id] == ["pending", "processing", "completed"]
    assert_monotonic(repo.status_history[job_id])

    # Property 7: client poll qua get_job thấy completed + result đầy đủ.
    polled = repo.get_job(job_id)
    assert polled["status"] == "completed"
    assert polled["result"]["extracted_content"] == EXTRACT_RESPONSE["extracted_content"]
    assert polled["result"]["confidence"] == 0.88
    assert polled["result"]["sources"] == EXTRACT_RESPONSE["sources"]
    assert polled["error_message"] is None

    # Progress coarse-grained phản ánh kết quả tóm tắt từ meowAI (batch/rounds).
    assert polled["progress"]["phase"] == "completed"
    assert polled["progress"]["batch_count"] == 4   # nhiều batch -> consolidate
    assert polled["progress"]["rounds_used"] == 2
    assert polled["progress"]["low_confidence"] is False


# --------------------------------------------------------------------------- #
# 2. Description lifecycle end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_description_lifecycle_end_to_end(monkeypatch):
    """create → run_description → completed với đủ 5 khóa nội dung persisted.

    Property 6 (Result Completeness) + Property 8 (path /cdd/process-intervention).
    """
    mock_call = AsyncMock(return_value=DESCRIBE_RESPONSE)
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = InMemoryJobRepository()
    runner = JobRunner(repo=repo)

    job_id = repo.create_job(
        "description",
        {
            "confirmed_content": "Nội dung đã xác nhận sau bước trích lọc",
            "context": {"book": "Sách Can Thiệp A"},
            "tone": "giáo viên",
        },
    )
    assert repo.get_job(job_id)["status"] == "pending"

    await runner.run_description(job_id)

    # Property 8: đúng MỘT lệnh gọi /cdd/process-intervention, không /chat.
    assert mock_call.await_count == 1
    called_path = mock_call.await_args.args[0]
    assert called_path == "/cdd/process-intervention"
    assert "/chat" not in called_path
    called_payload = mock_call.await_args.args[1]
    assert called_payload["goal"] == "Nội dung đã xác nhận sau bước trích lọc"
    assert called_payload["tone"] == "giáo viên"
    assert called_payload["context"] == {"book": "Sách Can Thiệp A"}

    # Property 2: pending → processing → completed.
    assert repo.status_history[job_id] == ["pending", "processing", "completed"]
    assert_monotonic(repo.status_history[job_id])

    # Property 6: result có đủ 5 khóa nội dung (+ low_confidence) persisted.
    polled = repo.get_job(job_id)
    assert polled["status"] == "completed"
    result = polled["result"]
    assert _DESCRIPTION_CONTENT_KEYS.issubset(result.keys())
    assert result["final_content"] == DESCRIBE_RESPONSE["final_content"]
    assert result["workflow_summary"] == DESCRIBE_RESPONSE["workflow_summary"]
    assert result["low_confidence"] is False
    assert polled["error_message"] is None
    assert polled["progress"]["phase"] == "completed"


# --------------------------------------------------------------------------- #
# 3. Failure path end-to-end (meowAI lỗi -> failed, không mất tính đơn điệu)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_extraction_failure_end_to_end(monkeypatch):
    """meowAI 5xx -> job failed với error_message; status đơn điệu (Property 2)."""
    mock_call = AsyncMock(side_effect=RuntimeError("meowAI 503: unavailable"))
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = InMemoryJobRepository()
    runner = JobRunner(repo=repo)

    job_id = repo.create_job(
        "extraction",
        {"intervention_goal": "g", "raw_content": []},
    )

    await runner.run_extraction(job_id)

    assert mock_call.await_count == 1
    assert mock_call.await_args.args[0] == "/cdd/extract"

    polled = repo.get_job(job_id)
    assert polled["status"] == "failed"
    assert polled["result"] is None
    assert polled["error_message"] == "meowAI 503: unavailable"

    # Property 2: pending → processing → failed (terminal, không quay lui).
    assert repo.status_history[job_id] == ["pending", "processing", "failed"]
    assert_monotonic(repo.status_history[job_id])


# --------------------------------------------------------------------------- #
# 4. Low-confidence (best-effort) được lưu nguyên vào result (Property 5)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_low_confidence_flag_persisted_end_to_end(monkeypatch):
    """Hết 3 vòng chưa đồng thuận: meowAI trả low_confidence=true; orchestrator
    PHẢI lưu nguyên cờ này vào result + progress (Property 5)."""
    low_conf_response = {
        **EXTRACT_RESPONSE,
        "low_confidence": True,
        "rounds_used": 3,
    }
    mock_call = AsyncMock(return_value=low_conf_response)
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = InMemoryJobRepository()
    runner = JobRunner(repo=repo)

    job_id = repo.create_job("extraction", {"intervention_goal": "g", "raw_content": []})
    await runner.run_extraction(job_id)

    polled = repo.get_job(job_id)
    assert polled["status"] == "completed"
    assert polled["result"]["low_confidence"] is True
    assert polled["progress"]["low_confidence"] is True
    assert polled["progress"]["rounds_used"] == 3
