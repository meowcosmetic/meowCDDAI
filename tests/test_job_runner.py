"""
Unit tests cho `JobRunner` (meowCDDAI/job_runner.py).

Xác thực hai property của orchestrator async mỏng (design B.2):

- **Property 8 — Orchestrator Has No Model Logic**: runner hoàn tất bằng ĐÚNG
  MỘT lệnh gọi tới endpoint cấp cao của meowAI (`/cdd/extract` cho extraction,
  `/cdd/process-intervention` cho description) và KHÔNG BAO GIỜ gọi raw `/chat`.
- **Property 2 — Status Monotonic Progression**: status chỉ tiến theo
  `pending → processing → (completed | failed)`, không quay lui.

Ngoài ra kiểm tra việc persist kết quả (set_result) khi thành công và set_error
khi meowAI lỗi.

Validates: Requirements 1.2, 6.3

Cách tiếp cận: patch `call_meow_ai_endpoint` (hàm module-level async trong
job_runner) bằng AsyncMock để bắt tham số `path` và trả về dict kịch bản; tiêm
một fake repo vào `JobRunner(repo=fake_repo)` để ghi lại thứ tự các lệnh gọi
update_status / update_progress / set_result / set_error.
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest

# Đảm bảo thư mục meowCDDAI (chứa job_runner.py) nằm trên sys.path khi chạy
# pytest từ bất kỳ rootdir nào. Tránh phụ thuộc vào conftest dùng chung
# (task 4.2 có thể thêm test khác trong cùng thư mục tests/).
_MEOWCDDAI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MEOWCDDAI_DIR not in sys.path:
    sys.path.insert(0, _MEOWCDDAI_DIR)

import job_runner  # noqa: E402
from job_runner import JobRunner  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes / helpers
# --------------------------------------------------------------------------- #
class FakeRepo:
    """Fake JobRepository ghi lại trình tự các lệnh ghi (không chạm DB)."""

    def __init__(self, job: dict | None):
        self._job = job
        self.status_calls: list[str] = []
        self.progress_calls: list[dict] = []
        self.result = None
        self.error = None

    def get_job(self, job_id: str):
        return self._job

    def update_status(self, job_id: str, status: str):
        self.status_calls.append(status)

    def update_progress(self, job_id: str, progress: dict):
        self.progress_calls.append(progress)

    def set_result(self, job_id: str, result: dict):
        self.result = result

    def set_error(self, job_id: str, message: str):
        self.error = message


# Thứ hạng để kiểm tra status tiến lên đơn điệu (Property 2).
_STATUS_RANK = {"pending": 0, "processing": 1, "completed": 2, "failed": 2}


def assert_monotonic(status_calls: list[str]):
    """Xác nhận status không bao giờ quay lui. Job khởi tạo ở 'pending'."""
    sequence = ["pending"] + status_calls
    ranks = [_STATUS_RANK[s] for s in sequence]
    assert ranks == sorted(ranks), f"Status không đơn điệu: {sequence}"
    # Không có trạng thái terminal nào xuất hiện trước rồi lại đổi.
    assert sequence[-1] in ("completed", "failed")


EXTRACTION_RESULT = {
    "extracted_content": "Nội dung đã trích lọc",
    "confidence": 0.87,
    "sources": [{"book_name": "Sách A", "chapter": "1", "page": 12, "score": 0.9}],
    "low_confidence": False,
    "batch_count": 3,
    "rounds_used": 2,
}

DESCRIPTION_RESULT = {
    "expert_analysis": "phân tích",
    "practical_content": "thực hành",
    "verified_content": "đã kiểm chứng",
    "final_content": "<p>nội dung cuối</p>",
    "workflow_summary": {"step_1": "a", "step_2": "b", "step_3": "c", "step_4": "d"},
    "low_confidence": False,
}


# --------------------------------------------------------------------------- #
# run_extraction
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_extraction_success(monkeypatch):
    """Thành công: gọi đúng /cdd/extract một lần, set_result, status completed."""
    mock_call = AsyncMock(return_value=EXTRACTION_RESULT)
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = FakeRepo(
        {
            "input": {
                "intervention_goal": "Cải thiện giao tiếp",
                "raw_content": ["đoạn 1", "đoạn 2"],
            }
        }
    )
    runner = JobRunner(repo=repo)

    await runner.run_extraction("job-1")

    # Property 8: đúng MỘT lệnh gọi endpoint cấp cao, path = /cdd/extract.
    assert mock_call.await_count == 1
    called_path = mock_call.await_args.args[0]
    assert called_path == "/cdd/extract"
    # KHÔNG BAO GIỜ gọi /chat.
    assert called_path != "/chat"
    assert "/chat" not in called_path

    # Payload chuyển đúng input.
    called_payload = mock_call.await_args.args[1]
    assert called_payload["goal"] == "Cải thiện giao tiếp"
    assert called_payload["raw_content"] == ["đoạn 1", "đoạn 2"]

    # Property 2: pending → processing → completed.
    assert repo.status_calls == ["processing", "completed"]
    assert_monotonic(repo.status_calls)

    # Kết quả meowAI được persist nguyên vẹn.
    assert repo.result == EXTRACTION_RESULT
    assert repo.error is None


@pytest.mark.asyncio
async def test_run_extraction_failure(monkeypatch):
    """meowAI lỗi: set_error được gọi, status → failed (không completed)."""
    mock_call = AsyncMock(side_effect=RuntimeError("meowAI 500"))
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = FakeRepo({"input": {"intervention_goal": "x", "raw_content": []}})
    runner = JobRunner(repo=repo)

    await runner.run_extraction("job-2")

    assert mock_call.await_count == 1
    assert mock_call.await_args.args[0] == "/cdd/extract"

    # Property 2: pending → processing → failed.
    assert repo.status_calls == ["processing", "failed"]
    assert_monotonic(repo.status_calls)

    assert repo.error == "meowAI 500"
    assert repo.result is None


# --------------------------------------------------------------------------- #
# run_description
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_run_description_success(monkeypatch):
    """Thành công: gọi đúng /cdd/process-intervention, set_result, completed."""
    mock_call = AsyncMock(return_value=DESCRIPTION_RESULT)
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = FakeRepo(
        {
            "input": {
                "confirmed_content": "Nội dung đã xác nhận",
                "context": {"book": "A"},
                "tone": "giáo viên",
            }
        }
    )
    runner = JobRunner(repo=repo)

    await runner.run_description("job-3")

    # Property 8: đúng MỘT lệnh gọi, path = /cdd/process-intervention, không /chat.
    assert mock_call.await_count == 1
    called_path = mock_call.await_args.args[0]
    assert called_path == "/cdd/process-intervention"
    assert "/chat" not in called_path

    called_payload = mock_call.await_args.args[1]
    assert called_payload["goal"] == "Nội dung đã xác nhận"
    assert called_payload["tone"] == "giáo viên"

    # Property 2: pending → processing → completed.
    assert repo.status_calls == ["processing", "completed"]
    assert_monotonic(repo.status_calls)

    assert repo.result == DESCRIPTION_RESULT
    assert repo.error is None


@pytest.mark.asyncio
async def test_run_description_failure(monkeypatch):
    """meowAI lỗi: set_error được gọi, status → failed."""
    mock_call = AsyncMock(side_effect=RuntimeError("timeout"))
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = FakeRepo({"input": {"confirmed_content": "y"}})
    runner = JobRunner(repo=repo)

    await runner.run_description("job-4")

    assert mock_call.await_count == 1
    assert mock_call.await_args.args[0] == "/cdd/process-intervention"

    # Property 2: pending → processing → failed.
    assert repo.status_calls == ["processing", "failed"]
    assert_monotonic(repo.status_calls)

    assert repo.error == "timeout"
    assert repo.result is None


# --------------------------------------------------------------------------- #
# Property 8 (explicit): runner chỉ dùng endpoint cấp cao, không /chat
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, expected_path, result",
    [
        ("run_extraction", "/cdd/extract", EXTRACTION_RESULT),
        ("run_description", "/cdd/process-intervention", DESCRIPTION_RESULT),
    ],
)
async def test_runner_only_calls_high_level_endpoints(
    monkeypatch, method, expected_path, result
):
    """Property 8: mọi path được gọi PHẢI là endpoint cấp cao, không có /chat."""
    mock_call = AsyncMock(return_value=result)
    monkeypatch.setattr(job_runner, "call_meow_ai_endpoint", mock_call)

    repo = FakeRepo(
        {
            "input": {
                "intervention_goal": "g",
                "raw_content": [],
                "confirmed_content": "c",
            }
        }
    )
    runner = JobRunner(repo=repo)

    await getattr(runner, method)("job-x")

    # Đúng một lệnh gọi tới endpoint cấp cao mong đợi.
    assert mock_call.await_count == 1
    all_called_paths = [c.args[0] for c in mock_call.await_args_list]
    assert all_called_paths == [expected_path]
    # Không path nào là /chat.
    assert all("/chat" not in p for p in all_called_paths)
