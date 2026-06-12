"""
Integration tests cho Job API (meowCDDAI/text_embeding/routes_jobs.py).

Dùng FastAPI TestClient mount CHỈ `jobs_router` (không cần toàn bộ app meowCDDAI
hay DB/meowAI sống). Các singleton mà router tham chiếu được patch:
  - `text_embeding.routes_jobs.job_repository`  -> fake repo (không chạm DB)
  - `text_embeding.routes_jobs.job_runner`      -> mock runner (background no-op)
Ngoài ra patch `job_runner.call_meow_ai_endpoint` để chứng minh KHÔNG có lệnh
gọi meowAI nào xảy ra đồng bộ trong lúc xử lý request tạo job.

Xác thực:
- **Property 1: Job Creation Returns Immediately** — `POST /extract-async` và
  `/describe-async` trả 202 + `job_id`, job được tạo (status=pending) TRƯỚC khi
  bất kỳ lệnh gọi meowAI nào được thực hiện.
- **Property 7: Job Resumability** — `GET /jobs/{job_id}` trả trạng thái hiện
  tại (pending/processing/completed) cho id đã biết, 404 khi không tồn tại.
- Validation: input rỗng -> 400.

Validates: Requirements 1.1, 1.7, 4.1, 5.4
"""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Đảm bảo meowCDDAI root nằm trên sys.path (conftest.py từ task 4.2 cũng làm
# việc này, nhưng lặp lại ở đây để test độc lập với rootdir của pytest).
_MEOWCDDAI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MEOWCDDAI_DIR not in sys.path:
    sys.path.insert(0, _MEOWCDDAI_DIR)

# Import module (không phải chỉ router) để có thể patch các singleton bên trong.
from text_embeding import routes_jobs  # noqa: E402
import job_runner as job_runner_module  # noqa: E402

FIXED_JOB_ID = "11111111-2222-3333-4444-555555555555"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeRepo:
    """Fake JobRepository: create_job trả id cố định, get_job theo script."""

    def __init__(self):
        self.created = []  # [(job_type, input_data)]
        self._jobs = {}    # job_id -> job dict trả bởi get_job

    def create_job(self, job_type: str, input_data: dict) -> str:
        self.created.append((job_type, input_data))
        return FIXED_JOB_ID

    def get_job(self, job_id: str):
        return self._jobs.get(job_id)

    def set_job(self, job_id: str, job: dict):
        self._jobs[job_id] = job


@pytest.fixture
def fake_repo():
    return FakeRepo()


@pytest.fixture
def mock_runner():
    """Mock JobRunner — các method background là no-op MagicMock (sync)."""
    runner = MagicMock()
    # run_extraction/run_description là MagicMock thường (không coroutine) nên
    # Starlette BackgroundTasks chạy chúng trong threadpool: an toàn, no-op.
    return runner


@pytest.fixture
def spy_meow_ai(monkeypatch):
    """Patch call_meow_ai_endpoint để chứng minh meowAI không bị gọi đồng bộ."""
    spy = MagicMock(name="call_meow_ai_endpoint")
    monkeypatch.setattr(job_runner_module, "call_meow_ai_endpoint", spy)
    return spy


@pytest.fixture
def client(monkeypatch, fake_repo, mock_runner):
    """TestClient với app chỉ mount jobs_router; patch các singleton."""
    monkeypatch.setattr(routes_jobs, "job_repository", fake_repo)
    monkeypatch.setattr(routes_jobs, "job_runner", mock_runner)

    app = FastAPI()
    app.include_router(routes_jobs.router)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Property 1: Job Creation Returns Immediately
# --------------------------------------------------------------------------- #
def test_extract_async_returns_202_with_job_id(client, fake_repo, mock_runner, spy_meow_ai):
    """POST /extract-async trả 202 + job_id; job tạo trước, meowAI chưa bị gọi."""
    resp = client.post(
        "/extract-async",
        json={"intervention_goal": "Cải thiện giao tiếp", "raw_content": ["đoạn 1"]},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": FIXED_JOB_ID}

    # Job được tạo với job_type=extraction, status=pending (do create_job đặt).
    assert fake_repo.created == [
        ("extraction", {"intervention_goal": "Cải thiện giao tiếp", "raw_content": ["đoạn 1"]})
    ]

    # meowAI KHÔNG bị gọi đồng bộ trong lúc xử lý request.
    assert spy_meow_ai.call_count == 0

    # Background task được schedule đúng runner.run_extraction với job_id.
    mock_runner.run_extraction.assert_called_once_with(FIXED_JOB_ID)
    # Không gọi nhầm sang description.
    mock_runner.run_description.assert_not_called()


def test_describe_async_returns_202_with_job_id(client, fake_repo, mock_runner, spy_meow_ai):
    """POST /describe-async trả 202 + job_id; job tạo trước, meowAI chưa bị gọi."""
    resp = client.post(
        "/describe-async",
        json={"confirmed_content": "Nội dung đã xác nhận", "tone": "giáo viên"},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": FIXED_JOB_ID}

    assert len(fake_repo.created) == 1
    job_type, input_data = fake_repo.created[0]
    assert job_type == "description"
    assert input_data["confirmed_content"] == "Nội dung đã xác nhận"
    assert input_data["tone"] == "giáo viên"

    assert spy_meow_ai.call_count == 0
    mock_runner.run_description.assert_called_once_with(FIXED_JOB_ID)
    mock_runner.run_extraction.assert_not_called()


# --------------------------------------------------------------------------- #
# Validation: input rỗng -> 400
# --------------------------------------------------------------------------- #
def test_extract_async_empty_goal_returns_400(client, fake_repo, mock_runner):
    """intervention_goal rỗng -> 400, không tạo job, không schedule runner."""
    resp = client.post("/extract-async", json={"intervention_goal": "   ", "raw_content": []})

    assert resp.status_code == 400
    assert fake_repo.created == []
    mock_runner.run_extraction.assert_not_called()


def test_describe_async_empty_content_returns_400(client, fake_repo, mock_runner):
    """confirmed_content rỗng -> 400, không tạo job, không schedule runner."""
    resp = client.post("/describe-async", json={"confirmed_content": ""})

    assert resp.status_code == 400
    assert fake_repo.created == []
    mock_runner.run_description.assert_not_called()


# --------------------------------------------------------------------------- #
# Property 7: Job Resumability — GET /jobs/{id} qua các phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "job, expected_status, expected_progress, expected_result, expected_error",
    [
        # pending: chưa có progress/result/error.
        (
            {"status": "pending", "progress": None, "result": None, "error_message": None},
            "pending",
            None,
            None,
            None,
        ),
        # processing: có progress phase=extracting.
        (
            {
                "status": "processing",
                "progress": {"phase": "extracting", "message": "Đang trích lọc nội dung"},
                "result": None,
                "error_message": None,
            },
            "processing",
            {"phase": "extracting", "message": "Đang trích lọc nội dung"},
            None,
            None,
        ),
        # completed: có result đầy đủ.
        (
            {
                "status": "completed",
                "progress": {"phase": "completed", "low_confidence": False},
                "result": {
                    "extracted_content": "kết quả",
                    "confidence": 0.9,
                    "sources": [],
                    "low_confidence": False,
                },
                "error_message": None,
            },
            "completed",
            {"phase": "completed", "low_confidence": False},
            {
                "extracted_content": "kết quả",
                "confidence": 0.9,
                "sources": [],
                "low_confidence": False,
            },
            None,
        ),
    ],
)
def test_get_job_returns_current_status_across_phases(
    client, fake_repo, job, expected_status, expected_progress, expected_result, expected_error
):
    """GET /jobs/{id} trả trạng thái hiện tại bất kể phase nào (Property 7)."""
    fake_repo.set_job(FIXED_JOB_ID, job)

    resp = client.get(f"/jobs/{FIXED_JOB_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == expected_status
    assert body["progress"] == expected_progress
    assert body["result"] == expected_result
    assert body["error_message"] == expected_error


def test_get_job_returns_404_when_not_found(client, fake_repo):
    """GET /jobs/{id} -> 404 khi get_job trả None."""
    resp = client.get("/jobs/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found"


def test_get_job_returns_failed_with_error_message(client, fake_repo):
    """GET /jobs/{id} trả error_message khi job failed."""
    fake_repo.set_job(
        FIXED_JOB_ID,
        {
            "status": "failed",
            "progress": {"phase": "extracting"},
            "result": None,
            "error_message": "meowAI 500",
        },
    )

    resp = client.get(f"/jobs/{FIXED_JOB_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["error_message"] == "meowAI 500"
