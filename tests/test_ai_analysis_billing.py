"""
Unit, Integration & Negative/Boundary Tests cho cơ chế khấu trừ Credit (AI_ANALYSIS)
và Auto-Refund trên pipeline Job bất đồng bộ (routes_jobs.py + job_runner.py + billing_client.py).

Tuân thủ:
- Rule 3 (Negative & Boundary Testing Protocol):
  1. Empty Body & Whitespace Payloads -> 400 Bad Request, KHÔNG gọi trừ tiền, KHÔNG tạo bản ghi job rác.
  2. Insufficient Credits (402) -> Trả 402 Payment Required, xác minh DB sạch (0 job được tạo).
  3. Invalid / Empty parameters ở BillingClient -> Ném ValueError.
  4. Auto-Refund khi JobRunner gặp lỗi -> Tự động gọi hoàn tiền và chuyển trạng thái job sang failed.
"""

import base64
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import billing_client as billing_module  # noqa: E402
from billing_client import (  # noqa: E402
    BillingClient,
    BillingServiceError,
    InsufficientCreditsError,
    extract_actor_from_headers,
    normalize_role,
)
import job_runner as job_runner_module  # noqa: E402
from job_runner import JobRunner  # noqa: E402
from text_embeding import routes_jobs  # noqa: E402

FIXED_JOB_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _make_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}.signature"


class FakeJobRepo:
    def __init__(self):
        self.created = []
        self._jobs = {}
        self.status_calls = []
        self.progress_calls = []
        self.result = None
        self.error = None

    def create_job(self, job_type: str, input_data: dict) -> str:
        self.created.append((job_type, input_data))
        self._jobs[FIXED_JOB_ID] = {
            "id": FIXED_JOB_ID,
            "job_type": job_type,
            "status": "pending",
            "input": input_data,
            "progress": None,
            "result": None,
            "error_message": None,
        }
        return FIXED_JOB_ID

    def get_job(self, job_id: str):
        return self._jobs.get(job_id)

    def set_job(self, job_id: str, job: dict):
        self._jobs[job_id] = job

    def update_status(self, job_id: str, status: str):
        self.status_calls.append(status)
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status

    def update_progress(self, job_id: str, progress: dict):
        self.progress_calls.append(progress)
        if job_id in self._jobs:
            self._jobs[job_id]["progress"] = progress

    def set_result(self, job_id: str, result: dict):
        self.result = result
        if job_id in self._jobs:
            self._jobs[job_id]["result"] = result

    def set_error(self, job_id: str, message: str):
        self.error = message
        if job_id in self._jobs:
            self._jobs[job_id]["error_message"] = message


@pytest.fixture
def fake_repo():
    return FakeJobRepo()


@pytest.fixture
def mock_runner():
    return MagicMock()


@pytest.fixture
def mock_billing(monkeypatch):
    mock = MagicMock()
    mock.charge.return_value = {
        "usageId": "usage-001",
        "userId": "user-123",
        "serviceCode": "AI_ANALYSIS",
        "chargedAmount": 10,
        "remainingBalance": 90,
        "status": "SUCCESS",
        "requestReference": "AI_ANALYSIS-ref-001",
    }
    mock.refund.return_value = {
        "usageId": "usage-001",
        "chargedAmount": 10,
        "remainingBalance": 100,
        "status": "REFUNDED",
    }
    monkeypatch.setattr(routes_jobs, "billing_client", mock)
    monkeypatch.setattr(job_runner_module, "billing_client", mock)
    return mock


@pytest.fixture
def client(monkeypatch, fake_repo, mock_runner, mock_billing):
    monkeypatch.setattr(routes_jobs, "job_repository", fake_repo)
    monkeypatch.setattr(routes_jobs, "job_runner", mock_runner)

    app = FastAPI()
    app.include_router(routes_jobs.router)
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# 1. Identity & Role Extraction Tests
# --------------------------------------------------------------------------- #
def test_extract_actor_from_jwt_and_role_normalization():
    token = _make_jwt({"sub": "parent-01", "roles": ["ROLE_CUSTOMER"]})
    uid, role = extract_actor_from_headers(authorization=f"Bearer {token}")
    assert uid == "parent-01"
    assert role == "USER"

    token_teacher = _make_jwt({"sub": "teacher-01", "roles": ["CUSTOMER", "ROLE_TEACHER"]})
    uid2, role2 = extract_actor_from_headers(authorization=f"Bearer {token_teacher}")
    assert uid2 == "teacher-01"
    assert role2 == "TEACHER"

    token_admin = _make_jwt({"sub": "admin-01", "roles": ["SUPER_ADMIN"]})
    uid3, role3 = extract_actor_from_headers(authorization=f"Bearer {token_admin}")
    assert uid3 == "admin-01"
    assert role3 == "ADMIN"


def test_extract_actor_fallback_headers():
    uid, role = extract_actor_from_headers(
        authorization="Bearer malformed.jwt",
        x_user_id="user-fallback",
        x_user_role="COLLABORATOR",
    )
    assert uid == "user-fallback"
    assert role == "COLLABORATOR"


def test_normalize_role_boundary_values():
    assert normalize_role(None) == "USER"
    assert normalize_role("   ") == "USER"
    assert normalize_role("ROLE_EXPERT") == "EXPERT"
    assert normalize_role("customer") == "USER"


# --------------------------------------------------------------------------- #
# 2. Happy Path: Pre-charge Credit on /extract-async & /describe-async
# --------------------------------------------------------------------------- #
def test_extract_async_charges_credits_with_jwt_and_persists_billing_ref(
    client, fake_repo, mock_runner, mock_billing
):
    token = _make_jwt({"sub": "user-123", "roles": ["CUSTOMER"]})
    resp = client.post(
        "/extract-async",
        headers={"Authorization": f"Bearer {token}"},
        json={"intervention_goal": "Tăng cường giao tiếp mắt", "raw_content": ["Nội dung 1"]},
    )

    assert resp.status_code == 202
    assert resp.json() == {"job_id": FIXED_JOB_ID}

    # Xác minh billing_client.charge đã được gọi đúng mã AI_ANALYSIS
    mock_billing.charge.assert_called_once()
    call_kwargs = mock_billing.charge.call_args.kwargs
    assert call_kwargs["user_id"] == "user-123"
    assert call_kwargs["user_role"] == "USER"
    assert call_kwargs["service_code"] == "AI_ANALYSIS"
    assert call_kwargs["request_reference"].startswith("AI_ANALYSIS-extraction-user-123-")

    # Xác minh DB lưu đúng owner_id, billing_reference, charged_amount
    assert len(fake_repo.created) == 1
    job_type, input_data = fake_repo.created[0]
    assert job_type == "extraction"
    assert input_data["owner_id"] == "user-123"
    assert input_data["user_role"] == "USER"
    assert input_data["billing_reference"] == "AI_ANALYSIS-ref-001"
    assert input_data["charged_amount"] == 10

    # GET /jobs/{id} với cùng JWT trả 200, với user khác trả 404
    get_ok = client.get(
        f"/jobs/{FIXED_JOB_ID}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_ok.status_code == 200

    other_token = _make_jwt({"sub": "intruder-999", "roles": ["USER"]})
    get_forbidden = client.get(
        f"/jobs/{FIXED_JOB_ID}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert get_forbidden.status_code == 404


def test_describe_async_charges_credits_with_x_user_id(
    client, fake_repo, mock_runner, mock_billing
):
    resp = client.post(
        "/describe-async",
        headers={"X-User-Id": "teacher-88", "X-User-Role": "TEACHER"},
        json={"confirmed_content": "Mục tiêu can thiệp vận động tinh", "tone": "giáo viên"},
    )

    assert resp.status_code == 202
    mock_billing.charge.assert_called_once()
    call_kwargs = mock_billing.charge.call_args.kwargs
    assert call_kwargs["user_id"] == "teacher-88"
    assert call_kwargs["user_role"] == "TEACHER"
    assert call_kwargs["service_code"] == "AI_ANALYSIS"

    assert len(fake_repo.created) == 1
    _, input_data = fake_repo.created[0]
    assert input_data["owner_id"] == "teacher-88"
    assert input_data["billing_reference"] == "AI_ANALYSIS-ref-001"


# --------------------------------------------------------------------------- #
# 3. Negative & Boundary Testing (Rule 3)
# --------------------------------------------------------------------------- #
def test_insufficient_credits_returns_402_and_creates_no_job_in_db(
    client, fake_repo, mock_runner, mock_billing
):
    """Khi user không đủ Credit -> trả 402 Payment Required và KHÔNG tạo job rác trong DB."""
    mock_billing.charge.side_effect = InsufficientCreditsError(
        "Số dư Credit không đủ để sử dụng dịch vụ AI_ANALYSIS", status_code=402
    )

    resp = client.post(
        "/extract-async",
        headers={"X-User-Id": "poor-user-01"},
        json={"intervention_goal": "Phân tích hành vi", "raw_content": ["đoạn 1"]},
    )

    assert resp.status_code == 402
    assert "Số dư Credit không đủ" in resp.json()["detail"]

    # Rule 3.4: Xác minh DB hoàn toàn sạch (0 bản ghi rác) và runner không chạy
    assert fake_repo.created == []
    assert fake_repo.get_job(FIXED_JOB_ID) is None
    mock_runner.run_extraction.assert_not_called()


def test_describe_async_insufficient_credits_returns_402_and_clean_db(
    client, fake_repo, mock_runner, mock_billing
):
    mock_billing.charge.side_effect = InsufficientCreditsError(
        "INSUFFICIENT_BALANCE", status_code=402
    )

    resp = client.post(
        "/describe-async",
        headers={"X-User-Id": "poor-user-02"},
        json={"confirmed_content": "Nội dung hợp lệ"},
    )

    assert resp.status_code == 402
    assert fake_repo.created == []
    mock_runner.run_description.assert_not_called()


def test_empty_or_whitespace_payload_rejected_before_charging_credits(
    client, fake_repo, mock_runner, mock_billing
):
    """Rule 3.1 & 3.2: Payload rỗng/whitespace bị chặn (400) TRƯỚC khi gọi trừ Credit."""
    resp1 = client.post(
        "/extract-async",
        headers={"X-User-Id": "user-123"},
        json={"intervention_goal": "    ", "raw_content": ["a"]},
    )
    assert resp1.status_code == 400

    resp2 = client.post(
        "/describe-async",
        headers={"X-User-Id": "user-123"},
        json={"confirmed_content": "   "},
    )
    assert resp2.status_code == 400

    # Tuyệt đối không gọi charge credit khi input không hợp lệ
    mock_billing.charge.assert_not_called()
    assert fake_repo.created == []


def test_billing_client_validation_boundary_errors():
    """Rule 3.2: Kiểm tra biên các tham số rỗng tại BillingClient."""
    bc = BillingClient(base_url="http://localhost:8000/api")
    with pytest.raises(ValueError, match="user_id"):
        bc.charge(user_id="   ", service_code="AI_ANALYSIS")
    with pytest.raises(ValueError, match="service_code"):
        bc.charge(user_id="user-1", service_code="")
    with pytest.raises(ValueError, match="user_id"):
        bc.refund(user_id="", request_reference="ref-1")
    with pytest.raises(ValueError, match="request_reference"):
        bc.refund(user_id="user-1", request_reference="   ")


# --------------------------------------------------------------------------- #
# 4. Auto-Refund on Job Failure Tests (JobRunner)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_job_runner_extraction_failure_triggers_auto_refund(monkeypatch, fake_repo, mock_billing):
    """Khi job extraction thất bại -> tự động gọi billing_client.refund trả lại Credit."""
    mock_ai = AsyncMock(side_effect=RuntimeError("LLM connection lost"))
    monkeypatch.setattr(job_runner_module, "call_meow_ai_endpoint", mock_ai)

    fake_repo.set_job(
        "job-err-1",
        {
            "input": {
                "intervention_goal": "Mục tiêu A",
                "raw_content": ["text"],
                "owner_id": "user-123",
                "user_role": "USER",
                "billing_reference": "AI_ANALYSIS-ref-001",
                "charged_amount": 10,
            }
        },
    )

    runner = JobRunner(repo=fake_repo)
    await runner.run_extraction("job-err-1")

    assert fake_repo.status_calls == ["processing", "failed"]
    assert fake_repo.error == "LLM connection lost"

    # Xác minh hoàn tiền tự động được kích hoạt đúng thông tin
    mock_billing.refund.assert_called_once()
    refund_kwargs = mock_billing.refund.call_args.kwargs
    assert refund_kwargs["user_id"] == "user-123"
    assert refund_kwargs["request_reference"] == "AI_ANALYSIS-ref-001"
    assert "job-err-1" in refund_kwargs["reason"]


@pytest.mark.asyncio
async def test_job_runner_description_failure_triggers_auto_refund(monkeypatch, fake_repo, mock_billing):
    """Khi job description thất bại -> tự động gọi billing_client.refund trả lại Credit."""
    mock_ai = AsyncMock(side_effect=RuntimeError("Model timeout"))
    monkeypatch.setattr(job_runner_module, "call_meow_ai_endpoint", mock_ai)

    fake_repo.set_job(
        "job-err-2",
        {
            "input": {
                "confirmed_content": "Nội dung B",
                "owner_id": "teacher-88",
                "user_role": "TEACHER",
                "billing_reference": "AI_ANALYSIS-ref-002",
                "charged_amount": 5,
            }
        },
    )

    runner = JobRunner(repo=fake_repo)
    await runner.run_description("job-err-2")

    assert fake_repo.status_calls == ["processing", "failed"]
    mock_billing.refund.assert_called_once()
    refund_kwargs = mock_billing.refund.call_args.kwargs
    assert refund_kwargs["user_id"] == "teacher-88"
    assert refund_kwargs["request_reference"] == "AI_ANALYSIS-ref-002"


@pytest.mark.asyncio
async def test_job_runner_success_does_not_refund_and_records_billing_progress(
    monkeypatch, fake_repo, mock_billing
):
    """Khi job thành công -> KHÔNG gọi refund, ghi nhận billing_reference vào progress."""
    mock_ai = AsyncMock(
        return_value={
            "extracted_content": "Kết quả trích xuất",
            "confidence": 0.95,
            "sources": [],
            "low_confidence": False,
            "batch_count": 1,
            "rounds_used": 1,
        }
    )
    monkeypatch.setattr(job_runner_module, "call_meow_ai_endpoint", mock_ai)

    fake_repo.set_job(
        "job-ok-1",
        {
            "input": {
                "intervention_goal": "Mục tiêu A",
                "raw_content": ["text"],
                "owner_id": "user-123",
                "billing_reference": "AI_ANALYSIS-ref-999",
                "charged_amount": 3,
            }
        },
    )

    runner = JobRunner(repo=fake_repo)
    await runner.run_extraction("job-ok-1")

    assert fake_repo.status_calls == ["processing", "completed"]
    mock_billing.refund.assert_not_called()
    assert fake_repo.progress_calls[-1]["billing_reference"] == "AI_ANALYSIS-ref-999"
    assert fake_repo.progress_calls[-1]["charged_amount"] == 3

