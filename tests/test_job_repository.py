"""Unit tests for JobRepository (meowCDDAI/job_repository.py).

Validates: Requirements 5.2, 5.4 — design component "B.1 Job Repository".

Hermetic strategy
-----------------
JobRepository talks to a real Postgres via psycopg2 and, at import time, the
module instantiates a singleton and calls `ensure_table()`. That import-time
call is already guarded by a try/except in the module (Requirement 5.5), so the
module imports cleanly even with no DB available.

For the test methods we never touch a real DB. Instead we patch
`job_repository.psycopg2.connect` with a factory that returns a *fake*
connection backed by a tiny in-memory store. The fake cursor interprets the
exact SQL the repository issues (INSERT / SELECT / UPDATE / CREATE) just enough
to exercise round-trip behavior:

- It records executed SQL + params (so we can assert UPDATEs touch updated_at).
- It unwraps psycopg2 `Json(...)` adapters back into plain dicts, which lets us
  assert JSONB fields (input / progress / result) survive as `dict in -> dict out`.
- SELECT returns a scripted RealDictCursor-style row (or None when absent).

All connections created during one test share the same store, so a value
written by `create_job` is visible to a later `get_job` even though each
repository method opens its own connection.
"""

import datetime
import re
import uuid
from unittest.mock import patch

import pytest
from psycopg2.extras import Json, RealDictCursor

import job_repository
from job_repository import JobRepository


# --------------------------------------------------------------------------- #
# In-memory fake psycopg2 layer
# --------------------------------------------------------------------------- #
def _unwrap(value):
    """Unwrap a psycopg2 Json adapter to its underlying Python object."""
    if isinstance(value, Json):
        return value.adapted
    return value


class FakeStore:
    """Shared in-memory backing store for all fake connections in a test."""

    def __init__(self):
        self.jobs = {}
        self._tick = 0

    def now(self):
        # Monotonically increasing timestamps so we can assert updated_at moved.
        self._tick += 1
        return datetime.datetime(2024, 1, 1) + datetime.timedelta(seconds=self._tick)


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self._result = None
        self.executed = []  # list of (normalized_sql, params)

    def execute(self, sql, params=None):
        s = " ".join(sql.split())  # normalize whitespace/newlines
        params = tuple(params) if params else ()
        self.executed.append((s, params))

        if s.startswith("CREATE TABLE") or s.startswith("CREATE INDEX"):
            return

        if s.startswith("INSERT INTO intervention_jobs"):
            job_id, job_type, input_json = params
            now = self.store.now()
            self.store.jobs[str(job_id)] = {
                "id": job_id,
                "job_type": job_type,
                "status": "pending",
                "input": _unwrap(input_json),
                "progress": None,
                "result": None,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
            return

        if s.startswith("UPDATE intervention_jobs"):
            m = re.search(r"SET (\w+) =", s)
            assert m, f"Could not find updated column in: {s}"
            assert "updated_at = CURRENT_TIMESTAMP" in s, (
                f"UPDATE must also bump updated_at: {s}"
            )
            column = m.group(1)
            value, job_id = params
            job = self.store.jobs.get(str(job_id))
            if job is not None:
                job[column] = _unwrap(value)
                job["updated_at"] = self.store.now()
            return

        if s.startswith("SELECT"):
            job_id = params[-1]
            job = self.store.jobs.get(str(job_id))
            self._result = dict(job) if job is not None else None
            return

        raise AssertionError(f"Unexpected SQL executed: {s}")

    def fetchone(self):
        return self._result

    def close(self):
        pass


class FakeConnection:
    def __init__(self, store):
        self.store = store
        self.commits = 0
        self.rollbacks = 0
        self.last_cursor = None

    def cursor(self, cursor_factory=None):
        # cursor_factory may be RealDictCursor for get_job; behavior is identical
        # for our fake because we already return plain dict rows.
        cur = FakeCursor(self.store)
        self.last_cursor = cur
        return cur

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


@pytest.fixture
def repo_and_store():
    """A JobRepository whose psycopg2.connect is patched to a shared fake store."""
    store = FakeStore()

    def fake_connect(*args, **kwargs):
        return FakeConnection(store)

    with patch.object(job_repository.psycopg2, "connect", side_effect=fake_connect):
        repo = JobRepository()
        yield repo, store


# --------------------------------------------------------------------------- #
# ensure_table
# --------------------------------------------------------------------------- #
def test_ensure_table_creates_table_and_index(repo_and_store):
    repo, _store = repo_and_store
    # Should not raise; issues CREATE TABLE + CREATE INDEX without a real DB.
    repo.ensure_table()


# --------------------------------------------------------------------------- #
# create -> get round-trip (Requirements 5.2, 5.4)
# --------------------------------------------------------------------------- #
def test_create_job_returns_uuid_string(repo_and_store):
    repo, _store = repo_and_store
    job_id = repo.create_job("extraction", {"intervention_goal": "g"})

    assert isinstance(job_id, str)
    # Parsing as UUID confirms it is a valid v4-style identifier string.
    assert str(uuid.UUID(job_id)) == job_id


def test_create_then_get_round_trip(repo_and_store):
    repo, _store = repo_and_store
    input_data = {
        "intervention_goal": "Cải thiện giao tiếp",
        "raw_content": ["đoạn 1", "đoạn 2"],
        "tone": "giáo viên",
    }
    job_id = repo.create_job("extraction", input_data)

    job = repo.get_job(job_id)
    assert job is not None
    assert job["id"] == job_id
    assert job["job_type"] == "extraction"
    assert job["status"] == "pending"
    # JSONB input survives as a dict, identical to what went in.
    assert job["input"] == input_data
    # created_at / updated_at normalized to ISO strings by get_job.
    assert isinstance(job["created_at"], str)
    assert isinstance(job["updated_at"], str)


def test_get_job_returns_none_when_absent(repo_and_store):
    repo, _store = repo_and_store
    assert repo.get_job(str(uuid.uuid4())) is None


# --------------------------------------------------------------------------- #
# update_status (Requirement 5.3 — updated_at bumped)
# --------------------------------------------------------------------------- #
def test_update_status_changes_status_and_bumps_updated_at(repo_and_store):
    repo, store = repo_and_store
    job_id = repo.create_job("extraction", {"intervention_goal": "g"})

    created_updated_at = store.jobs[job_id]["updated_at"]
    repo.update_status(job_id, "processing")

    job = repo.get_job(job_id)
    assert job["status"] == "processing"
    # updated_at must have advanced after the write.
    assert store.jobs[job_id]["updated_at"] > created_updated_at


# --------------------------------------------------------------------------- #
# update_progress JSONB round-trip
# --------------------------------------------------------------------------- #
def test_update_progress_round_trips_dict(repo_and_store):
    repo, _store = repo_and_store
    job_id = repo.create_job("extraction", {"intervention_goal": "g"})

    progress = {"phase": "extracting", "message": "Đang trích lọc", "batch_count": 3}
    repo.update_progress(job_id, progress)

    job = repo.get_job(job_id)
    assert job["progress"] == progress
    assert isinstance(job["progress"], dict)


# --------------------------------------------------------------------------- #
# set_result JSONB round-trip (Requirement 5.4)
# --------------------------------------------------------------------------- #
def test_set_result_round_trips_nested_dict(repo_and_store):
    repo, _store = repo_and_store
    job_id = repo.create_job("description", {"confirmed_content": "c"})

    result = {
        "expert_analysis": "a",
        "practical_content": "b",
        "verified_content": "c",
        "final_content": "<p>html</p>",
        "workflow_summary": {"step_1": "1", "step_2": "2"},
        "low_confidence": False,
        "sources": [{"book_name": "B", "page": 12, "score": 0.83}],
    }
    repo.set_result(job_id, result)

    job = repo.get_job(job_id)
    assert job["result"] == result
    # Nested structures survive as dicts/lists (true JSONB round-trip).
    assert job["result"]["workflow_summary"] == {"step_1": "1", "step_2": "2"}
    assert job["result"]["sources"][0]["page"] == 12


# --------------------------------------------------------------------------- #
# set_error
# --------------------------------------------------------------------------- #
def test_set_error_stores_message(repo_and_store):
    repo, _store = repo_and_store
    job_id = repo.create_job("extraction", {"intervention_goal": "g"})

    repo.set_error(job_id, "meowAI 503: unavailable")

    job = repo.get_job(job_id)
    assert job["error_message"] == "meowAI 503: unavailable"


# --------------------------------------------------------------------------- #
# Full lifecycle: create -> processing -> result -> completed
# --------------------------------------------------------------------------- #
def test_full_lifecycle_round_trip(repo_and_store):
    repo, _store = repo_and_store
    job_id = repo.create_job("extraction", {"intervention_goal": "g", "raw_content": []})

    repo.update_status(job_id, "processing")
    repo.update_progress(job_id, {"phase": "extracting"})
    repo.set_result(job_id, {"extracted_content": "x", "confidence": 0.9})
    repo.update_status(job_id, "completed")

    job = repo.get_job(job_id)
    assert job["status"] == "completed"
    assert job["progress"] == {"phase": "extracting"}
    assert job["result"] == {"extracted_content": "x", "confidence": 0.9}
    assert job["error_message"] is None
