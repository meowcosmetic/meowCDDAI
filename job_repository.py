"""
JobRepository — lưu trữ job xử lý nội dung can thiệp (async) trong CDD database.

Kết nối CDD database qua `DATABASE_URL` của meowCDDAI (giống `postgres_service.py`).
Bảng `intervention_jobs` lưu trạng thái/tiến trình/kết quả của các job extraction/description
để hỗ trợ mô hình async (trả `job_id` ngay, chạy nền, frontend poll).

Requirement 5: Lưu trữ Job trong CDD Database.
"""

import uuid
import logging

import psycopg2
from psycopg2.extras import RealDictCursor, Json

from config import Config

logger = logging.getLogger(__name__)


class JobRepository:
    """Repository cho bảng `intervention_jobs` trên CDD database."""

    def __init__(self):
        self.database_url = Config.DATABASE_URL
        if not self.database_url:
            params = Config.get_postgres_params()
            self.conn_params = {
                "host": params["host"],
                "port": params["port"],
                "dbname": params["database"],
                "user": params["user"],
                "password": params["password"],
                "sslmode": "prefer",  # prefer for local, require for cloud
            }

    def _get_connection(self):
        if self.database_url:
            # Always go through get_postgres_params() so %27/%20 in db name are unquoted
            params = Config.get_postgres_params()
            return psycopg2.connect(
                host=params["host"],
                port=params["port"],
                dbname=params["database"],
                user=params["user"],
                password=params["password"],
                sslmode="require",
            )
        return psycopg2.connect(**self.conn_params)

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def ensure_table(self):
        """Tạo bảng `intervention_jobs` + index nếu chưa tồn tại (Requirement 5.1)."""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS intervention_jobs (
                    id            UUID PRIMARY KEY,
                    job_type      TEXT NOT NULL,
                    status        TEXT NOT NULL DEFAULT 'pending',
                    input         JSONB NOT NULL,
                    progress      JSONB,
                    result        JSONB,
                    error_message TEXT,
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_intervention_jobs_status
                ON intervention_jobs(status);
                """
            )
            conn.commit()
            cur.close()
            logger.info("[JOB_REPO] ✅ Table intervention_jobs ensured")
        except Exception as e:
            logger.error(f"[JOB_REPO] ❌ Error ensuring table: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------ #
    # Create / Read
    # ------------------------------------------------------------------ #
    def create_job(self, job_type: str, input_data: dict) -> str:
        """Tạo job mới với status=pending, trả về job_id (UUID). (Requirement 5.2)"""
        job_id = str(uuid.uuid4())
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO intervention_jobs (id, job_type, status, input)
                VALUES (%s, %s, 'pending', %s)
                """,
                (job_id, job_type, Json(input_data or {})),
            )
            conn.commit()
            cur.close()
            logger.info(f"[JOB_REPO] ✅ Created job {job_id} (type={job_type})")
            return job_id
        except Exception as e:
            logger.error(f"[JOB_REPO] ❌ Error creating job: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def get_job(self, job_id: str) -> dict | None:
        """Lấy job theo job_id, trả dict tất cả cột (JSONB đã deserialize). (Requirement 5.4)"""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """
                SELECT id, job_type, status, input, progress, result,
                       error_message, created_at, updated_at
                FROM intervention_jobs
                WHERE id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            cur.close()
            if row is None:
                return None
            job = dict(row)
            # Normalize UUID/datetime to serializable primitives
            job["id"] = str(job["id"])
            if job.get("created_at") is not None:
                job["created_at"] = job["created_at"].isoformat()
            if job.get("updated_at") is not None:
                job["updated_at"] = job["updated_at"].isoformat()
            return job
        except Exception as e:
            logger.error(f"[JOB_REPO] ❌ Error getting job {job_id}: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------ #
    # Updates — mỗi lần ghi đều cập nhật updated_at (Requirement 5.3)
    # ------------------------------------------------------------------ #
    def _update_field(self, job_id: str, column: str, value):
        """Helper: cập nhật một cột + updated_at trong một câu lệnh."""
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(
                f"""
                UPDATE intervention_jobs
                SET {column} = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (value, job_id),
            )
            conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"[JOB_REPO] ❌ Error updating {column} for job {job_id}: {str(e)}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def update_status(self, job_id: str, status: str):
        """Cập nhật status + updated_at."""
        self._update_field(job_id, "status", status)

    def update_progress(self, job_id: str, progress: dict):
        """Cập nhật progress (JSONB) + updated_at."""
        self._update_field(job_id, "progress", Json(progress or {}))

    def set_result(self, job_id: str, result: dict):
        """Lưu result (JSONB) + updated_at."""
        self._update_field(job_id, "result", Json(result or {}))

    def set_error(self, job_id: str, message: str):
        """Lưu error_message + updated_at."""
        self._update_field(job_id, "error_message", message)


# Module-level singleton
job_repository = JobRepository()

# Cố gắng tạo bảng lúc import. Nếu DB chưa sẵn sàng, log lỗi rõ ràng nhưng
# KHÔNG làm crash import (Requirement 5.5) — tương tự cách postgres_service
# xử lý lỗi kết nối khi khởi tạo.
try:
    job_repository.ensure_table()
except Exception as e:
    logger.error(
        f"[JOB_REPO] ⚠️ Could not ensure intervention_jobs table at import "
        f"(DB may be unavailable): {str(e)}"
    )
