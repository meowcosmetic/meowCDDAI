"""
BillingClient — HTTP client giao tiếp với meowBase Billing & Wallet API.

Hỗ trợ:
  1. Giải mã định danh (userId, userRole) từ JWT Bearer Token hoặc Header X-User-Id / X-User-Role.
  2. Khấu trừ Credit (Pre-charge) qua `POST /api/v1/billing/charge` với mã dịch vụ (vd: `AI_ANALYSIS`).
  3. Hoàn trả Credit tự động (Auto-refund) qua `POST /api/v1/billing/refund` khi tiến trình AI job thất bại.
"""

import base64
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Thứ tự ưu tiên vai trò (từ mức ưu đãi cao nhất xuống mặc định)
_ROLE_PRIORITY = ["ADMIN", "EXPERT", "TEACHER", "COLLABORATOR", "USER"]


class InsufficientCreditsError(Exception):
    """Ném ra khi ví người dùng không đủ số dư Credit (HTTP 402 Payment Required)."""

    def __init__(self, message: str, status_code: int = 402):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BillingServiceError(Exception):
    """Ném ra khi giao tiếp với dịch vụ thanh toán meowBase gặp lỗi."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def normalize_role(raw_role: Optional[str]) -> str:
    """Chuẩn hóa tên role về chuẩn bảng giá meowBase."""
    if not raw_role or not str(raw_role).strip():
        return "USER"
    r = str(raw_role).strip().upper()
    if r.startswith("ROLE_"):
        r = r[5:]
    if r in ("CUSTOMER", "PARENT"):
        return "USER"
    if r in ("SUPER_ADMIN", "SUPERADMIN", "SYSTEM_ADMIN", "STORE_MANAGER"):
        return "ADMIN"
    return r


def _select_primary_role(raw_roles: List[Any]) -> str:
    """Chọn role có độ ưu tiên cao nhất từ danh sách roles trong JWT."""
    normalized = []
    for item in raw_roles:
        if isinstance(item, dict):
            val = item.get("name") or item.get("authority") or item.get("role") or ""
        else:
            val = str(item)
        if val:
            normalized.append(normalize_role(val))

    if not normalized:
        return "USER"

    for priority_role in _ROLE_PRIORITY:
        if priority_role in normalized:
            return priority_role

    return normalized[0]


def decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Giải mã phần payload của JWT (base64url) không phụ thuộc thư viện ngoài."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format")
    payload_b64 = parts[1]
    padding = 4 - (len(payload_b64) % 4)
    if padding != 4:
        payload_b64 += "=" * padding
    decoded_bytes = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
    data = json.loads(decoded_bytes.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JWT payload is not a JSON object")
    return data


def extract_actor_from_headers(
    authorization: Optional[str] = None,
    x_user_id: Optional[str] = None,
    x_user_role: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Trích xuất (user_id, user_role) từ Authorization Bearer JWT hoặc X-User-Id / X-User-Role headers."""
    jwt_user_id: Optional[str] = None
    jwt_role: Optional[str] = None

    if authorization and authorization.strip().lower().startswith("bearer "):
        raw_token = authorization.strip()[7:].strip()
        if raw_token:
            try:
                payload = decode_jwt_payload(raw_token)
                sub = payload.get("sub") or payload.get("userId") or payload.get("id")
                if sub is not None and str(sub).strip():
                    jwt_user_id = str(sub).strip()

                roles_claim = payload.get("roles") or payload.get("authorities")
                if isinstance(roles_claim, list) and roles_claim:
                    jwt_role = _select_primary_role(roles_claim)
                elif isinstance(roles_claim, str) and roles_claim.strip():
                    jwt_role = normalize_role(roles_claim)
                elif payload.get("role"):
                    jwt_role = normalize_role(str(payload.get("role")))
            except Exception as e:
                logger.warning(f"[BILLING] Không thể giải mã JWT header: {e}")

    final_user_id = jwt_user_id
    if not final_user_id and x_user_id and str(x_user_id).strip():
        final_user_id = str(x_user_id).strip()

    if jwt_role:
        final_role = jwt_role
    elif x_user_role and str(x_user_role).strip():
        final_role = normalize_role(x_user_role)
    else:
        final_role = "USER"

    return final_user_id, final_role


class BillingClient:
    """Client đồng bộ/bất đồng bộ gọi meowBase Billing API (`/v1/billing/charge` & `/v1/billing/refund`)."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 10.0):
        env_url = base_url or os.getenv("MEOWBASE_URL")
        if env_url and env_url.strip():
            cleaned = env_url.strip().rstrip("/")
            self.candidate_urls = [cleaned]
        else:
            self.candidate_urls = [
                "http://localhost/api",
                "http://localhost:8000/api",
                "http://meow-base:8000/api",
            ]
        self.timeout = timeout

    def _post_with_fallback(self, subpath: str, payload: Dict[str, Any]) -> httpx.Response:
        last_exc: Optional[Exception] = None
        for base in self.candidate_urls:
            url = f"{base}{subpath}"
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    return client.post(url, json=payload)
            except httpx.RequestError as exc:
                last_exc = exc
                logger.debug(f"[BILLING] Kết nối {url} thất bại ({exc}), thử endpoint tiếp theo...")
        raise BillingServiceError(
            f"Không thể kết nối tới dịch vụ thanh toán meowBase: {last_exc}",
            status_code=503,
        )

    def charge(
        self,
        user_id: str,
        user_role: str = "USER",
        service_code: str = "AI_ANALYSIS",
        request_reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Khấu trừ Credit cho dịch vụ trả phí.

        Raises:
            ValueError: Khi thiếu user_id hoặc service_code.
            InsufficientCreditsError: Khi ví không đủ số dư (HTTP 402).
            BillingServiceError: Khi meowBase trả lỗi khác hoặc mất kết nối.
        """
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id must not be empty for credit charging")
        if not service_code or not str(service_code).strip():
            raise ValueError("service_code must not be empty for credit charging")

        clean_uid = str(user_id).strip()
        clean_role = normalize_role(user_role)
        clean_service = str(service_code).strip().upper()
        ref = (
            str(request_reference).strip()
            if request_reference and str(request_reference).strip()
            else f"{clean_service}-{clean_uid}-{uuid.uuid4().hex[:12]}"
        )

        payload = {
            "userId": clean_uid,
            "userRole": clean_role,
            "serviceCode": clean_service,
            "requestReference": ref,
            "metadata": metadata or {},
        }

        logger.info(
            f"[BILLING] → Charging credits: service={clean_service}, user={clean_uid}, role={clean_role}, ref={ref}"
        )
        resp = self._post_with_fallback("/v1/billing/charge", payload)

        if resp.status_code >= 200 and resp.status_code < 300:
            data = resp.json()
            logger.info(
                f"[BILLING] ✅ Charged {data.get('chargedAmount', 0)} credits for user={clean_uid}, "
                f"remaining={data.get('remainingBalance')}, ref={ref}"
            )
            return data

        body_text = resp.text or ""
        error_msg = body_text
        try:
            err_json = resp.json()
            if isinstance(err_json, dict):
                error_msg = err_json.get("message") or err_json.get("detail") or err_json.get("error") or body_text
        except Exception:
            pass

        if resp.status_code == 402 or "INSUFFICIENT_BALANCE" in body_text:
            logger.warning(f"[BILLING] ⚠️ Insufficient balance for user={clean_uid}, service={clean_service}: {error_msg}")
            raise InsufficientCreditsError(
                error_msg or f"Số dư Credit không đủ để sử dụng dịch vụ {clean_service}",
                status_code=402,
            )

        logger.error(f"[BILLING] ❌ Charge error status={resp.status_code}, body={body_text}")
        raise BillingServiceError(
            error_msg or f"Lỗi khấu trừ Credit (HTTP {resp.status_code})",
            status_code=resp.status_code,
        )

    def refund(
        self,
        user_id: str,
        request_reference: str,
        reason: str = "Auto-refund due to AI job failure",
    ) -> Dict[str, Any]:
        """Hoàn trả Credit khi job AI thất bại."""
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id must not be empty for credit refund")
        if not request_reference or not str(request_reference).strip():
            raise ValueError("request_reference must not be empty for credit refund")

        clean_uid = str(user_id).strip()
        clean_ref = str(request_reference).strip()

        payload = {
            "userId": clean_uid,
            "requestReference": clean_ref,
            "reason": reason,
        }

        logger.info(f"[BILLING] ↩️ Refunding credits for user={clean_uid}, ref={clean_ref}, reason={reason}")
        resp = self._post_with_fallback("/v1/billing/refund", payload)

        if resp.status_code >= 200 and resp.status_code < 300:
            data = resp.json()
            logger.info(
                f"[BILLING] ✅ Refunded {data.get('chargedAmount', 0)} credits for user={clean_uid}, "
                f"remaining={data.get('remainingBalance')}, ref={clean_ref}"
            )
            return data

        body_text = resp.text or ""
        error_msg = body_text
        try:
            err_json = resp.json()
            if isinstance(err_json, dict):
                error_msg = err_json.get("message") or err_json.get("detail") or err_json.get("error") or body_text
        except Exception:
            pass

        logger.error(f"[BILLING] ❌ Refund error status={resp.status_code}, body={body_text}")
        raise BillingServiceError(
            error_msg or f"Lỗi hoàn tiền Credit (HTTP {resp.status_code})",
            status_code=resp.status_code,
        )


billing_client = BillingClient()

