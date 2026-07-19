"""
Shared LLM client for meowCDDAI.

Cung cấp một điểm gọi LLM dùng chung qua 9router trực tiếp (OpenAI-compatible).
Được dùng bởi router trích lọc (`routes_extraction.py`) và pipeline orchestrator
cho các giai đoạn batch summarize / consensus / sinh mô tả chi tiết.
"""

import logging

import httpx
from fastapi import HTTPException

from config import Config

logger = logging.getLogger(__name__)

# Xử lý lâu (batch + consensus), cần timeout dài.
MEOW_AI_TIMEOUT = 600.0


async def call_meow_ai(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.3,
) -> str:
    """Gọi 9router trực tiếp (OpenAI-compatible) và trả về `content`."""
    url = f"{Config.NINE_ROUTER_BASE_URL}/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": "my-combo",
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {Config.NINE_ROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=MEOW_AI_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="9router service unavailable")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="9router request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"9router returned {e.response.status_code}",
        )
