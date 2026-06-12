"""
Shared LLM client for meowCDDAI.

Cung cấp một điểm gọi LLM dùng chung qua meowAI `/chat` (provider=ollama).
Được dùng bởi router trích lọc (`routes_extraction.py`) và pipeline orchestrator
cho các giai đoạn batch summarize / consensus / sinh mô tả chi tiết.

Phân bổ model:
  - MODEL_SMALL (`gemma4:e4b`): trích lọc / batch summarize / consensus loop.
  - MODEL_LARGE (`gemma4:26b`): sinh mô tả chi tiết (4 agents + thảo luận).
"""

import logging

import httpx
from fastapi import HTTPException

from config import Config

logger = logging.getLogger(__name__)

# Central AI Hub URL (meowAI), ví dụ http://meow-ai:8003
MEOW_AI_URL = Config.MEOW_AI_URL

# Xử lý lâu (batch + consensus), cần timeout dài.
MEOW_AI_TIMEOUT = 600.0

# Hằng số model phân bổ theo giai đoạn.
MODEL_SMALL = "gemma4:e4b"   # extraction / batch summarize / consensus
MODEL_LARGE = "gemma4:26b"   # description generation (4 agents)


async def call_meow_ai(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.3,
) -> str:
    """Gọi meowAI `/chat` (route tới Ollama) và trả về `content`.

    Args:
        system: System prompt (có thể rỗng).
        user: User prompt.
        model: Tên model Ollama (vd `MODEL_SMALL`, `MODEL_LARGE`).
        temperature: Nhiệt độ sinh, mặc định 0.3.

    Returns:
        Chuỗi `content` từ phản hồi của meowAI (rỗng nếu không có).

    Raises:
        HTTPException: 503 khi không kết nối được meowAI, 504 khi timeout,
            502 khi meowAI trả mã lỗi HTTP.
    """
    url = f"{MEOW_AI_URL}/chat"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "provider": "ollama",
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=MEOW_AI_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("content", "")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="meowAI service unavailable")
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="meowAI request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"meowAI returned {e.response.status_code}",
        )
