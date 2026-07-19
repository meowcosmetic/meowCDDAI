import requests
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class AIService:
    """
    Central AI Service for meowCDDAI.
    Proxies all LLM requests to meowAI or meowMessage.
    """
    def __init__(self):
        self.meow_ai_url = Config.MEOW_AI_URL
        logger.info(f"[AI-SERVICE] Initialized pointing to {self.meow_ai_url}")

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a chat request to 9router directly
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            headers = {
                "Authorization": f"Bearer {Config.NINE_ROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                f"{Config.NINE_ROUTER_BASE_URL}/chat/completions",
                json={
                    "model": "my-combo",
                    "messages": messages,
                    "temperature": 0.7
                },
                headers=headers,
                timeout=600
            )

            if response.status_code != 200:
                logger.error(f"[AI-SERVICE] ❌ API Error ({response.status_code}): {response.text}")
                return f"Error from AI Service: {response.text}"

            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

        except Exception as e:
            logger.error(f"[AI-SERVICE] ❌ Connection error: {str(e)}")
            return f"Error connecting to AI Hub: {str(e)}"

# Singleton
ai_service = AIService()
