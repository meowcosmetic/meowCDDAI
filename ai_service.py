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
        Send a chat request to meowAI
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                f"{self.meow_ai_url}/chat",
                json={
                    "messages": messages,
                    "provider": "ollama", # Default to ollama as configured in meowAI
                    "temperature": 0.7
                },
                timeout=600
            )

            if response.status_code != 200:
                logger.error(f"[AI-SERVICE] ❌ API Error ({response.status_code}): {response.text}")
                return f"Error from AI Service: {response.text}"

            data = response.json()
            return data.get("content", "")

        except Exception as e:
            logger.error(f"[AI-SERVICE] ❌ Connection error: {str(e)}")
            return f"Error connecting to AI Hub: {str(e)}"

# Singleton
ai_service = AIService()
