import requests
from typing import Dict, Any, Optional
import logging
from config import Config

# Setup logger
logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class client cho AI agents - Gọi sang meowAI"""
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.api_url = f"{Config.MEOW_AI_URL}/cdd/process-intervention"

    def process(self, input_data: str, context: Dict[str, Any] = None) -> str:
        # Trong kiến trúc client mới, các bước lẻ thường được gộp vào workflow chính
        # Tuy nhiên để giữ logic cũ, ta có thể gọi endpoint chuyên biệt nếu cần.
        # Ở đây ta sẽ giữ các class để không làm gãy code hiện tại.
        raise NotImplementedError("Sử dụng InterventionProcessor.process_intervention_goal để gọi workflow tập trung qua meowAI")

class ExpertAgent(BaseAgent): pass
class EditorAgent(BaseAgent): pass
class PracticalAgent(BaseAgent): pass
class VerifierAgent(BaseAgent): pass
class HTMLFormatterAgent(BaseAgent): pass

class InterventionProcessor:
    """
    Client cho AI Intervention Processor.
    Ủy quyền toàn bộ việc xử lý cho meowAI để tiết kiệm tài nguyên.
    """
    def __init__(self):
        self.api_url = f"{Config.MEOW_AI_URL}/cdd/process-intervention"
    
    def process_intervention_goal(self, intervention_goal: str, context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Gửi yêu cầu xử lý sang meowAI Central Hub
        """
        logger.info(f"[AGENTS] [CLIENT] Gửi mục tiêu can thiệp tới meowAI: {intervention_goal[:50]}...")
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "goal": intervention_goal,
                    "context": context
                },
                timeout=120 # AI processing can be slow
            )
            
            if response.status_code != 200:
                error_msg = f"MeowAI Agents failed ({response.status_code}): {response.text}"
                logger.error(f"[AGENTS] ❌ {error_msg}")
                return {
                    "original_goal": intervention_goal,
                    "error": error_msg,
                    "status": "error"
                }
                
            result = response.json()
            logger.info(f"[AGENTS] ✅ Đã nhận kết quả xử lý từ meowAI")
            return result
            
        except Exception as e:
            logger.error(f"[AGENTS] ❌ Lỗi kết nối meowAI: {str(e)}")
            return {
                "original_goal": intervention_goal,
                "error": str(e),
                "status": "error"
            }
