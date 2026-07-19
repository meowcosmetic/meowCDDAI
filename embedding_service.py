import requests
from typing import List, Optional
import logging
from datetime import datetime
from config import Config

# Setup logger
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Client for meowAI Embedding Service.
    Sends text to meowAI for encoding instead of loading models locally.

    E5 model uses asymmetric prefixes:
      - "passage:" for indexing documents into Qdrant
      - "query:"   for search queries
    """
    def __init__(self):
        logger.info(f"[EMBEDDING] Khởi tạo Embedding Client (Connecting to {Config.MEOW_AI_URL})")
        self.api_url = f"{Config.MEOW_AI_URL}/embeddings"
        self.batch_size = Config.BATCH_SIZE
        logger.info(f"[EMBEDDING] ✅ Client đã sẵn sàng, trạm đích: {self.api_url}")

    def encode_text(self, texts: List[str], batch_size: Optional[int] = None, task_type: str = "query") -> List[List[float]]:
        """
        Encode text to vectors by calling meowAI API.

        Args:
            texts: List of texts to encode.
            batch_size: Batch size for processing.
            task_type: "query" for search, "passage" for indexing into Qdrant.
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size
        logger.info(f"[EMBEDDING] [CLIENT] Gửi {len(texts)} texts tới meowAI (task_type={task_type})...")
        start_time = datetime.now()

        try:
            response = requests.post(
                self.api_url,
                json={
                    "texts": texts,
                    "batch_size": batch_size,
                    "task_type": task_type,
                },
                timeout=60
            )

            if response.status_code != 200:
                logger.error(f"[EMBEDDING] ❌ Lỗi API meowAI ({response.status_code}): {response.text}")
                raise Exception(f"MeowAI Embedding failed: {response.text}")

            data = response.json()
            result = data.get("embeddings", [])

            elapsed = (datetime.now() - start_time).total_seconds()
            vector_dim = len(result[0]) if result else 0
            logger.info(f"[EMBEDDING] ✅ Đã nhận {len(result)} vectors ({vector_dim}D) từ meowAI ({elapsed:.2f}s)")
            return result

        except Exception as e:
            logger.error(f"[EMBEDDING] ❌ Lỗi kết nối meowAI: {str(e)}")
            raise

    def encode_text_passage(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Encode documents for indexing into Qdrant (passage prefix)."""
        return self.encode_text(texts, batch_size, task_type="passage")

    def encode_text_query(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Encode search queries (query prefix)."""
        return self.encode_text(texts, batch_size, task_type="query")

    def encode_single_text(self, text: str) -> List[float]:
        """Encode single search query (query prefix)."""
        return self.encode_text_query([text])[0]
