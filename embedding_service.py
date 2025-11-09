from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
import logging
from datetime import datetime
from config import Config

# Setup logger
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        logger.info(f"[EMBEDDING] Khởi tạo EmbeddingService")
        logger.info(f"[EMBEDDING] Model: {Config.MODEL_NAME}")
        
        # Determine device (this will set self.device_id)
        self.device = self._get_device()
        self.batch_size = Config.BATCH_SIZE
        
        # Log GPU info if available
        if "cuda" in self.device and self.device_id is not None:
            logger.info(f"[EMBEDDING] GPU Device: {self.device}")
            logger.info(f"[EMBEDDING] GPU Name: {torch.cuda.get_device_name(self.device_id)}")
            logger.info(f"[EMBEDDING] GPU Memory: {torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3:.2f} GB")
        else:
            logger.info(f"[EMBEDDING] Using CPU")
        
        logger.info(f"[EMBEDDING] Batch size: {self.batch_size}")
        logger.info(f"[EMBEDDING] Đang load model... (có thể mất vài phút lần đầu)")
        
        # Load model
        self.model = SentenceTransformer(Config.MODEL_NAME, device=self.device)
        
        logger.info(f"[EMBEDDING] ✅ Model đã load, device: {self.device}")
    
    def _get_device(self) -> str:
        """
        Determine which device to use based on config
        Sets self.device_id if GPU is available
        """
        if Config.USE_GPU == "false":
            logger.info("[EMBEDDING] GPU bị tắt theo config (USE_GPU=false)")
            self.device_id = None
            return "cpu"
        
        if Config.USE_GPU == "true" or Config.USE_GPU == "auto":
            if torch.cuda.is_available():
                self.device_id = Config.GPU_DEVICE_ID
                if self.device_id >= torch.cuda.device_count():
                    logger.warning(f"[EMBEDDING] GPU device {self.device_id} không tồn tại, dùng device 0")
                    self.device_id = 0
                device = f"cuda:{self.device_id}"
                logger.info(f"[EMBEDDING] ✅ GPU được bật: {device}")
                return device
            else:
                if Config.USE_GPU == "true":
                    logger.warning("[EMBEDDING] USE_GPU=true nhưng không tìm thấy GPU, chuyển sang CPU")
                else:
                    logger.info("[EMBEDDING] Không tìm thấy GPU, sử dụng CPU")
                self.device_id = None
                return "cpu"
        
        self.device_id = None
        return "cpu"
    
    def encode_text(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Encode text to vectors using multilingual-e5-large model
        Optimized for GPU with batch processing
        """
        if not texts:
            logger.warning("[EMBEDDING] Danh sách texts rỗng")
            return []
        
        batch_size = batch_size or self.batch_size
        logger.info(f"[EMBEDDING] Bắt đầu encode {len(texts)} texts... (batch_size={batch_size}, device={self.device})")
        start_time = datetime.now()
        
        # Add prefix for better performance with multilingual-e5-large
        prefixed_texts = [f"query: {text}" for text in texts]
        logger.debug(f"[EMBEDDING] Text lengths: {[len(t) for t in texts[:5]]}{'...' if len(texts) > 5 else ''}")
        
        # Use batch processing and device optimization
        encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": len(texts) > 10,  # Show progress for large batches
            "convert_to_tensor": True,
            "device": self.device,
            "normalize_embeddings": False,
        }
        
        # If on GPU, use better memory management
        if "cuda" in self.device and self.device_id is not None:
            # Clear cache before encoding
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug(f"[EMBEDDING] GPU memory before: {torch.cuda.memory_allocated(self.device_id) / 1024**2:.2f} MB")
        
        try:
            embeddings = self.model.encode(prefixed_texts, **encode_kwargs)
            
            # Move to CPU and convert to list
            if isinstance(embeddings, torch.Tensor):
                result = embeddings.cpu().numpy().tolist()
            else:
                result = embeddings.tolist()
            
            # Clear GPU cache after encoding
            if "cuda" in self.device and self.device_id is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug(f"[EMBEDDING] GPU memory after: {torch.cuda.memory_allocated(self.device_id) / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"[EMBEDDING] ❌ Lỗi khi encode: {str(e)}", exc_info=True)
            # Clear cache on error
            if "cuda" in self.device and self.device_id is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        
        elapsed = (datetime.now() - start_time).total_seconds()
        vector_dim = len(result[0]) if result else 0
        logger.info(f"[EMBEDDING] ✅ Đã encode {len(result)} texts → {vector_dim}D vectors ({elapsed:.2f}s)")
        logger.info(f"[EMBEDDING] Throughput: {len(texts)/elapsed:.2f} texts/second")
        logger.debug(f"[EMBEDDING] Avg time per text: {elapsed/len(texts):.3f}s")
        
        return result
    
    def encode_single_text(self, text: str) -> List[float]:
        """
        Encode single text to vector
        """
        return self.encode_text([text])[0]
