"""
GPU Utilities - Loại bỏ code duplication
"""
import logging
import cv2
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class GPUManager:
    """Singleton class để quản lý GPU detection và usage"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize GPU detection - chỉ chạy một lần"""
        if GPUManager._initialized:
            return
        
        self.use_gpu: str = "auto"
        self.gpu_available: bool = False
        self.gpu_device_id: int = 0
        self._detect_gpu()
        GPUManager._initialized = True
    
    def _detect_gpu(self, config_obj=None) -> None:
        """
        Detect GPU availability - loại bỏ duplication
        """
        # Get config
        if config_obj:
            self.use_gpu = config_obj.USE_GPU.lower() if hasattr(config_obj, 'USE_GPU') else "auto"
            self.gpu_device_id = config_obj.GPU_DEVICE_ID if hasattr(config_obj, 'GPU_DEVICE_ID') else 0
        else:
            # Try to import config
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from config import Config
                self.use_gpu = Config.USE_GPU.lower() if hasattr(Config, 'USE_GPU') else "auto"
                self.gpu_device_id = Config.GPU_DEVICE_ID if hasattr(Config, 'GPU_DEVICE_ID') else 0
            except:
                self.use_gpu = "auto"
                self.gpu_device_id = 0
        
        # Check OpenCV CUDA
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.gpu_available = True
                logger.info(f"[GPU] ✅ OpenCV GPU detected: {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
                logger.info(f"[GPU] Using GPU device: {self.gpu_device_id}")
            else:
                logger.info("[GPU] OpenCV không có CUDA support, sử dụng CPU")
        except Exception as e:
            logger.info(f"[GPU] OpenCV GPU check failed: {str(e)}, sử dụng CPU")
        
        # Check PyTorch CUDA (nếu OpenCV GPU không available)
        if not self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available() and (self.use_gpu == "auto" or self.use_gpu == "true"):
                    self.gpu_available = True
                    logger.info(f"[GPU] ✅ PyTorch GPU available: {torch.cuda.get_device_name(0)}")
            except ImportError:
                pass
    
    @property
    def is_available(self) -> bool:
        """Check if GPU is available"""
        return self.gpu_available
    
    @property
    def device_id(self) -> int:
        """Get GPU device ID"""
        return self.gpu_device_id
    
    def get_opencv_backend(self) -> Optional[str]:
        """Get OpenCV backend for GPU if available"""
        if self.gpu_available:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    return "cuda"
            except:
                pass
        return None

