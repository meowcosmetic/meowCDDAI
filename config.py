import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "books")
    MODEL_NAME = "intfloat/multilingual-e5-large"
    
    # GPU Configuration
    USE_GPU = os.getenv("USE_GPU", "auto").lower()  # "auto", "true", "false"
    GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))  # Which GPU to use (0, 1, 2, ...)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Batch size for encoding
    
    # Google AI Configuration
    GOOGLE_AI_API_KEY = "AIzaSyB0FiJmN7021PCM4B2EASfAtY_wXh_muVk"