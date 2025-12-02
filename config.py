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
    
    # Local LLM Configuration
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    LLM_TYPE = os.getenv("LLM_TYPE", "ollama").lower()  # "ollama" hoặc "openai-compatible"
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M")  # Tên model trong Ollama
    
    # OpenAI-compatible API Configuration (nếu dùng vLLM, llama.cpp server, etc.)
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "gpt-20b")
    LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "not-needed")