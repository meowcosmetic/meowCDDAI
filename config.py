import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "books")
    MODEL_NAME = "intfloat/multilingual-e5-large"
    
    # Central AI Hub URL
    MEOW_AI_URL = os.getenv("MEOW_AI_URL", "http://meow-ai:8003")
    
    # GPU Configuration
    USE_GPU = os.getenv("USE_GPU", "auto").lower()  # "auto", "true", "false"
    GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))  # Which GPU to use (0, 1, 2, ...)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Batch size for encoding
    
    # Google AI Configuration (moved to meowAI - use via meowAI /chat endpoint with provider="google")
    # GOOGLE_AI_API_KEY is no longer needed here
    
    # Local LLM Configuration
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    LLM_TYPE = os.getenv("LLM_TYPE", "ollama").lower()  # "ollama" hoặc "openai-compatible"
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen3:8b")  # Tên model trong Ollama
    
    # OpenAI-compatible API Configuration (nếu dùng vLLM, llama.cpp server, etc.)
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "gpt-20b")
    LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "not-needed")

    # Postgres Configuration
    DATABASE_URL = os.getenv("DATABASE_URL")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "host.docker.internal")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "MeowCDD")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "cdd_app_admin")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "cdd_app_admin")
    
    @classmethod
    def get_postgres_params(cls):
        """Parse DATABASE_URL or use individual params"""
        if cls.DATABASE_URL:
            # Handle standard postgres:// or postgresql:// URLs using standard parser
            try:
                from urllib.parse import urlparse, unquote
                result = urlparse(cls.DATABASE_URL)
                
                # Unquote is critical for database names with special characters like %27 (quote)
                db_name = unquote(result.path[1:]) if result.path else cls.POSTGRES_DB
                
                return {
                    "host": result.hostname or cls.POSTGRES_HOST,
                    "port": result.port or cls.POSTGRES_PORT,
                    "database": db_name,
                    "user": result.username or cls.POSTGRES_USER,
                    "password": result.password or cls.POSTGRES_PASSWORD
                }
            except Exception as e:
                print(f"Error parsing DATABASE_URL: {e}")
        
        return {
            "host": cls.POSTGRES_HOST,
            "port": cls.POSTGRES_PORT,
            "database": cls.POSTGRES_DB,
            "user": cls.POSTGRES_USER,
            "password": cls.POSTGRES_PASSWORD
        }
