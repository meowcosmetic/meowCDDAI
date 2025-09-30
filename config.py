import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.AVB9V8TSsz1z1mcHYSErTH3vS_tfJdtu_4ixXTHsV1w"
    QDRANT_URL = os.getenv("QDRANT_URL", "https://a9e41cc0-b83e-40b1-9d7a-188d5ffd8629.us-east4-0.gcp.cloud.qdrant.io")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "books")
    MODEL_NAME = "intfloat/multilingual-e5-large"
    
    # Google AI Configuration
    GOOGLE_AI_API_KEY = "AIzaSyB0FiJmN7021PCM4B2EASfAtY_wXh_muVk"