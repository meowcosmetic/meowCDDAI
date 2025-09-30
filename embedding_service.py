from sentence_transformers import SentenceTransformer
from typing import List
import torch
from config import Config

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def encode_text(self, texts: List[str]) -> List[List[float]]:
        """
        Encode text to vectors using multilingual-e5-large model
        """
        # Add prefix for better performance with multilingual-e5-large
        prefixed_texts = [f"query: {text}" for text in texts]
        embeddings = self.model.encode(prefixed_texts, convert_to_tensor=True)
        return embeddings.cpu().numpy().tolist()
    
    def encode_single_text(self, text: str) -> List[float]:
        """
        Encode single text to vector
        """
        return self.encode_text([text])[0]
