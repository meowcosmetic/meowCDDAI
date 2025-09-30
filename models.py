from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class BookPayload(BaseModel):
    book_id: str
    title: str
    author: str
    year: int
    chapter: int
    chapter_title: str
    page: int
    paragraph_index: int
    content: str
    tags: List[str] = []
    language: str = "vi"
    category: str = "Lập trình"

class BookVector(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]
    payload: BookPayload

class BookUploadRequest(BaseModel):
    book_id: str
    title: str
    author: str
    year: int
    content: str
    tags: List[str] = []
    language: str = "vi"
    category: str = "Lập trình"

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    score_threshold: float = 0.7
    alpha: float = 0.7  # Weight for embedding score
    beta: float = 0.3   # Weight for keyword score
    use_hybrid: bool = True  # Enable hybrid search

class HybridSearchRequest(BaseModel):
    query: str
    limit: int = 10
    score_threshold: float = 0.5
    alpha: float = 0.7  # Weight for embedding score
    beta: float = 0.3   # Weight for keyword score
    keyword_fields: List[str] = ["content", "title", "tags"]  # Fields to search with keywords

class SearchResponse(BaseModel):
    id: str
    score: float
    embedding_score: Optional[float] = None
    keyword_score: Optional[float] = None
    payload: BookPayload

class HybridSearchResponse(BaseModel):
    id: str
    hybrid_score: float
    embedding_score: float
    keyword_score: float
    payload: BookPayload
