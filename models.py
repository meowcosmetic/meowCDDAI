from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any, Dict
import uuid

class BookPayload(BaseModel):
    """Payload cho format JSON upload mới"""
    book_id: str
    summary: Optional[str] = None
    content: str
    book_name: Optional[str] = None
    chapter: Optional[str] = None
    page: Optional[float] = None
    postgres_id: Optional[int] = None
    prev_content: Optional[str] = None
    next_content: Optional[str] = None
    link_refs: Optional[List[Dict[str, Any]]] = None
    record_type: Optional[str] = None
    source_book_number: Optional[int] = None
    domain: Optional[str] = None
    skill_codes: Optional[List[str]] = None
    age_ranges: Optional[List[str]] = None
    linked_book_ids: Optional[List[str]] = None
    linked_sections: Optional[List[str]] = None
    table_page: Optional[float] = None
    pdf_physical_page: Optional[int] = None
    printed_page: Optional[int] = None
    printed_page_candidates: Optional[List[int]] = None
    visual_assets: Optional[Dict[str, Any]] = None
    extraction_status: Optional[str] = None
    needs_visual_review: Optional[bool] = None
    source_json_row_indices: Optional[List[int]] = None
    source_match_score: Optional[float] = None
    content_sha256: Optional[str] = None
    source_type: Optional[str] = None
    source_notice: Optional[str] = None

class MeowBookItem(BaseModel):
    """Mô hình dữ liệu cho sách Meow CDD.

    Qdrant chỉ index một vector ``content``. ``summary`` là metadata text,
    không phải một named vector thứ hai.
    """
    Book: Optional[str] = ""
    Chapter: Optional[str] = ""
    Page: Optional[float] = 0.0
    Content: Optional[str] = ""
    Summary: Optional[str] = ""
    Link: Optional[str] = ""
    CleanedContent: Optional[str] = ""
    SearchQueries: Optional[List[str]] = []
    link_refs: Optional[List[Dict[str, Any]]] = None
    record_type: Optional[str] = None
    source_book_number: Optional[int] = None
    domain: Optional[str] = None
    skill_codes: Optional[List[str]] = None
    age_ranges: Optional[List[str]] = None
    linked_book_ids: Optional[List[str]] = None
    linked_sections: Optional[List[str]] = None
    table_page: Optional[float] = None
    Debug_RawCleaner: Optional[str] = None
    Debug_RawQuery: Optional[str] = None
    pdf_physical_page: Optional[int] = None
    printed_page: Optional[int] = None
    printed_page_candidates: Optional[List[int]] = None
    visual_assets: Optional[Dict[str, Any]] = None
    extraction_status: Optional[str] = None
    needs_visual_review: Optional[bool] = None
    source_json_row_indices: Optional[List[int]] = None
    source_match_score: Optional[float] = None
    content_sha256: Optional[str] = None
    source_type: Optional[str] = None
    source_notice: Optional[str] = None

    @field_validator('Page', mode='before')
    @classmethod
    def parse_page(cls, v: Any) -> float:
        if v == "" or v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

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
    score_threshold: float = 0.85
    alpha: float = 0.7  # Weight for embedding score
    beta: float = 0.3   # Weight for keyword score
    use_hybrid: bool = True  # Enable hybrid search
    include_context: bool = False  # Trả về đoạn trước và sau

class HybridSearchRequest(BaseModel):
    query: str
    limit: int = 10
    score_threshold: float = 0.85
    alpha: float = 0.7  # Weight for embedding score
    beta: float = 0.3   # Weight for keyword score
    keyword_fields: List[str] = ["content", "title", "tags"]  # Fields to search with keywords
    include_context: bool = False  # Trả về đoạn trước và sau

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
