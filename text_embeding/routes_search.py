from typing import List, Dict, Any
import logging

from fastapi import APIRouter, HTTPException, Query

from models import SearchRequest, SearchResponse, BookPayload, HybridSearchRequest, HybridSearchResponse
from .services import (
    embedding_service,
    qdrant_service,
    hybrid_search_service,
    ensure_keyword_index_if_needed,
)

logger = logging.getLogger(__name__)

def parse_payload(payload: Dict[str, Any]) -> BookPayload:
    """
    Parse payload từ Qdrant và convert sang BookPayload (format mới)
    """
    return BookPayload(
        book_id=payload.get("book_id", "unknown"),
        summary=payload.get("summary"),
        content=payload.get("content", ""),
    )


router = APIRouter()


@router.post("/search", response_model=List[SearchResponse])
async def search_books(search_request: SearchRequest):
    """Tìm kiếm sách dựa trên nội dung (embedding search)"""
    try:
        query_vector = embedding_service.encode_single_text(search_request.query)
        search_results = qdrant_service.search_similar(
            query_vector=query_vector,
            limit=search_request.limit,
            score_threshold=search_request.score_threshold,
        )

        responses: List[SearchResponse] = []
        for result in search_results:
            payload = parse_payload(result.payload)
            response = SearchResponse(
                id=result.id,
                score=result.score,
                embedding_score=result.score,
                keyword_score=None,
                payload=payload,
            )
            responses.append(response)
        return responses
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm: {str(exc)}")


@router.post("/search-keywords", response_model=List[SearchResponse])
async def search_keywords(search_request: SearchRequest):
    """Tìm kiếm sách dựa trên từ khóa (BM25)"""
    try:
        ensure_keyword_index_if_needed()

        keyword_results = hybrid_search_service.search_keywords(
            search_request.query, search_request.limit
        )

        all_points = qdrant_service.get_all_vectors()
        points_dict = {point.id: point for point in all_points}

        responses: List[SearchResponse] = []
        for doc_id, score in keyword_results:
            if doc_id in points_dict:
                payload = parse_payload(points_dict[doc_id].payload)
                response = SearchResponse(
                    id=doc_id,
                    score=score,
                    embedding_score=None,
                    keyword_score=score,
                    payload=payload,
                )
                responses.append(response)
        return responses
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm từ khóa: {str(exc)}")


@router.post("/search-hybrid", response_model=List[HybridSearchResponse])
async def search_hybrid(hybrid_request: HybridSearchRequest):
    """Tìm kiếm hybrid kết hợp keyword và embedding"""
    try:
        ensure_keyword_index_if_needed()

        hybrid_results = hybrid_search_service.hybrid_search(
            query=hybrid_request.query,
            limit=hybrid_request.limit,
            alpha=hybrid_request.alpha,
            beta=hybrid_request.beta,
            score_threshold=hybrid_request.score_threshold,
        )
        return hybrid_results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm hybrid: {str(exc)}")


@router.get("/search-by-book-id/{book_id}", response_model=List[SearchResponse])
async def search_by_book_id(
    book_id: str,
    limit: int = Query(default=50, description="Số lượng kết quả tối đa"),
    score_threshold: float = Query(default=0.0, description="Ngưỡng điểm tối thiểu"),
):
    """Tìm kiếm tất cả đoạn văn của một cuốn sách theo book_id"""
    try:
        all_points = qdrant_service.get_all_vectors()
        filtered_points = [point for point in all_points if point.payload.get("book_id") == book_id]

        if not filtered_points:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy sách với book_id: {book_id}")

        responses: List[SearchResponse] = []
        for point in filtered_points:
            score = 1.0
            if score >= score_threshold:
                payload = parse_payload(point.payload)
                response = SearchResponse(
                    id=point.id,
                    score=score,
                    embedding_score=score,
                    keyword_score=None,
                    payload=payload,
                )
                responses.append(response)

        responses.sort(key=lambda x: x.score, reverse=True)
        return responses[:limit]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm theo book_id: {str(exc)}")


@router.get("/search-by-tags", response_model=List[SearchResponse])
async def search_by_tags(
    tags: str = Query(..., description="Tags cần tìm kiếm, phân cách bằng dấu phẩy"),
    limit: int = Query(default=20, description="Số lượng kết quả tối đa"),
    score_threshold: float = Query(default=0.5, description="Ngưỡng điểm tối thiểu"),
    match_all: bool = Query(default=False, description="Phải match tất cả tags (True) hay chỉ cần match ít nhất 1 tag (False)"),
):
    """Tìm kiếm sách dựa trên tags"""
    try:
        search_tags = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        if not search_tags:
            raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất một tag")

        all_points = qdrant_service.get_all_vectors()

        filtered_points = []
        for point in all_points:
            point_tags = [tag.lower() for tag in point.payload.get("tags", [])]
            if match_all:
                if all(search_tag in point_tags for search_tag in search_tags):
                    filtered_points.append(point)
            else:
                if any(search_tag in point_tags for search_tag in search_tags):
                    filtered_points.append(point)

        if not filtered_points:
            return []

        scored_points = []
        for point in filtered_points:
            point_tags = [tag.lower() for tag in point.payload.get("tags", [])]
            matched_tags = sum(1 for search_tag in search_tags if search_tag in point_tags)
            score = matched_tags / len(search_tags) if search_tags else 0.0
            if score >= score_threshold:
                scored_points.append((point, score))

        responses: List[SearchResponse] = []
        for point, score in scored_points:
            payload = parse_payload(point.payload)
            response = SearchResponse(
                id=point.id,
                score=score,
                embedding_score=score,
                keyword_score=None,
                payload=payload,
            )
            responses.append(response)

        responses.sort(key=lambda x: x.score, reverse=True)
        return responses[:limit]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm theo tags: {str(exc)}")


