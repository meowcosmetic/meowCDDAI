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
from postgres_service import postgres_service

logger = logging.getLogger(__name__)

def parse_payload(payload: Dict[str, Any]) -> BookPayload:
    """
    Parse payload từ Qdrant và convert sang BookPayload (format mới)
    """
    return BookPayload(
        book_id=payload.get("book_id", "unknown"),
        book_name=payload.get("book_name"),
        chapter=payload.get("chapter"),
        page=payload.get("page"),
        postgres_id=payload.get("postgres_id"),
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
            
            # Fetch context if requested
            if search_request.include_context and payload.postgres_id:
                prev_c, next_c = postgres_service.get_neighbor_content(payload.postgres_id)
                payload.prev_content = prev_c
                payload.next_content = next_c

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
                
                # Fetch context if requested
                if search_request.include_context and payload.postgres_id:
                    prev_c, next_c = postgres_service.get_neighbor_content(payload.postgres_id)
                    payload.prev_content = prev_c
                    payload.next_content = next_c

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

        # Fetch context if requested for hybrid results
        if hybrid_request.include_context:
            for result in hybrid_results:
                if result.payload.postgres_id:
                    prev_c, next_c = postgres_service.get_neighbor_content(result.payload.postgres_id)
                    result.payload.prev_content = prev_c
                    result.payload.next_content = next_c
        
        return hybrid_results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm hybrid: {str(exc)}")


@router.get("/search-by-book-id/{book_id}", response_model=List[SearchResponse])
async def search_by_book_id(
    book_id: str,
    limit: int = Query(default=50, description="Số lượng kết quả tối đa"),
    score_threshold: float = Query(default=0.0, description="Ngưỡng điểm tối thiểu"),
):
    """Tìm kiếm tất cả đoạn văn của một cuốn sách theo book_id (server-side filter)."""
    try:
        filtered_points = qdrant_service.scroll_by_filter({"book_id": book_id}, limit=limit)

        if not filtered_points:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy sách với book_id: {book_id}")

        responses: List[SearchResponse] = []
        for point in filtered_points:
            payload = parse_payload(point.payload)
            responses.append(SearchResponse(
                id=point.id, score=1.0, embedding_score=1.0,
                keyword_score=None, payload=payload,
            ))

        return responses[:limit]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm theo book_id: {str(exc)}")


@router.get("/search-by-tags", response_model=List[SearchResponse])
async def search_by_tags(
    tags: str = Query(..., description="Tags cần tìm kiếm, phân cách bằng dấu phẩy"),
    limit: int = Query(default=20, description="Số lượng kết quả tối đa"),
    score_threshold: float = Query(default=0.85, description="Ngưỡng điểm tối thiểu"),
    match_all: bool = Query(default=False, description="Phải match tất cả tags (True) hay chỉ cần match ít nhất 1 tag (False)"),
):
    """Tìm kiếm sách dựa trên tags.

    - match_all=True: Qdrant filter từng tag rồi intersect (server-side).
    - match_any: scroll theo từng tag, union + score bằng Python.
    """
    try:
        search_tags = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
        if not search_tags:
            raise HTTPException(status_code=400, detail="Vui lòng cung cấp ít nhất một tag")

        if match_all:
            # Server-side: scroll theo từng tag rồi intersect
            point_sets = []
            for tag in search_tags:
                pts = qdrant_service.scroll_by_filter({"tags": tag}, limit=1000)
                point_sets.append({p.id for p in pts})
            common_ids = point_sets[0] if point_sets else set()
            for ps in point_sets[1:]:
                common_ids &= ps

            responses: List[SearchResponse] = []
            for point_set in point_sets:
                for pt in (point_set if False else []):
                    pass
            # Re-fetch points with matching ids (scroll doesn't support id filter easily)
            # Fall back: iterate first set and check membership
            first_pts = point_sets[0] if point_sets else []
            for tag in search_tags:
                pts = qdrant_service.scroll_by_filter({"tags": tag}, limit=1000)
                for p in pts:
                    if p.id in common_ids and all(
                        p.id in ps for ps in point_sets
                    ):
                        payload = parse_payload(p.payload)
                        responses.append(SearchResponse(
                            id=p.id, score=1.0, embedding_score=1.0,
                            keyword_score=None, payload=payload,
                        ))
                        common_ids.discard(p.id)  # avoid duplicates
            return responses[:limit]

        else:
            # match_any: union qua từng tag, score = matched_tags / total_tags
            seen: dict = {}  # id -> (point, matched_count)
            for tag in search_tags:
                pts = qdrant_service.scroll_by_filter({"tags": tag}, limit=500)
                for p in pts:
                    if p.id not in seen:
                        seen[p.id] = [p, 0]
                    seen[p.id][1] += 1

            responses = []
            for pt, matched in seen.values():
                score = matched / len(search_tags)
                if score >= score_threshold:
                    payload = parse_payload(pt.payload)
                    responses.append(SearchResponse(
                        id=pt.id, score=score, embedding_score=score,
                        keyword_score=None, payload=payload,
                    ))

            responses.sort(key=lambda x: x.score, reverse=True)
            return responses[:limit]

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tìm kiếm theo tags: {str(exc)}")


