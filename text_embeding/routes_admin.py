from fastapi import APIRouter, HTTPException, Depends, Header
import os
import secrets

from .services import (
    qdrant_service,
    hybrid_search_service,
    ensure_keyword_index_if_needed,
    keyword_index_needs_rebuild,
)


router = APIRouter()


async def verify_token(authorization: str = Header(None)):
    """Verify service token from Authorization header."""
    token = os.getenv("SERVICE_TOKEN", "")
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    provided = authorization.split(" ", 1)[1]
    if not secrets.compare_digest(provided, token):
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/")
async def root():
    return {"message": "Book Vector Service với Hybrid Search đang hoạt động!"}


@router.get("/health")
async def health_check():
    try:
        collection_info = qdrant_service.get_collection_info()
        return {
            "status": "healthy",
            "collection": collection_info.name,
            "vectors_count": collection_info.vectors_count,
            "features": [
                "embedding_search",
                "keyword_search",
                "hybrid_search",
                "book_id_search",
                "tags_search",
            ],
            "keyword_index_ready": not keyword_index_needs_rebuild,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/delete-book/{book_id}", dependencies=[Depends(verify_token)])
async def delete_book(book_id: str):
    """Xóa tất cả vector của một cuốn sách"""
    global keyword_index_needs_rebuild
    try:
        qdrant_service.delete_book(book_id)
        keyword_index_needs_rebuild = True
        return {"message": f"Đã xóa tất cả vector của sách {book_id}"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/collection-info")
async def get_collection_info():
    """Lấy thông tin về collection"""
    try:
        info = qdrant_service.get_collection_info()
        return {
            "name": info.name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "config": {
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
            },
            "keyword_index_ready": not keyword_index_needs_rebuild,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rebuild-keyword-index", dependencies=[Depends(verify_token)])
async def rebuild_keyword_index():
    """Rebuild keyword index from all vectors in collection"""
    global keyword_index_needs_rebuild
    try:
        all_points = qdrant_service.get_all_vectors()
        if not all_points:
            return {"message": "Không có dữ liệu để rebuild index"}

        all_book_vectors = []
        for point in all_points:
            book_vector = type("BookVector", (), {
                "id": point.id,
                "payload": type("Payload", (), point.payload)(),
            })()
            all_book_vectors.append(book_vector)

        hybrid_search_service.update_index(all_book_vectors)
        keyword_index_needs_rebuild = False

        return {
            "message": "Keyword index đã được rebuild thành công",
            "documents_count": len(all_book_vectors),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal server error")


