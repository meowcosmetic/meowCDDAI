from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import json
import hashlib
import logging
from datetime import datetime
from qdrant_client.models import PointStruct

from models import BookUploadRequest, MeowBookItem
from .services import (
    embedding_service,
    qdrant_service,
    keyword_index_needs_rebuild,
)
from postgres_service import postgres_service

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


router = APIRouter()


def _stable_point_id(book_id: str, book_name: str, chapter: str, page: float, content: str) -> str:
    """Create a deterministic Qdrant point id for retry-safe book uploads."""
    identity = "|".join([
        str(book_id),
        str(book_name),
        str(chapter),
        str(page),
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    ])
    # Qdrant accepts UUID strings; use the first 32 hex chars as a UUID-shaped id.
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _summary_for_item(item: MeowBookItem) -> str:
    """Return explicit summary metadata without creating a second vector."""
    if item.Summary and item.Summary.strip():
        return item.Summary.strip()
    return f"{item.Book or 'Unknown'} - {item.Chapter or ''} (Trang {item.Page or 0.0})"


@router.post("/upload-book-file")
async def upload_book_file(
    file: UploadFile = File(...),
    book_id: str = Form(None, description="ID của sách"),
):
    """Upload sách từ file .json (MeowBook format hoặc legacy summary/content)."""
    global keyword_index_needs_rebuild

    logger.info(f"[UPLOAD] Bắt đầu upload file: {file.filename}")
    start_time = datetime.now()

    try:
        logger.info(f"[UPLOAD] Đang đọc file: {file.filename}")
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        content_str = content.decode("utf-8")
        logger.info(f"[UPLOAD] Đã đọc file: {len(content_str)} bytes")

        # New: JSON upload without chunking
        if file.filename.endswith(".json"):
            logger.info(f"[UPLOAD] Xử lý file JSON với book_id: {book_id}")
            
            try:
                logger.info("[UPLOAD] Đang parse JSON...")
                items_raw = json.loads(content_str)
                logger.info(f"[UPLOAD] Parse thành công: {len(items_raw)} items")
            except Exception as e:
                logger.error(f"[UPLOAD] Lỗi parse JSON: {str(e)}")
                raise HTTPException(status_code=400, detail="File JSON không hợp lệ")

            if not isinstance(items_raw, list) or not items_raw:
                logger.error(f"[UPLOAD] JSON không phải array hoặc rỗng")
                raise HTTPException(status_code=400, detail="JSON phải là một mảng các object")

            # Check if it's the new MeowBook format or the old summary/content format
            is_meow_format = all(key in items_raw[0] for key in ["Book", "CleanedContent"]) if items_raw else False

            if is_meow_format:
                logger.info("[UPLOAD] Phát hiện format MeowBook mới")
                meow_items = []
                for idx, item in enumerate(items_raw):
                    try:
                        meow_items.append(MeowBookItem(**item))
                    except Exception as e:
                        logger.error(f"[UPLOAD] Item {idx} không đúng format MeowBook: {str(e)}")
                        raise HTTPException(status_code=400, detail=f"Dữ liệu tại index {idx} không hợp lệ")

                # Save to Postgres
                logger.info(f"[UPLOAD] Đang lưu {len(meow_items)} items vào Postgres...")
                pg_ids = postgres_service.insert_book_items(meow_items)
                logger.info("[UPLOAD] ✅ Lưu vào Postgres thành công")

                # Prepare for Qdrant — single "content" vector
                cleaned_contents = [item.CleanedContent or "" for item in meow_items]

                logger.info("[UPLOAD] Đang tạo embeddings cho Qdrant (passage prefix)...")
                embeddings = embedding_service.encode_text_passage(cleaned_contents)

                points = []
                for i, item in enumerate(meow_items):
                    payload = {
                        "book_id": book_id or item.Book or "Unknown",
                        "book_name": item.Book or "Unknown",
                        "chapter": item.Chapter or "",
                        "page": item.Page or 0.0,
                        "summary": _summary_for_item(item),
                        "content": item.CleanedContent or "",
                        "postgres_id": pg_ids[i] if pg_ids and i < len(pg_ids) else None,
                        "link_refs": item.link_refs or [],
                        "record_type": item.record_type,
                        "source_book_number": item.source_book_number,
                        "domain": item.domain,
                        "skill_codes": item.skill_codes or [],
                        "age_ranges": item.age_ranges or [],
                        "linked_book_ids": item.linked_book_ids or [],
                        "linked_sections": item.linked_sections or [],
                        "table_page": item.table_page or item.Page or 0.0,
                        "pdf_physical_page": item.pdf_physical_page,
                        "printed_page": item.printed_page,
                        "printed_page_candidates": item.printed_page_candidates or [],
                        "visual_assets": item.visual_assets or {},
                        "extraction_status": item.extraction_status,
                        "needs_visual_review": item.needs_visual_review or False,
                        "source_json_row_indices": item.source_json_row_indices or [],
                        "source_match_score": item.source_match_score,
                        "content_sha256": item.content_sha256,
                        "source_type": item.source_type,
                        "source_notice": item.source_notice,
                    }
                    point_id = _stable_point_id(
                        payload["book_id"], payload["book_name"], payload["chapter"],
                        payload["page"], payload["content"],
                    )
                    point = PointStruct(
                        id=point_id,
                        vector={"content": embeddings[i]},
                        payload=payload
                    )
                    points.append(point)
                
                logger.info(f"[UPLOAD] Đang upsert {len(points)} points vào Qdrant...")
                try:
                    vector_ids = qdrant_service.upsert_named_points(points)
                except Exception:
                    # Compensating cleanup prevents a failed vector write from
                    # leaving Postgres rows that cannot provide valid neighbors.
                    postgres_service.delete_book_items([x for x in pg_ids or [] if x is not None])
                    raise

                keyword_index_needs_rebuild = True
                
                return {
                    "message": "Upload MeowBook thành công",
                    "postgres_count": len(meow_items),
                    "qdrant_count": len(vector_ids),
                    "book_name": meow_items[0].Book if meow_items else "Unknown"
                }

            else:
                # Old format: summary/content — embed content + lưu Postgres
                logger.info(f"[UPLOAD] Xử lý format summary/content cũ")
                if not book_id:
                    logger.error("[UPLOAD] Thiếu tham số book_id cho format cũ")
                    raise HTTPException(status_code=400, detail="Thiếu tham số book_id")

                # Lưu vào Postgres để có postgres_id cho neighbor lookup
                meow_items = []
                for item in items_raw:
                    meow_items.append(MeowBookItem(
                        Book=item.get("book_name", book_id),
                        Chapter=item.get("chapter", ""),
                        Page=item.get("page", 0.0),
                        Content=item.get("content", ""),
                        CleanedContent=item.get("content", ""),
                    ))

                logger.info(f"[UPLOAD] Đang lưu {len(meow_items)} items vào Postgres...")
                pg_ids = postgres_service.insert_book_items(meow_items)
                logger.info("[UPLOAD] ✅ Lưu vào Postgres thành công")

                contents = [item.get("content", "") for item in items_raw]
                content_vectors = embedding_service.encode_text_passage(contents)

                points = []
                for i in range(len(items_raw)):
                    payload = {
                        "book_id": book_id,
                        "book_name": items_raw[i].get("book_name", book_id),
                        "chapter": items_raw[i].get("chapter", ""),
                        "page": items_raw[i].get("page", 0.0),
                        "summary": items_raw[i].get("summary", ""),
                        "content": contents[i],
                        "postgres_id": pg_ids[i] if pg_ids and i < len(pg_ids) else None,
                        "link_refs": [],
                        "record_type": None,
                        "source_book_number": None,
                        "domain": None,
                        "skill_codes": [],
                        "age_ranges": [],
                        "linked_book_ids": [],
                        "linked_sections": [],
                        "table_page": items_raw[i].get("page", 0.0),
                    }
                    point_id = _stable_point_id(
                        payload["book_id"], payload["book_name"], payload["chapter"],
                        payload["page"], payload["content"],
                    )
                    point = PointStruct(
                        id=point_id,
                        vector={"content": content_vectors[i]},
                        payload=payload,
                    )
                    points.append(point)

                try:
                    vector_ids = qdrant_service.upsert_named_points(points)
                except Exception:
                    postgres_service.delete_book_items([x for x in pg_ids or [] if x is not None])
                    raise
                keyword_index_needs_rebuild = True

                return {
                    "message": "Upload JSON (legacy) thành công",
                    "book_id": book_id,
                    "items_count": len(items_raw),
                    "postgres_count": len(pg_ids) if pg_ids else 0,
                    "vector_ids": vector_ids,
                }

        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .json")

    except HTTPException:
        logger.error("[UPLOAD] HTTPException đã được raise")
        raise
    except Exception as exc:
        logger.error(f"[UPLOAD] ❌ Lỗi khi xử lý file: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


