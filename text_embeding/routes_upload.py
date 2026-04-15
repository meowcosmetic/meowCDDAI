from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import json
import uuid
import logging
from datetime import datetime
from qdrant_client.models import PointStruct

from models import BookUploadRequest, MeowBookItem
from .services import (
    embedding_service,
    qdrant_service,
    text_processor,
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


@router.post("/upload-book-file")
async def upload_book_file(
    file: UploadFile = File(...),
    book_id: str = Form(None, description="ID của sách"),
    title: str = Form(None, description="Tiêu đề sách"),
    author: str = Form(None, description="Tác giả sách"),
    year: str = Form(None, description="Năm xuất bản"),
    tags: str = Form(None, description="Tags, phân cách bằng dấu phẩy"),
    language: str = Form("vi", description="Ngôn ngữ"),
    category: str = Form("Lập trình", description="Thể loại"),
):
    """Upload sách từ file .json (array of {summary, content}) hoặc .txt (legacy)."""
    global keyword_index_needs_rebuild

    logger.info(f"[UPLOAD] Bắt đầu upload file: {file.filename}")
    start_time = datetime.now()

    try:
        logger.info(f"[UPLOAD] Đang đọc file: {file.filename}")
        content = await file.read()
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

                # Prepare for Qdrant
                cleaned_contents = [item.CleanedContent or "" for item in meow_items]
                
                # In Qdrant "old config", both summary and content vectors are required
                # We'll use CleanedContent for both or use a derived summary
                logger.info("[UPLOAD] Đang tạo embeddings cho Qdrant (named vectors: summary, content)...")
                embeddings = embedding_service.encode_text(cleaned_contents)
                
                points = []
                for i, item in enumerate(meow_items):
                    point_id = str(uuid.uuid4())
                    payload = {
                        "book_id": book_id or item.Book or "Unknown",
                        "book_name": item.Book or "Unknown",
                        "chapter": item.Chapter or "",
                        "page": item.Page or 0.0,
                        "content": item.CleanedContent or "",
                        "summary": f"{item.Book or 'Unknown'} - {item.Chapter or ''} (Trang {item.Page or 0.0})",
                        "postgres_id": pg_ids[i] if pg_ids and i < len(pg_ids) else None
                    }
                    point = PointStruct(
                        id=point_id,
                        vector={
                            "summary": embeddings[i], # Using same embedding for both as "old config" fallback
                            "content": embeddings[i]
                        },
                        payload=payload
                    )
                    points.append(point)
                
                logger.info(f"[UPLOAD] Đang upsert {len(points)} points vào Qdrant...")
                vector_ids = qdrant_service.upsert_named_points(points)
                
                keyword_index_needs_rebuild = True
                
                return {
                    "message": "Upload MeowBook thành công",
                    "postgres_count": len(meow_items),
                    "qdrant_count": len(vector_ids),
                    "book_name": meow_items[0].Book if meow_items else "Unknown"
                }

            else:
                # Old format: summary/content
                logger.info(f"[UPLOAD] Xử lý format summary/content cũ")
                if not book_id:
                    logger.error("[UPLOAD] Thiếu tham số book_id cho format cũ")
                    raise HTTPException(status_code=400, detail="Thiếu tham số book_id")
                
                summaries = []
                contents = []
                for idx, item in enumerate(items_raw):
                    s = item.get("summary", "")
                    c = item.get("content", "")
                    summaries.append(s)
                    contents.append(c)
                
                summary_vectors = embedding_service.encode_text(summaries)
                content_vectors = embedding_service.encode_text(contents)

                points = []
                for i in range(len(items_raw)):
                    point_id = str(uuid.uuid4())
                    payload = {
                        "book_id": book_id,
                        "summary": summaries[i],
                        "content": contents[i],
                    }
                    point = PointStruct(
                        id=point_id,
                        vector={
                            "summary": summary_vectors[i],
                            "content": content_vectors[i],
                        },
                        payload=payload,
                    )
                    points.append(point)

                vector_ids = qdrant_service.upsert_named_points(points)
                keyword_index_needs_rebuild = True
                
                return {
                    "message": "Upload JSON (legacy) thành công",
                    "book_id": book_id,
                    "items_count": len(items_raw),
                    "vector_ids": vector_ids
                }

        # Legacy: .txt upload with chunking
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .json hoặc .txt")

        tag_list = []
        if tags and tags.strip():
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        default_book_id = file.filename.replace(".txt", "")
        default_title = file.filename.replace(".txt", "")

        book_request_data = {
            "book_id": book_id if book_id else default_book_id,
            "title": title if title else default_title,
            "author": author if author else "Unknown",
            "year": int(year) if year and year.isdigit() else 2024,
            "content": content_str,
            "tags": tag_list,
            "language": language,
            "category": category,
        }
        
        # Inline logic from old upload_book for .txt files
        paragraphs = text_processor.split_into_paragraphs(content_str)
        if not paragraphs:
            raise HTTPException(status_code=400, detail="Không thể chia nội dung thành các đoạn có ý nghĩa")

        book_vectors = text_processor.create_book_vectors(book_request_data, paragraphs)
        contents = [bv.payload.content for bv in book_vectors]
        embeddings = embedding_service.encode_text(contents)

        for i, book_vector in enumerate(book_vectors):
            book_vector.vector = embeddings[i]

        vector_ids = qdrant_service.add_book_vectors(book_vectors)
        keyword_index_needs_rebuild = True

        return {
            "message": "Sách (.txt) đã được upload thành công",
            "book_id": book_request_data["book_id"],
            "paragraphs_count": len(paragraphs),
            "vectors_created": len(vector_ids),
            "vector_ids": vector_ids,
            "keyword_index_needs_rebuild": True,
        }
    except HTTPException:
        logger.error("[UPLOAD] HTTPException đã được raise")
        raise
    except Exception as exc:
        logger.error(f"[UPLOAD] ❌ Lỗi khi xử lý file: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file: {str(exc)}")


