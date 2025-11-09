from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import json
import uuid
import logging
from datetime import datetime
from qdrant_client.models import PointStruct

from models import BookUploadRequest
from .services import (
    embedding_service,
    qdrant_service,
    text_processor,
    keyword_index_needs_rebuild,
)

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


router = APIRouter()


@router.post("/upload-book", response_model=dict)
async def upload_book(book_request: BookUploadRequest):
    """Upload một cuốn sách và chia thành các đoạn để lưu vào vector database"""
    global keyword_index_needs_rebuild
    try:
        paragraphs = text_processor.split_into_paragraphs(book_request.content)
        if not paragraphs:
            raise HTTPException(status_code=400, detail="Không thể chia nội dung thành các đoạn có ý nghĩa")

        book_data = book_request.dict()
        book_vectors = text_processor.create_book_vectors(book_data, paragraphs)

        contents = [bv.payload.content for bv in book_vectors]
        embeddings = embedding_service.encode_text(contents)

        for i, book_vector in enumerate(book_vectors):
            book_vector.vector = embeddings[i]

        vector_ids = qdrant_service.add_book_vectors(book_vectors)

        keyword_index_needs_rebuild = True

        return {
            "message": "Sách đã được upload thành công",
            "book_id": book_request.book_id,
            "paragraphs_count": len(paragraphs),
            "vectors_created": len(vector_ids),
            "vector_ids": vector_ids,
            "keyword_index_needs_rebuild": True,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi upload sách: {str(exc)}")


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
            
            if not book_id:
                logger.error("[UPLOAD] Thiếu tham số book_id")
                raise HTTPException(status_code=400, detail="Thiếu tham số book_id")

            try:
                logger.info("[UPLOAD] Đang parse JSON...")
                items = json.loads(content_str)
                logger.info(f"[UPLOAD] Parse thành công: {len(items)} items")
            except Exception as e:
                logger.error(f"[UPLOAD] Lỗi parse JSON: {str(e)}")
                raise HTTPException(status_code=400, detail="File JSON không hợp lệ")

            if not isinstance(items, list) or not items:
                logger.error(f"[UPLOAD] JSON không phải array hoặc rỗng")
                raise HTTPException(status_code=400, detail="JSON phải là một mảng các object")

            logger.info(f"[UPLOAD] Bắt đầu validate {len(items)} items...")
            summaries = []
            contents = []
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    logger.error(f"[UPLOAD] Item {idx} không phải object")
                    raise HTTPException(status_code=400, detail=f"Phần tử tại index {idx} không phải object")
                s = item.get("summary", "")
                c = item.get("content", "")
                if not isinstance(s, str) or not isinstance(c, str):
                    logger.error(f"[UPLOAD] Item {idx}: summary/content không phải string")
                    raise HTTPException(status_code=400, detail=f"summary/content tại index {idx} phải là string")
                summaries.append(s)
                contents.append(c)
            
            logger.info(f"[UPLOAD] Validation thành công:")
            logger.info(f"  - Tổng số items: {len(items)}")
            logger.info(f"  - Summary lengths: {[len(s) for s in summaries]}")
            logger.info(f"  - Content lengths: {[len(c) for c in contents]}")

            # Embeddings for both fields
            logger.info("[UPLOAD] Bắt đầu tạo embeddings cho summary...")
            embed_start = datetime.now()
            summary_vectors = embedding_service.encode_text(summaries)
            embed_time = (datetime.now() - embed_start).total_seconds()
            logger.info(f"[UPLOAD] Đã tạo {len(summary_vectors)} summary embeddings ({embed_time:.2f}s)")

            logger.info("[UPLOAD] Bắt đầu tạo embeddings cho content...")
            embed_start = datetime.now()
            content_vectors = embedding_service.encode_text(contents)
            embed_time = (datetime.now() - embed_start).total_seconds()
            logger.info(f"[UPLOAD] Đã tạo {len(content_vectors)} content embeddings ({embed_time:.2f}s)")

            # Upsert points with named vectors
            logger.info("[UPLOAD] Đang tạo points với named vectors...")
            points = []
            for i in range(len(items)):
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
                logger.debug(f"[UPLOAD] Tạo point {i+1}/{len(items)}: id={point_id}")

            logger.info(f"[UPLOAD] Đã tạo {len(points)} points, bắt đầu upsert vào Qdrant...")
            upsert_start = datetime.now()
            vector_ids = qdrant_service.upsert_named_points(points)
            upsert_time = (datetime.now() - upsert_start).total_seconds()
            logger.info(f"[UPLOAD] Upsert thành công {len(vector_ids)} points vào Qdrant ({upsert_time:.2f}s)")

            keyword_index_needs_rebuild = True
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"[UPLOAD] ✅ Upload JSON thành công!")
            logger.info(f"  - Book ID: {book_id}")
            logger.info(f"  - Items: {len(items)}")
            logger.info(f"  - Points created: {len(vector_ids)}")
            logger.info(f"  - Total time: {total_time:.2f}s")

            return {
                "message": "Upload JSON thành công",
                "book_id": book_id,
                "items_count": len(items),
                "vector_ids": vector_ids,
                "keyword_index_needs_rebuild": True,
            }

        # Legacy: .txt upload with chunking
        if not file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .json hoặc .txt")

        tag_list = []
        if tags and tags.strip():
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        default_book_id = file.filename.replace(".txt", "")
        default_title = file.filename.replace(".txt", "")

        book_request = BookUploadRequest(
            book_id=book_id if book_id else default_book_id,
            title=title if title else default_title,
            author=author if author else "Unknown",
            year=year if year else 2024,
            content=content_str,
            tags=tag_list,
            language=language,
            category=category,
        )
        return await upload_book(book_request)
    except HTTPException:
        logger.error("[UPLOAD] HTTPException đã được raise")
        raise
    except Exception as exc:
        logger.error(f"[UPLOAD] ❌ Lỗi khi xử lý file: {str(exc)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file: {str(exc)}")


