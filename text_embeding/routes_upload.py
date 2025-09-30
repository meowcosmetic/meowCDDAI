from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from models import BookUploadRequest
from .services import (
    embedding_service,
    qdrant_service,
    text_processor,
    keyword_index_needs_rebuild,
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
    """Upload sách từ file text"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file .txt")

    try:
        content = await file.read()
        content_str = content.decode("utf-8")

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file: {str(exc)}")


