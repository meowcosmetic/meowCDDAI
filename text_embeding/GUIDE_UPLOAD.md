# Hướng dẫn sử dụng Upload Sách (Meow CDD AI)

Tài liệu này hướng dẫn cách sử dụng API upload sách mới, tích hợp lưu trữ song song vào **PostgreSQL** và **Qdrant**.

## 1. Endpoint
*   **URL**: `/upload-book-file`
*   **Method**: `POST`
*   **Content-Type**: `multipart/form-data`

## 2. Các tham số (Form Data)
- `file`: File cần upload (hỗ trợ `.json` hoặc `.txt`).
- `book_id` (Tùy chọn): ID định danh cho sách (nếu không có sẽ tự lấy từ tên file).
- Các tham số khác (`title`, `author`, `year`, `tags`, `category`): Dùng cho format cũ hoặc file `.txt`.

## 3. Định dạng File JSON hỗ trợ

### A. Định dạng MeowBook mới (Khuyến nghị)
Hệ thống sẽ tự động nhận diện nếu dữ liệu là một mảng các object có các trường sau:

| Trường | Kiểu dữ liệu | Lưu tại | Ghi chú |
| :--- | :--- | :--- | :--- |
| `Book` | String | Postgres & Qdrant | Tên sách |
| `Chapter` | String | Postgres & Qdrant | Tên chương |
| `Page` | Float/Int | Postgres & Qdrant | Số trang |
| `Content` | String | Postgres | Nội dung thô (Optional) |
| `CleanedContent` | String | Postgres & Qdrant | Nội dung đã làm sạch (Dùng để tạo Vector) |
| `SearchQueries` | Array[String] | Postgres | Danh sách các câu hỏi tìm kiếm liên quan |

**Ví dụ JSON:**
```json
[
  {
    "Book": "Từng bước nhỏ một quyển 1",
    "Chapter": "Chương1: TỪNG BƯỚC NHỎ MỘT LÀ GÌ ?",
    "Page": 2.0,
    "Content": "Nội dung thô từ PDF...",
    "CleanedContent": "Nội dung đã được xử lý chuẩn hóa...",
    "SearchQueries": [
      "chương trình can thiệp sớm cho trẻ CPTTT",
      "định nghĩa CPTTT là gì?"
    ]
  }
]
```

### B. Định dạng Legacy (Cũ)
Sử dụng cho các dữ liệu chỉ có `summary` và `content`:
```json
[
  {
    "summary": "Tóm tắt đoạn văn",
    "content": "Nội dung chi tiết"
  }
]
```
*Lưu ý: Format này chỉ lưu vào Qdrant, không lưu vào Postgres.*

## 4. Luồng xử lý dữ liệu (MeowBook Format)
1.  **PostgreSQL**: Lưu toàn bộ thông tin metadata và nội dung thô + search queries vào bảng `book_contents`.
    - Database: `cdd_db`
    - Table: `book_contents`
    - Credentials: `cdd_app_admin` / `cdd_app_admin`
2.  **Qdrant**:
    - Chuyển `CleanedContent` thành Vector (1024 dims).
    - Lưu vào collection `books` với 2 named vectors: `summary` và `content`.
    - Payload lưu metadata cơ bản (Book, Chapter, Page) để phục vụ Hybrid Search.

## 5. Lưu ý kỹ thuật
- Hệ thống tự động khởi tạo bảng trong Postgres nếu chưa có.
- Sau khi upload thành công, index từ khóa (Keyword Index) sẽ được đánh dấu cần rebuild để cập nhật kết quả tìm kiếm Hybrid.
- Nếu gặp lỗi kết nối Postgres, vui lòng kiểm tra cấu hình trong `config.py`.
