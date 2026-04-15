# Hướng dẫn sử dụng API Tìm kiếm (Search)

Hệ thống cung cấp nhiều phương thức tìm kiếm linh hoạt dựa trên Vector (Embedding), Từ khóa (Keyword), và kết hợp cả hai (Hybrid Search).

## 1. Tìm kiếm theo Vector (Embedding Search)
Tìm kiếm dựa trên ý nghĩa của câu thay vì chỉ khớp từ ngữ chính xác.

- **Endpoint**: `/search`
- **Method**: `POST`
- **Body**:
```json
{
  "query": "cách giúp trẻ tự lập",
  "limit": 10,
  "score_threshold": 0.7,
  "include_context": true
}
```
- **Tham số mới**: `include_context: true` sẽ trả về thêm đoạn văn bản trước (`prev_content`) và sau (`next_content`) của đoạn kêt quả tìm thấy.
- **Ghi chú**: Sử dụng model `intfloat/multilingual-e5-large` để chuyển câu hỏi thành vector và so sánh độ tương đồng trong Qdrant.

## 2. Tìm kiếm theo Từ khóa (Keyword Search)
Tìm kiếm chính xác các từ xuất hiện trong nội dung (Sử dụng thuật toán BM25).

- **Endpoint**: `/search-keywords`
- **Method**: `POST`
- **Body**: Tương tự như `/search`.
- **Ghi chú**: Phù hợp khi bạn nhớ chính xác từ chuyên môn hoặc tên riêng.

## 3. Tìm kiếm Hybrid (Khuyến nghị)
Kết hợp cả ý nghĩa (Vector) và từ khóa chính xác để đưa ra kết quả tốt nhất.

- **Endpoint**: `/search-hybrid`
- **Method**: `POST`
- **Body**:
```json
{
  "query": "chương trình can thiệp sớm cho trẻ CPTTT",
  "limit": 10,
  "alpha": 0.7,
  "beta": 0.3,
  "score_threshold": 0.5
}
```
- **Tham số quan trọng**:
    - `alpha` (0.0 -> 1.0): Trọng số cho kết quả Vector (ý nghĩa).
    - `beta` (0.0 -> 1.0): Trọng số cho kết quả Từ khóa (Keyword).
    - Tổng `alpha + beta` nên bằng 1.0.

## 4. Các API hỗ trợ khác

### Tìm kiếm theo ID Sách
- **Endpoint**: `/search-by-book-id/{book_id}`
- **Method**: `GET`
- **Ví dụ**: `/search-by-book-id/1?limit=20`

### Tìm kiếm theo Tags
- **Endpoint**: `/search-by-tags`
- **Method**: `GET`
- **Tham số**:
    - `tags`: Danh sách tag, cách nhau bởi dấu phẩy (vd: `autism,ADHD`).
    - `match_all`: `true` (phải khớp tất cả tags), `false` (chỉ cần khớp 1 tag).

## 5. Metadata trong kết quả trả về
Mỗi kết quả tìm kiếm sẽ trả về một `payload` chứa:
- `book_id`: ID định danh sách.
- `book_name`: Tên sách.
- `chapter`: Tên chương.
- `page`: Số trang.
- `content`: Nội dung đoạn văn (CleanedContent).
- `summary`: Tóm tắt ngắn gọn đoạn văn.
- `prev_content`: Nội dung của đoạn văn ngay trước đoạn này (nếu có và nếu `include_context=true`).
- `next_content`: Nội dung của đoạn văn ngay sau đoạn này (nếu có và nếu `include_context=true`).
- `postgres_id`: ID của đoạn văn trong PostgreSQL.

---
**Lưu ý**: Dữ liệu tìm kiếm được truy xuất từ **Qdrant**. Các thông tin bổ sung và `SearchQueries` chi tiết chỉ có trong **PostgreSQL**.
