# Book Vector Service với Hybrid Search

Service FastAPI để lưu trữ và tìm kiếm vector của sách sử dụng Qdrant với Hybrid Search kết hợp keyword và embedding.

## Tính năng

- Upload sách và tự động chia thành các đoạn có ý nghĩa
- Sinh vector embedding sử dụng model `intfloat/multilingual-e5-large`
- Lưu trữ vector trong Qdrant với cấu trúc dữ liệu chi tiết
- **Hybrid Search**: Kết hợp BM25 keyword search + embedding search
- Tìm kiếm semantic search dựa trên nội dung
- Hỗ trợ upload file text và JSON
- API đầy đủ với documentation tự động

## Hybrid Search

Service hỗ trợ 3 loại tìm kiếm:

1. **Embedding Search**: Tìm kiếm dựa trên semantic similarity
2. **Keyword Search**: Tìm kiếm dựa trên từ khóa (BM25)
3. **Hybrid Search**: Kết hợp cả hai với công thức:
   ```
   hybrid_score = α * embedding_score + β * keyword_score
   ```

### Cấu trúc dữ liệu

Mỗi vector được lưu với cấu trúc:

```json
{
  "id": "uuid",
  "vector": [0.123, 0.532, ...],
  "payload": {
    "book_id": "12345",
    "title": "Lập trình Python cơ bản",
    "author": "Nguyễn Văn A",
    "year": 2021,
    "chapter": 3,
    "chapter_title": "Cấu trúc dữ liệu",
    "page": 56,
    "paragraph_index": 2,
    "content": "Danh sách trong Python là một cấu trúc dữ liệu cơ bản...",
    "tags": ["Python", "lập trình", "cấu trúc dữ liệu"],
    "language": "vi",
    "category": "Lập trình"
  }
}
```

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd meowCDDAI
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Cấu hình Qdrant:
   - Tạo file `.env` với nội dung:
   ```
   QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.AVB9V8TSsz1z1mcHYSErTH3vS_tfJdtu_4ixXTHsV1w
   QDRANT_URL=https://your-qdrant-instance.qdrant.io
   COLLECTION_NAME=books
   ```

4. Chạy service:
```bash
python main.py
```

Service sẽ chạy tại `http://localhost:8000`

## API Endpoints

### 1. Health Check
```
GET /health
```
Kiểm tra trạng thái service và collection.

### 2. Upload Book (JSON)
```
POST /upload-book
```
Upload sách bằng JSON với cấu trúc:
```json
{
  "book_id": "12345",
  "title": "Lập trình Python cơ bản",
  "author": "Nguyễn Văn A",
  "year": 2021,
  "content": "Nội dung sách...",
  "tags": ["Python", "lập trình"],
  "language": "vi",
  "category": "Lập trình"
}
```

### 3. Upload Book File
```
POST /upload-book-file
```
Upload file text (.txt) với metadata.

### 4. Embedding Search
```
POST /search
```
Tìm kiếm dựa trên semantic similarity:
```json
{
  "query": "cấu trúc dữ liệu Python",
  "limit": 10,
  "score_threshold": 0.7
}
```

### 5. Keyword Search
```
POST /search-keywords
```
Tìm kiếm dựa trên từ khóa (BM25):
```json
{
  "query": "Python lập trình",
  "limit": 10,
  "score_threshold": 0.5
}
```

### 6. Hybrid Search
```
POST /search-hybrid
```
Tìm kiếm kết hợp keyword và embedding:
```json
{
  "query": "cấu trúc dữ liệu Python",
  "limit": 10,
  "score_threshold": 0.5,
  "alpha": 0.7,
  "beta": 0.3,
  "keyword_fields": ["content", "title", "tags"]
}
```

### 7. Delete Book
```
DELETE /delete-book/{book_id}
```
Xóa tất cả vector của một cuốn sách.

### 8. Collection Info
```
GET /collection-info
```
Lấy thông tin về collection.

### 9. Rebuild Keyword Index
```
POST /rebuild-keyword-index
```
Rebuild keyword index từ tất cả vectors trong collection.

## Sử dụng

### Upload sách
```python
import requests

# Upload bằng JSON
book_data = {
    "book_id": "python_basic",
    "title": "Lập trình Python cơ bản",
    "author": "Nguyễn Văn A",
    "year": 2021,
    "content": "Python là ngôn ngữ lập trình...",
    "tags": ["Python", "lập trình"],
    "language": "vi",
    "category": "Lập trình"
}

response = requests.post("http://localhost:8000/upload-book", json=book_data)
print(response.json())
```

### Tìm kiếm Hybrid
```python
# Hybrid search
hybrid_data = {
    "query": "cấu trúc dữ liệu",
    "limit": 5,
    "score_threshold": 0.3,
    "alpha": 0.7,  # Weight cho embedding
    "beta": 0.3,   # Weight cho keyword
    "keyword_fields": ["content", "title", "tags"]
}

response = requests.post("http://localhost:8000/search-hybrid", json=hybrid_data)
results = response.json()

for result in results:
    print(f"Hybrid Score: {result['hybrid_score']:.3f}")
    print(f"Embedding Score: {result['embedding_score']:.3f}")
    print(f"Keyword Score: {result['keyword_score']:.3f}")
    print(f"Title: {result['payload']['title']}")
    print(f"Content: {result['payload']['content'][:100]}...")
    print("---")
```

### Tìm kiếm riêng lẻ
```python
# Embedding search
embedding_data = {
    "query": "cấu trúc dữ liệu",
    "limit": 5,
    "score_threshold": 0.7
}

response = requests.post("http://localhost:8000/search", json=embedding_data)

# Keyword search
keyword_data = {
    "query": "Python lập trình",
    "limit": 5,
    "score_threshold": 0.5
}

response = requests.post("http://localhost:8000/search-keywords", json=keyword_data)
```

## Documentation

Truy cập `http://localhost:8000/docs` để xem Swagger UI documentation đầy đủ.

## Cấu hình

### Text Processing
- `min_paragraph_length`: 50 ký tự (đoạn tối thiểu)
- `max_paragraph_length`: 1000 ký tự (đoạn tối đa)

### Vector Model
- Model: `intfloat/multilingual-e5-large`
- Vector size: 1024 dimensions
- Distance metric: Cosine similarity

### Keyword Search
- Algorithm: BM25
- Tokenization: Simple word splitting cho tiếng Việt
- Fields: content, title, tags

### Hybrid Search
- Formula: `hybrid_score = α * embedding_score + β * keyword_score`
- Default weights: α = 0.7, β = 0.3
- Score normalization: [0, 1] range

### Qdrant
- Collection name: `books`
- Vector size: 1024
- Distance: Cosine

## Lưu ý

1. Lần đầu chạy sẽ tải model `multilingual-e5-large` (~2GB)
2. Cần có kết nối internet để tải model
3. Đảm bảo Qdrant instance có đủ dung lượng lưu trữ
4. API key Qdrant đã được cấu hình sẵn trong code
5. Keyword index được tự động rebuild khi upload sách mới

## Troubleshooting

### Lỗi kết nối Qdrant
- Kiểm tra URL và API key trong `config.py`
- Đảm bảo Qdrant instance đang hoạt động

### Lỗi tải model
- Kiểm tra kết nối internet
- Đảm bảo có đủ dung lượng ổ cứng (~2GB)

### Lỗi memory
- Giảm `max_paragraph_length` trong `text_processor.py`
- Xử lý sách theo batch nhỏ hơn

### Lỗi keyword search
- Chạy `/rebuild-keyword-index` để rebuild index
- Kiểm tra dữ liệu trong collection
