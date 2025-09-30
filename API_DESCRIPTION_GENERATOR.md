# API Generate Description - Hướng dẫn sử dụng

## Tổng quan
API này cho phép generate mô tả song ngữ (Tiếng Việt - Tiếng Anh) từ một mục tiêu can thiệp nhỏ cho trẻ đặc biệt.

## Endpoint
```
POST /generate-description
```

## Request Format
```json
{
  "intervention_goal": "Mục tiêu can thiệp nhỏ"
}
```

## Response Format
```json
{
  "description": {
    "vi": "Mô tả chi tiết bằng tiếng Việt",
    "en": "Detailed description in English"
  },
  "original_goal": "Mục tiêu can thiệp gốc"
}
```

## Ví dụ sử dụng

### Request
```json
{
  "intervention_goal": "Trẻ có thể ngồi vững mà không cần hỗ trợ trong ít nhất 30 giây"
}
```

### Response
```json
{
  "description": {
    "vi": "Trẻ có thể ngồi vững mà không cần hỗ trợ trong ít nhất 30 giây",
    "en": "Child can sit steadily without support for at least 30 seconds"
  },
  "original_goal": "Trẻ có thể ngồi vững mà không cần hỗ trợ trong ít nhất 30 giây"
}
```

## Cách test

### 1. Khởi động server
```bash
python main.py
```

### 2. Test bằng curl
```bash
curl -X POST "http://localhost:8102/generate-description" \
     -H "Content-Type: application/json" \
     -d '{"intervention_goal": "Trẻ có thể nhận biết 5 màu cơ bản"}'
```

### 3. Test bằng Python script
```bash
python test_description_api.py
```

### 4. Xem API documentation
Mở trình duyệt và truy cập: `http://localhost:8102/docs`

## Các ví dụ mục tiêu can thiệp

1. **Kỹ năng vận động thô:**
   - "Trẻ có thể ngồi vững trong 30 giây"
   - "Trẻ có thể đi bộ 10 bước mà không ngã"

2. **Kỹ năng vận động tinh:**
   - "Trẻ có thể cầm bút chì và vẽ nét thẳng"
   - "Trẻ có thể cài khuy áo"

3. **Kỹ năng nhận thức:**
   - "Trẻ có thể nhận biết 5 màu cơ bản"
   - "Trẻ có thể đếm từ 1 đến 10"

4. **Kỹ năng giao tiếp:**
   - "Trẻ có thể giao tiếp bằng mắt khi được gọi tên"
   - "Trẻ có thể nói 10 từ đơn giản"

5. **Kỹ năng tự chăm sóc:**
   - "Trẻ có thể tự mặc áo sơ mi"
   - "Trẻ có thể tự rửa tay"

## Lưu ý
- API sử dụng Google Gemini AI để generate mô tả
- Cần có API key Google AI được cấu hình trong `config.py`
- Response sẽ được format theo chuẩn JSON với 2 ngôn ngữ
- Nếu AI không thể parse được response, sẽ trả về fallback description
