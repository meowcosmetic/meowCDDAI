# Hướng dẫn gọi API xử lý CSV

## Endpoint
```
POST http://localhost:8102/process-csv
```

## Cách 1: Sử dụng cURL

```bash
curl -X POST "http://localhost:8102/process-csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/file.csv"
```

## Cách 2: Sử dụng Python requests

```python
import requests

url = "http://localhost:8102/process-csv"

# Mở file CSV
with open("your_file.csv", "rb") as f:
    files = {"file": ("your_file.csv", f, "text/csv")}
    response = requests.post(url, files=files)

# Kiểm tra kết quả
if response.status_code == 200:
    result = response.json()
    print(f"Đã xử lý {result['total_rows']} dòng")
    print(f"Dữ liệu đã xử lý: {result['processed_data']}")
    if result['errors']:
        print(f"Có {len(result['errors'])} lỗi: {result['errors']}")
else:
    print(f"Lỗi: {response.status_code}")
    print(response.text)
```

## Cách 3: Sử dụng JavaScript/Fetch

```javascript
const formData = new FormData();
const fileInput = document.querySelector('input[type="file"]');
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8102/process-csv', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Đã xử lý:', data.total_rows, 'dòng');
  console.log('Dữ liệu:', data.processed_data);
  if (data.errors.length > 0) {
    console.error('Lỗi:', data.errors);
  }
})
.catch(error => console.error('Error:', error));
```

## Cách 4: Sử dụng Postman

1. Mở Postman
2. Chọn method: **POST**
3. URL: `http://localhost:8102/process-csv`
4. Vào tab **Body** → chọn **form-data**
5. Key: chọn type là **File**, tên key là `file`
6. Value: chọn file CSV của bạn
7. Click **Send**

## Format file CSV yêu cầu

File CSV phải có các cột sau:
- `CTCT`
- `minAgeMonths`
- `maxAgeMonth`
- `domain`
- `item`
- `cirtialItem`

### Ví dụ file CSV:

```csv
CTCT,minAgeMonths,maxAgeMonth,domain,item,cirtialItem
ABC,6,12,Phát triển ngôn ngữ,Trẻ có thể phát âm từ đơn giản,Trẻ phát âm được ít nhất 5 từ đơn giản
DEF,12,18,Phát triển vận động,Trẻ có thể ngồi vững,Trẻ ngồi vững không cần hỗ trợ trong 30 giây
```

## Response format

```json
{
  "message": "Đã xử lý thành công 2 dòng",
  "total_rows": 2,
  "processed_data": [
    {
      "CTCT": "ABC",
      "minAgeMonths": "6",
      "maxAgeMonth": "12",
      "domain": "Phát triển ngôn ngữ",
      "domainId": "123",
      "item": "Trẻ có thể phát âm từ đơn giản",
      "cirtialItem": "Trẻ phát âm được ít nhất 5 từ đơn giản",
      "title": {
        "en": "Child can pronounce simple words",
        "vi": "Trẻ có thể phát âm từ đơn giản"
      },
      "description": {
        "en": "This intervention goal aims to...",
        "vi": "Mục tiêu can thiệp này nhằm..."
      },
      "level": 1
    }
  ],
  "errors": []
}
```

## Xem API Documentation

Sau khi chạy server, bạn có thể xem tài liệu API tại:
- Swagger UI: http://localhost:8102/docs
- ReDoc: http://localhost:8102/redoc


