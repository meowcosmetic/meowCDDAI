# Hệ Thống 4 AI Agents Xử Lý Mục Tiêu Can Thiệp

## Tổng Quan

Hệ thống này sử dụng 4 AI agents chuyên biệt để xử lý mục tiêu can thiệp cho trẻ đặc biệt, sử dụng Google AI API và LangChain framework.

## Kiến Trúc Hệ Thống

### 4 AI Agents

1. **ExpertAgent** 🔬
   - **Chức năng**: Phân tích chủ đề và tạo khung lý thuyết
   - **Input**: Mục tiêu can thiệp
   - **Output**: Phân tích chuyên sâu, khung lý thuyết khoa học

2. **EditorAgent** ✏️
   - **Chức năng**: Biên tập và diễn đạt dễ hiểu
   - **Input**: Kết quả từ ExpertAgent
   - **Output**: Nội dung được biên tập, dễ hiểu hơn

3. **PracticalAgent** 🛠️
   - **Chức năng**: Thêm ví dụ và checklist
   - **Input**: Nội dung đã biên tập + mục tiêu gốc
   - **Output**: Nội dung thực tiễn với ví dụ và checklist

4. **VerifierAgent** ✅
   - **Chức năng**: Kiểm chứng và thêm nguồn tham khảo
   - **Input**: Nội dung thực tiễn
   - **Output**: Nội dung đã kiểm chứng với nguồn tham khảo

## Workflow

```
Mục tiêu can thiệp
       ↓
   ExpertAgent (Phân tích lý thuyết)
       ↓
   EditorAgent (Biên tập dễ hiểu)
       ↓
   PracticalAgent (Thêm ví dụ & checklist)
       ↓
   VerifierAgent (Kiểm chứng & nguồn)
       ↓
   Kết quả hoàn chỉnh
```

## Cài Đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình API Key

API key Google AI đã được cấu hình trong `config.py`:
```python
GOOGLE_AI_API_KEY = "AIzaSyB0FiJmN7021PCM4B2EASfAtY_wXh_muVk"
```

### 3. Chạy server

```bash
python main.py
```

Server sẽ chạy trên `http://localhost:8102`

## Sử Dụng

### 1. API Endpoint

**POST** `/process-intervention-goal`

**Request Body:**
```json
{
    "intervention_goal": "Trẻ quay đầu và nhìn về phía âm thanh khi phát gần trước mặt (0.5–1m). Thực hiện ≥70% số lần trong 5–6 cơ hội mỗi buổi.",
    "title": "Can thiệp phản ứng âm thanh cho trẻ đặc biệt"
}
```

**Response:**
```json
{
    "message": "Xử lý mục tiêu can thiệp thành công",
    "post_id": "intervention_12345678",
    "title": "Can thiệp phản ứng âm thanh cho trẻ đặc biệt",
    "original_goal": "...",
    "processing_results": {
        "expert_analysis": "...",
        "edited_content": "...",
        "practical_content": "...",
        "verified_content": "..."
    },
    "workflow_summary": {
        "step_1": "ExpertAgent đã phân tích và tạo khung lý thuyết",
        "step_2": "EditorAgent đã biên tập và diễn đạt dễ hiểu",
        "step_3": "PracticalAgent đã thêm ví dụ và checklist",
        "step_4": "VerifierAgent đã kiểm chứng và thêm nguồn tham khảo"
    }
}
```

### 2. Sử dụng trực tiếp trong code

```python
from ai_agents import InterventionProcessor

# Khởi tạo processor
processor = InterventionProcessor()

# Mục tiêu can thiệp
intervention_goal = "Trẻ quay đầu và nhìn về phía âm thanh khi phát gần trước mặt (0.5–1m). Thực hiện ≥70% số lần trong 5–6 cơ hội mỗi buổi."

# Xử lý
result = processor.process_intervention_goal(intervention_goal)

if result["status"] == "success":
    print("Expert Analysis:", result["expert_analysis"])
    print("Edited Content:", result["edited_content"])
    print("Practical Content:", result["practical_content"])
    print("Verified Content:", result["verified_content"])
```

## Test Hệ Thống

Chạy file test để kiểm tra hệ thống:

```bash
python test_intervention_system.py
```

File test sẽ:
1. Test trực tiếp qua `InterventionProcessor`
2. Test qua API endpoint (cần server đang chạy)

## Ví Dụ Mục Tiêu Can Thiệp

Hệ thống được thiết kế để xử lý các mục tiêu can thiệp như:

- "Trẻ quay đầu và nhìn về phía âm thanh khi phát gần trước mặt (0.5–1m). Thực hiện ≥70% số lần trong 5–6 cơ hội mỗi buổi."
- "Trẻ có thể ngồi độc lập trong 30 giây mà không cần hỗ trợ."
- "Trẻ phản ứng với tên của mình bằng cách quay đầu hoặc nhìn về phía người gọi."

## Cấu Trúc File

```
├── ai_agents.py              # 4 AI agents và InterventionProcessor
├── config.py                 # Cấu hình API keys
├── text_embeding/
│   └── routes_post.py        # API endpoints
├── test_intervention_system.py # File test
└── requirements.txt          # Dependencies
```

## Lưu Ý

1. **API Key**: Đảm bảo API key Google AI hợp lệ và có đủ quota
2. **Network**: Cần kết nối internet để gọi Google AI API
3. **Language**: Tất cả prompts được thiết kế cho tiếng Việt
4. **Error Handling**: Hệ thống có xử lý lỗi cơ bản, cần mở rộng cho production

## Mở Rộng

Có thể mở rộng hệ thống bằng cách:
1. Thêm agents mới cho các chức năng khác
2. Tích hợp với database để lưu trữ kết quả
3. Thêm authentication và authorization
4. Cải thiện error handling và logging
5. Thêm caching để tối ưu performance
