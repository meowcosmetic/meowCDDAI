# Hướng dẫn cấu hình Ollama

## 1. Cài đặt Ollama

### Windows/Mac/Linux:
Tải và cài đặt từ: https://ollama.ai

### Hoặc dùng Docker:
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## 2. Pull model GPT-OSS-20B

Model này là GGUF format từ Hugging Face. Có 2 cách để sử dụng:

### Cách 1: Sử dụng Ollama với Modelfile (Khuyến nghị)

Tạo file `Modelfile`:

```dockerfile
FROM hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M

TEMPLATE """{{ .System }}

{{ .Prompt }}"""
```

Sau đó tạo model trong Ollama:

```bash
ollama create gpt-oss-20b -f Modelfile
```

### Cách 2: Pull trực tiếp từ Hugging Face

Nếu Ollama hỗ trợ pull từ Hugging Face:

```bash
ollama pull hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

Hoặc đặt tên custom:

```bash
ollama pull gpt-oss-20b
```

### Cách 3: Download thủ công và import

1. Download model từ Hugging Face: https://hf.co/unsloth/gpt-oss-20b-GGUF
2. Tạo Modelfile trỏ đến file local
3. Import vào Ollama

### Kiểm tra model đã được tạo:

```bash
ollama list
```

Bạn sẽ thấy `gpt-oss-20b` trong danh sách.

## 3. Khởi động Ollama

Ollama sẽ tự động chạy sau khi cài đặt. Nếu không, chạy:

```bash
ollama serve
```

Kiểm tra Ollama đang chạy:
```bash
curl http://localhost:11434/api/tags
```

## 4. Cấu hình trong .env

Tạo file `.env` trong thư mục gốc của project:

```env
# Bật local LLM
USE_LOCAL_LLM=true

# Chọn loại LLM: "ollama" hoặc "openai-compatible"
LLM_TYPE=ollama

# Cấu hình Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=gpt-oss-20b

# Nếu dùng OpenAI-compatible API (vLLM, llama.cpp server, etc.)
# LOCAL_LLM_BASE_URL=http://localhost:8000/v1
# LOCAL_LLM_MODEL_NAME=gpt-20b
# LOCAL_LLM_API_KEY=not-needed
```

## 5. Cài đặt dependencies

```bash
pip install langchain-ollama
```

Hoặc cài tất cả:
```bash
pip install -r requirements.txt
```

## 6. Test kết nối

Sau khi cấu hình, test xem Ollama có hoạt động không:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gpt-oss-20b",
    base_url="http://localhost:11434"
)

response = llm.invoke("Hello!")
print(response.content)
```

## 7. Sử dụng với API

Sau khi cấu hình xong, API `/process-csv` sẽ tự động sử dụng Ollama model đã cấu hình.

## Troubleshooting

### Lỗi: "Connection refused"
- Đảm bảo Ollama đang chạy: `ollama serve`
- Kiểm tra port: `curl http://localhost:11434/api/tags`

### Lỗi: "Model not found"
- Pull model trước: `ollama pull <model_name>`
- Kiểm tra model đã pull: `ollama list`

### Lỗi: "langchain-ollama not found"
- Cài đặt: `pip install langchain-ollama`

### Thay đổi model
Chỉ cần thay đổi `OLLAMA_MODEL_NAME` trong `.env` và restart server.

