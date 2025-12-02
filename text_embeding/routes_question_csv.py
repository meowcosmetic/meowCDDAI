from typing import List, Dict, Any, Optional
import json
import csv
import io
import logging
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import sys

# Thêm path để import config và langchain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from langchain_core.messages import HumanMessage
import requests

# Setup logger
logger = logging.getLogger(__name__)

# Import LLM setup - sử dụng lại logic từ routes_csv
def get_llm():
    """
    Khởi tạo LLM - sử dụng Ollama hoặc OpenAI-compatible API
    """
    if not Config.USE_LOCAL_LLM:
        raise ValueError("USE_LOCAL_LLM phải được bật (true) để sử dụng model local")
    
    if Config.LLM_TYPE == "ollama":
        try:
            from langchain_ollama import ChatOllama
            logger.info(f"[QuestionCSV] Khởi tạo Ollama model: {Config.OLLAMA_MODEL_NAME} tại {Config.OLLAMA_BASE_URL}")
            return ChatOllama(
                model=Config.OLLAMA_MODEL_NAME,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.7,
                timeout=120
            )
        except ImportError:
            raise ImportError("langchain-ollama chưa được cài đặt. Cài đặt bằng: pip install langchain-ollama")
    else:
        try:
            from langchain_openai import ChatOpenAI
            logger.info(f"[QuestionCSV] Khởi tạo OpenAI-compatible model: {Config.LOCAL_LLM_MODEL_NAME} tại {Config.LOCAL_LLM_BASE_URL}")
            return ChatOpenAI(
                model=Config.LOCAL_LLM_MODEL_NAME,
                base_url=Config.LOCAL_LLM_BASE_URL,
                api_key=Config.LOCAL_LLM_API_KEY,
                temperature=0.7,
                timeout=120
            )
        except ImportError:
            raise ImportError("langchain-openai chưa được cài đặt. Cài đặt bằng: pip install langchain-openai")

def call_llm(prompt: str) -> str:
    """
    Gọi LLM local với prompt và trả về response text
    """
    if llm is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model local chưa được khởi tạo. Kiểm tra cấu hình tại {Config.OLLAMA_BASE_URL if Config.LLM_TYPE == 'ollama' else Config.LOCAL_LLM_BASE_URL}"
        )
    
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[QuestionCSV] Lỗi khi gọi model local: {error_msg}")
        if "connection" in error_msg.lower() or "refused" in error_msg.lower() or "10061" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Không thể kết nối đến model local. "
                    f"Vui lòng đảm bảo model local đang chạy. "
                    f"Lỗi: {error_msg}"
                )
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi khi gọi model local: {error_msg}"
            )

router = APIRouter()

# Khởi tạo LLM (sau khi định nghĩa get_llm)
llm = None
try:
    llm = get_llm()
except Exception as e:
    logger.error(f"[QuestionCSV] Không thể khởi tạo LLM: {str(e)}")

# Đường dẫn file output
OUTPUT_FILE = "output.json"


class QuestionCSVProcessResponse(BaseModel):
    message: str = Field(..., description="Thông báo kết quả")
    total_rows: int = Field(..., description="Tổng số dòng đã xử lý")
    processed_rows: int = Field(..., description="Số dòng đã xử lý thành công")
    output_file: str = Field(..., description="Đường dẫn file output")
    errors: List[str] = Field(default=[], description="Danh sách lỗi nếu có")


class UploadQuestionsResponse(BaseModel):
    message: str = Field(..., description="Thông báo kết quả")
    total_items: int = Field(..., description="Tổng số items trong file")
    success_count: int = Field(..., description="Số items đã upload thành công")
    failed_count: int = Field(..., description="Số items upload thất bại")
    errors: List[str] = Field(default=[], description="Danh sách lỗi nếu có")


def load_existing_output() -> List[Dict[str, Any]]:
    """
    Load dữ liệu từ file output.json nếu đã tồn tại
    """
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return []
        except Exception as e:
            logger.warning(f"[QuestionCSV] Không thể đọc file output.json: {str(e)}, tạo file mới")
            return []
    return []


def append_to_output(row_data: Dict[str, Any]) -> None:
    """
    Append một dòng đã xử lý vào file output.json
    """
    try:
        # Load dữ liệu hiện có
        existing_data = load_existing_output()
        
        # Thêm dòng mới
        existing_data.append(row_data)
        
        # Ghi lại toàn bộ file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[QuestionCSV] Đã append dòng vào {OUTPUT_FILE}")
    except Exception as e:
        logger.error(f"[QuestionCSV] Lỗi khi append vào output.json: {str(e)}")
        raise


TRANSLATION_API_URL = "http://192.168.1.184/api/ai/transformers/task"


def translate_text(input_text: str, source_lang: str, target_lang: str) -> str:
    """
    Gọi API dịch: http://192.168.1.184/api/ai/transformers/task
    và trả về chuỗi đã dịch.
    """
    try:
        payload = {
            "task": "translation",
            "model": "facebook/nllb-200-1.3B",
            "input_text": input_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "max_length": 512,
        }
        resp = requests.post(TRANSLATION_API_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Tùy theo API, chỉnh lại key kết quả ở đây
        if isinstance(data, dict):
            if "output_text" in data:
                return data["output_text"]
            if "translation" in data:
                return data["translation"]
            # Trường hợp API trả về: {"result": [{"translation_text": "..."}], ...}
            if "result" in data:
                result = data["result"]
                if isinstance(result, str):
                    return result
                if isinstance(result, list) and result:
                    first = result[0]
                    if isinstance(first, dict):
                        if "translation_text" in first:
                            return first["translation_text"]
                        # fallback: lấy giá trị đầu tiên dạng string nếu có
                        for v in first.values():
                            if isinstance(v, str):
                                return v

        raise ValueError(f"Không tìm thấy trường kết quả trong response: {data}")
    except Exception as e:
        logger.error(f"[QuestionCSV] Lỗi khi gọi API dịch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi gọi API dịch: {str(e)}"
        )


def generate_display_category(category: str) -> Dict[str, str]:
    """
    Tạo display_category (JSON en/vi) từ category:
    - en: giữ nguyên category (giả sử là tiếng Anh)
    - vi: dịch từ en -> vi qua API NLLB (eng_Latn -> vie_Latn)
    """
    if not category:
        return {"en": "", "vi": ""}

    try:
        en_text = category
        vi_text = translate_text(category, source_lang="eng_Latn", target_lang="vie_Latn")
        return {
            "en": en_text,
            "vi": vi_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QuestionCSV] Lỗi khi tạo display_category cho category '{category}': {str(e)}")
        # Fallback: dùng nguyên category cho cả hai ngôn ngữ
        return {
            "en": category,
            "vi": category,
        }


def generate_question(raw_question: str) -> Dict[str, str]:
    """
    Tạo question (JSON en/vi) dựa trên raw_question:
    - vi: chính là raw_question (theo yêu cầu)
    - en: dịch từ vi -> en qua API NLLB (vie_Latn -> eng_Latn)
    """
    try:
        if not raw_question:
            return {"en": "", "vi": ""}

        # Dịch từ Việt -> Anh
        en_text = translate_text(raw_question, source_lang="vie_Latn", target_lang="eng_Latn")

        return {
            "en": en_text,
            "vi": raw_question,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QuestionCSV] Lỗi khi generate question cho raw_question '{raw_question[:50]}...': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi generate question: {str(e)}"
        )


@router.post("/process-question-csv", response_model=QuestionCSVProcessResponse)
async def process_question_csv(file: UploadFile = File(...)):
    """
    Xử lý file CSV câu hỏi với các cột: category, display_category, sub_category, display_sub_category, raw_question, question, summary, post_link
    
    Thực hiện:
    1. Generate display_category (JSON en/vi) dựa trên category
    2. Generate display_sub_category (JSON en/vi) dựa trên sub_category
    3. Generate question (JSON en/vi) dựa trên raw_question (vi = raw_question)
    4. Generate summary dựa trên raw_question, category, sub_category
    5. Append từng dòng đã xử lý vào output.json ngay sau khi xử lý xong
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File phải có định dạng CSV")
        
        logger.info(f"[QuestionCSV] Bắt đầu xử lý file: {file.filename}")
        
        # Đọc file CSV
        content = await file.read()
        content_str = content.decode('utf-8-sig')  # utf-8-sig để xử lý BOM
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(content_str))
        rows = list(csv_reader)
        
        if not rows:
            raise HTTPException(status_code=400, detail="File CSV không có dữ liệu")
        
        logger.info(f"[QuestionCSV] Đã đọc được {len(rows)} dòng từ CSV")
        
        # Kiểm tra các cột bắt buộc (chấp nhận cả catagory và category)
        required_columns_variants = {
            'category': ['category', 'catagory'],
            'display_category': ['display_category', 'display_catagory'],
            'sub_category': ['sub_category', 'sub_catagory'],
            'display_sub_category': ['display_sub_category', 'display_sub_catagory'],
            'raw_question': ['raw_question'],
            'question': ['question'],
            'summary': ['summary'],
            'post_link': ['post_link']
        }
        
        first_row_keys = set(rows[0].keys())
        missing_columns = []
        
        for col_name, variants in required_columns_variants.items():
            if not any(variant in first_row_keys for variant in variants):
                missing_columns.append(col_name)
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"File CSV thiếu các cột: {', '.join(missing_columns)}"
            )
        
        # Xử lý từng dòng
        processed_rows = 0
        errors = []
        
        logger.info("[QuestionCSV] Bắt đầu xử lý từng dòng...")
        
        for idx, row in enumerate(rows, start=1):
            try:
                # Lấy các giá trị từ CSV (chấp nhận cả catagory và category)
                category = row.get('category') or row.get('catagory', '')
                category = category.strip() if category else ''
                
                sub_category = row.get('sub_category') or row.get('sub_catagory', '')
                sub_category = sub_category.strip() if sub_category else ''
                
                raw_question = row.get('raw_question', '').strip()
                
                if not category or not raw_question:
                    errors.append(f"Dòng {idx}: Thiếu category hoặc raw_question")
                    continue
                
                # Generate display_category
                logger.info(f"[QuestionCSV] Dòng {idx}: Đang generate display_category cho '{category}'...")
                display_category = generate_display_category(category)
                
                # Generate display_sub_category (nếu có sub_category)
                display_sub_category = {}
                if sub_category:
                    logger.info(f"[QuestionCSV] Dòng {idx}: Đang generate display_sub_category cho '{sub_category}'...")
                    display_sub_category = generate_display_category(sub_category)
                
                # Generate question
                logger.info(f"[QuestionCSV] Dòng {idx}: Đang generate question cho raw_question '{raw_question[:50]}...'")
                question = generate_question(raw_question)
                
                # Giữ nguyên summary từ CSV (không generate)
                summary = row.get('summary', '').strip()
                
                # Tạo dữ liệu đã xử lý
                processed_row = {
                    "category": category,
                    "displayCategory": display_category,
                    "subCategory": sub_category,
                    "displaySubCategory": display_sub_category if display_sub_category else {},
                    "raw_question": raw_question,
                    "question": question,
                    "summary": summary,
                    "postLink": row.get('post_link', '').strip(),  # Giữ nguyên nếu có
                    "processed_at": datetime.now().isoformat()
                }
                
                # Append vào output.json ngay lập tức
                append_to_output(processed_row)
                processed_rows += 1
                
                logger.info(f"[QuestionCSV] ✅ Dòng {idx} đã được xử lý và lưu vào {OUTPUT_FILE}")
                
            except HTTPException:
                raise
            except Exception as e:
                error_msg = f"Dòng {idx}: Lỗi khi xử lý - {str(e)}"
                logger.error(f"[QuestionCSV] {error_msg}")
                errors.append(error_msg)
        
        logger.info(f"[QuestionCSV] Hoàn thành xử lý {processed_rows}/{len(rows)} dòng")
        
        return QuestionCSVProcessResponse(
            message=f"Đã xử lý thành công {processed_rows}/{len(rows)} dòng",
            total_rows=len(rows),
            processed_rows=processed_rows,
            output_file=OUTPUT_FILE,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[QuestionCSV] Lỗi khi xử lý file CSV: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý file CSV: {str(exc)}"
        )


# API endpoint để đẩy câu hỏi vào database
FAQ_QUESTIONS_API_URL = "http://192.168.1.184/api/cdd/api/v1/neon/faq-questions"


def transform_question_data(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform dữ liệu từ format output.json sang format API yêu cầu
    """
    # Lấy category và subCategory (có thể cần map sang enum)
    category = item.get("category", "").upper().replace(" ", "_")
    sub_category = item.get("sub_category", "").upper().replace(" ", "_")
    
    # Lấy displayCategory và displaySubCategory (chuyển từ object {en, vi} sang string vi)
    display_category = item.get("displayCategory", {})
    if isinstance(display_category, dict):
        display_category = display_category.get("vi", display_category.get("en", ""))
    
    display_sub_category = item.get("displaySubCategory", {})
    if isinstance(display_sub_category, dict):
        display_sub_category = display_sub_category.get("vi", display_sub_category.get("en", ""))
    
    # Lấy question (chuyển từ object {en, vi} sang string vi)
    question = item.get("question", {})
    if isinstance(question, dict):
        question = question.get("vi", question.get("en", ""))
    
    # Lấy các trường khác
    summary = item.get("summary", "")
    post_link = item.get("postLink", "")
    
    return {
        "category": category,
        "displayCategory": display_category,
        "subCategory": sub_category,
        "displaySubCategory": display_sub_category,
        "question": question,
        "summary": summary,
        "postLink": post_link
    }


@router.post("/upload-questions", response_model=UploadQuestionsResponse)
async def upload_questions(file: UploadFile = File(...)):
    """
    Đẩy câu hỏi vào database từ file JSON
    
    File JSON phải là array các object với format:
    [
      {
        "category": "...",
        "displayCategory": {"en": "...", "vi": "..."},
        "subCategory": "...",
        "displaySubCategory": {"en": "...", "vi": "..."},
        "question": {"en": "...", "vi": "..."},
        "summary": "...",
        "postLink": "..."
      }
    ]
    
    Mỗi item sẽ được transform và gửi tới API POST /api/v1/neon/faq-questions
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="File phải có định dạng JSON")
        
        logger.info(f"[UploadQuestions] Bắt đầu upload file: {file.filename}")
        
        # Đọc file JSON
        content = await file.read()
        content_str = content.decode('utf-8')
        
        try:
            data = json.loads(content_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"File JSON không hợp lệ: {str(e)}")
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="File JSON phải là một mảng các object")
        
        if not data:
            raise HTTPException(status_code=400, detail="File JSON không có dữ liệu")
        
        logger.info(f"[UploadQuestions] Đã đọc được {len(data)} items từ file")
        
        # Xử lý từng item
        success_count = 0
        failed_count = 0
        errors = []
        
        logger.info("[UploadQuestions] Bắt đầu upload từng item...")
        
        for idx, item in enumerate(data, start=1):
            try:
                # Transform dữ liệu
                transformed_data = transform_question_data(item)
                
                # Gửi tới API
                logger.info(f"[UploadQuestions] Đang upload item {idx}/{len(data)}: {transformed_data.get('question', '')[:50]}...")
                
                response = requests.post(
                    FAQ_QUESTIONS_API_URL,
                    json=transformed_data,
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    success_count += 1
                    logger.info(f"[UploadQuestions] ✅ Item {idx} đã được upload thành công")
                else:
                    failed_count += 1
                    error_msg = f"Item {idx}: API trả về status {response.status_code} - {response.text[:200]}"
                    logger.error(f"[UploadQuestions] {error_msg}")
                    errors.append(error_msg)
                    
            except Exception as e:
                failed_count += 1
                error_msg = f"Item {idx}: Lỗi khi upload - {str(e)}"
                logger.error(f"[UploadQuestions] {error_msg}")
                errors.append(error_msg)
        
        logger.info(f"[UploadQuestions] Hoàn thành: {success_count} thành công, {failed_count} thất bại")
        
        return UploadQuestionsResponse(
            message=f"Đã upload {success_count}/{len(data)} items thành công",
            total_items=len(data),
            success_count=success_count,
            failed_count=failed_count,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[UploadQuestions] Lỗi khi xử lý file: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý file: {str(exc)}"
        )

