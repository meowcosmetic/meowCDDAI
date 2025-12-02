from typing import List, Dict, Any, Optional
import json
import csv
import io
import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import sys
import os
import requests

# Thêm path để import config và langchain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from langchain_core.messages import HumanMessage

# Setup logger
logger = logging.getLogger(__name__)

# Cấu hình LLM - Sử dụng Ollama hoặc OpenAI-compatible API
def get_llm():
    """
    Khởi tạo LLM - sử dụng Ollama hoặc OpenAI-compatible API
    """
    if not Config.USE_LOCAL_LLM:
        raise ValueError("USE_LOCAL_LLM phải được bật (true) để sử dụng model local")
    
    if Config.LLM_TYPE == "ollama":
        # Sử dụng Ollama
        try:
            from langchain_ollama import ChatOllama
            logger.info(f"[CSV] Khởi tạo Ollama model: {Config.OLLAMA_MODEL_NAME} tại {Config.OLLAMA_BASE_URL}")
            
            llm_instance = ChatOllama(
                model=Config.OLLAMA_MODEL_NAME,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.7,
                timeout=120
            )
            
            # Test connection với Ollama
            try:
                test_response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags", timeout=5)
                if test_response.status_code == 200:
                    logger.info(f"[CSV] ✅ Ollama đang chạy tại {Config.OLLAMA_BASE_URL}")
                    # Kiểm tra xem model có tồn tại không
                    models = test_response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    if Config.OLLAMA_MODEL_NAME not in model_names:
                        logger.warning(f"[CSV] ⚠️ Model '{Config.OLLAMA_MODEL_NAME}' chưa được pull trong Ollama")
                        logger.warning(f"[CSV] ⚠️ Chạy: ollama pull {Config.OLLAMA_MODEL_NAME}")
                else:
                    logger.warning(f"[CSV] ⚠️ Không thể kết nối đến Ollama tại {Config.OLLAMA_BASE_URL}")
            except Exception as e:
                logger.warning(f"[CSV] ⚠️ Không thể kiểm tra Ollama: {str(e)}")
                logger.warning(f"[CSV] ⚠️ Đảm bảo Ollama đang chạy: ollama serve")
            
            return llm_instance
        except ImportError:
            raise ImportError(
                "langchain-ollama chưa được cài đặt. "
                "Cài đặt bằng: pip install langchain-ollama"
            )
    else:
        # Sử dụng OpenAI-compatible API (vLLM, llama.cpp server, etc.)
        try:
            from langchain_openai import ChatOpenAI
            logger.info(f"[CSV] Khởi tạo OpenAI-compatible model: {Config.LOCAL_LLM_MODEL_NAME} tại {Config.LOCAL_LLM_BASE_URL}")
            
            llm_instance = ChatOpenAI(
                model=Config.LOCAL_LLM_MODEL_NAME,
                base_url=Config.LOCAL_LLM_BASE_URL,
                api_key=Config.LOCAL_LLM_API_KEY,
                temperature=0.7,
                timeout=120
            )
            
            # Test connection
            try:
                test_response = requests.get(f"{Config.LOCAL_LLM_BASE_URL.replace('/v1', '')}/health", timeout=5)
                logger.info(f"[CSV] ✅ Model local đang chạy tại {Config.LOCAL_LLM_BASE_URL}")
            except:
                logger.warning(f"[CSV] ⚠️ Không thể kiểm tra health của model local tại {Config.LOCAL_LLM_BASE_URL}")
                logger.warning(f"[CSV] ⚠️ Đảm bảo model local đang chạy trước khi sử dụng")
            
            return llm_instance
        except ImportError:
            raise ImportError(
                "langchain-openai chưa được cài đặt. "
                "Cài đặt bằng: pip install langchain-openai"
            )

llm = get_llm()

router = APIRouter()

# API endpoint để lấy domain information
DOMAIN_API_URL = "http://192.168.1.184/api/cdd/api/v1/neon/developmental-domains"


class CSVProcessResponse(BaseModel):
    message: str = Field(..., description="Thông báo kết quả")
    total_rows: int = Field(..., description="Tổng số dòng đã xử lý")
    processed_data: List[Dict[str, Any]] = Field(..., description="Dữ liệu đã xử lý")
    errors: List[str] = Field(default=[], description="Danh sách lỗi nếu có")


def fetch_domains() -> Dict[str, str]:
    """
    Lấy danh sách domains từ API và tạo mapping từ displayed_name.vi sang id
    Returns: Dict với key là displayed_name.vi và value là id
    """
    domain_mapping = {}
    page = 0
    size = 10
    max_pages = 1  # Giới hạn tối đa để tránh lặp vô tận
    
    try:
        while page < max_pages:
            url = f"{DOMAIN_API_URL}?page={page}&size={size}"
            logger.info(f"[CSV] Đang lấy domains từ API: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Giả sử API trả về format: {"content": [...], "totalElements": ..., "totalPages": ...}
            # Hoặc có thể là array trực tiếp
            if isinstance(data, list):
                # Nếu API trả về list trực tiếp, chỉ lấy một lần rồi break
                domains = data
                if not domains:
                    break
                
                # Tạo mapping từ displayed_name.vi sang id
                for domain in domains:
                    if isinstance(domain, dict):
                        displayed_name = domain.get("displayed_name", {})
                        if isinstance(displayed_name, dict) and "vi" in displayed_name:
                            domain_id = domain.get("id")
                            if domain_id:
                                domain_mapping[displayed_name["vi"]] = str(domain_id)
                
                # Break sau khi xử lý list
                break
                
            elif isinstance(data, dict) and "content" in data:
                domains = data["content"]
                if not domains:
                    break
                
                # Tạo mapping từ displayed_name.vi sang id
                for domain in domains:
                    if isinstance(domain, dict):
                        displayed_name = domain.get("displayed_name", {})
                        if isinstance(displayed_name, dict) and "vi" in displayed_name:
                            domain_id = domain.get("id")
                            if domain_id:
                                domain_mapping[displayed_name["vi"]] = str(domain_id)
                
                # Kiểm tra xem còn trang nào không
                total_pages = data.get("totalPages")
                if total_pages is not None:
                    if page >= total_pages - 1:
                        break
                else:
                    # Nếu không có totalPages, kiểm tra xem có ít hơn size items không
                    if len(domains) < size:
                        break
            else:
                # Không phải format mong đợi, break
                logger.warning(f"[CSV] API trả về format không mong đợi: {type(data)}")
                break
            
            page += 1
        
        if page >= max_pages:
            logger.warning(f"[CSV] Đã đạt giới hạn {max_pages} trang, dừng lại")
        
        logger.info(f"[CSV] Đã lấy được {len(domain_mapping)} domains từ {page + 1} trang")
        return domain_mapping
        
    except Exception as e:
        logger.error(f"[CSV] Lỗi khi lấy domains từ API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Không thể lấy thông tin domains từ API: {str(e)}"
        )


def match_domain_id(domain_name: str, domain_mapping: Dict[str, str]) -> Optional[str]:
    """
    Match domain name với ID dựa trên displayed_name.vi
    """
    if not domain_name:
        return None
    
    # Tìm exact match
    if domain_name in domain_mapping:
        return domain_mapping[domain_name]
    
    # Tìm partial match (case-insensitive)
    domain_name_lower = domain_name.lower().strip()
    for key, value in domain_mapping.items():
        if key.lower().strip() == domain_name_lower:
            return value
    
    return None


def call_llm(prompt: str) -> str:
    """
    Gọi LLM local với prompt và trả về response text
    Chỉ sử dụng model local, không có fallback
    """
    if llm is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model local chưa được khởi tạo. Kiểm tra cấu hình tại {Config.LOCAL_LLM_BASE_URL}"
        )
    
    try:
        # Sử dụng langchain LLM
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[CSV] Lỗi khi gọi model local: {error_msg}")
        
        # Kiểm tra xem có phải lỗi connection không
        if "connection" in error_msg.lower() or "refused" in error_msg.lower() or "10061" in error_msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Không thể kết nối đến model local tại {Config.LOCAL_LLM_BASE_URL}. "
                    f"Vui lòng đảm bảo model local đang chạy. "
                    f"Lỗi: {error_msg}"
                )
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Lỗi khi gọi model local: {error_msg}"
            )


def generate_title(item: str) -> Dict[str, str]:
    """
    Generate title song ngữ từ item sử dụng GPT 20B
    """
    try:
        prompt = f"""
        Hãy tạo một tiêu đề ngắn gọn và rõ ràng cho mục tiêu can thiệp sau đây, theo định dạng song ngữ Việt-Anh.

        Mục tiêu can thiệp: "{item}"

        Yêu cầu:
        1. Tiêu đề phải ngắn gọn, súc tích (không quá 20 từ)
        2. Phù hợp với trẻ em đặc biệt (autism, ADHD, khuyết tật phát triển)
        3. Sử dụng ngôn ngữ chuyên nghiệp nhưng dễ hiểu

        Định dạng trả về (JSON, chỉ trả về JSON không có markdown):
        {{
            "en": "Title in English",
            "vi": "Tiêu đề bằng tiếng Việt"
        }}

        Ví dụ:
        {{
            "en": "Turn toward the source of the cheerful sounds.",
            "vi": "Nhìn về phía phát ra những âm thanh vui nhộn"
        }}
        """
        
        ai_response = call_llm(prompt).strip()
        
        if not ai_response:
            raise ValueError("Không nhận được response từ AI")
        
        # Xử lý response để extract JSON
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]
        if ai_response.startswith("```"):
            ai_response = ai_response[3:]
        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]
        ai_response = ai_response.strip()
        
        # Tìm JSON trong response
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("Không tìm thấy JSON trong response")
        
        json_str = ai_response[start_idx:end_idx]
        title_data = json.loads(json_str)
        
        # Validate structure
        if "vi" not in title_data or "en" not in title_data:
            raise ValueError("JSON không có đủ trường vi và en")
        
        return title_data
        
    except HTTPException:
        # Re-raise HTTPException từ call_llm
        raise
    except Exception as e:
        logger.error(f"[CSV] Lỗi khi generate title cho item '{item}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi generate title cho item '{item[:50]}...': {str(e)}"
        )


def generate_description(cirtial_item: str) -> Dict[str, str]:
    """
    Generate description song ngữ từ cirtialItem sử dụng GPT 20B
    """
    try:
        prompt = f"""
        Hãy tạo một mô tả chi tiết và chuyên nghiệp cho mục tiêu can thiệp sau đây, theo định dạng song ngữ Việt-Anh.

        Mục tiêu can thiệp quan trọng: "{cirtial_item}"

        Yêu cầu:
        1. Mô tả phải rõ ràng, cụ thể và có thể đo lường được
        2. Phù hợp với trẻ em đặc biệt (autism, ADHD, khuyết tật phát triển)
        3. Sử dụng ngôn ngữ chuyên nghiệp nhưng dễ hiểu
        4. Bao gồm thời gian, khoảng cách, hoặc điều kiện cụ thể nếu có thể
        5. Mô tả nên dài khoảng 150-300 từ cho mỗi ngôn ngữ

        Định dạng trả về (JSON, chỉ trả về JSON không có markdown):
        {{
            "en": "Detailed description in English",
            "vi": "Mô tả chi tiết bằng tiếng Việt"
        }}

        Ví dụ:
        {{
            "en": "This intervention goal aims to improve the child's ability to localize sound. Specifically, the child will turn their head and look towards the source of a sound when it is presented in front of them, within a range of 0.5 meters to 1 meter. The success criterion is that the child performs this behavior successfully in at least 70% of trials during each intervention session, with each session providing 5 to 6 opportunities for the child to respond to the sound. This intervention is designed to support children with developmental challenges, including autism, ADHD, and other developmental disabilities, by enhancing their awareness of the surrounding environment through auditory perception.",
            "vi": "Mục tiêu can thiệp này nhằm cải thiện khả năng định vị âm thanh của trẻ. Cụ thể, trẻ sẽ quay đầu và nhìn về phía nguồn âm thanh khi âm thanh đó được phát ra ở vị trí gần phía trước mặt trẻ, trong khoảng cách từ 0.5 mét đến 1 mét. Tiêu chí đánh giá thành công là trẻ thực hiện hành vi này thành công ít nhất 70% số lần trong mỗi buổi can thiệp, với mỗi buổi sẽ có từ 5 đến 6 cơ hội để trẻ phản ứng với âm thanh. Việc can thiệp này được thiết kế để hỗ trợ trẻ em có các khó khăn về phát triển, bao gồm tự kỷ (autism), tăng động giảm chú ý (ADHD) và các khuyết tật phát triển khác, bằng cách tăng cường nhận thức về môi trường xung quanh thông qua thính giác."
        }}
        """
        
        ai_response = call_llm(prompt).strip()
        
        if not ai_response:
            raise ValueError("Không nhận được response từ AI")
        
        # Xử lý response để extract JSON
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]
        if ai_response.startswith("```"):
            ai_response = ai_response[3:]
        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]
        ai_response = ai_response.strip()
        
        # Tìm JSON trong response
        start_idx = ai_response.find('{')
        end_idx = ai_response.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("Không tìm thấy JSON trong response")
        
        json_str = ai_response[start_idx:end_idx]
        description_data = json.loads(json_str)
        
        # Validate structure
        if "vi" not in description_data or "en" not in description_data:
            raise ValueError("JSON không có đủ trường vi và en")
        
        return description_data
        
    except HTTPException:
        # Re-raise HTTPException từ call_llm
        raise
    except Exception as e:
        logger.error(f"[CSV] Lỗi khi generate description cho cirtialItem '{cirtial_item}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi generate description cho cirtialItem '{cirtial_item[:50]}...': {str(e)}"
        )


@router.post("/process-csv", response_model=CSVProcessResponse)
async def process_csv(file: UploadFile = File(...)):
    """
    Xử lý file CSV với các cột: CTCT, minAgeMonths, maxAgeMonth, domain, item, cirtialItem
    
    Thực hiện:
    1. Match domain với ID từ database dựa trên displayed_name.vi
    2. Generate title từ cột item
    3. Generate description từ cột cirtialItem
    4. Thêm cột level (số thứ tự cho các dòng có cùng item)
    """
    try:
        # Kiểm tra file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File phải có định dạng CSV")
        
        logger.info(f"[CSV] Bắt đầu xử lý file: {file.filename}")
        
        # Đọc file CSV
        content = await file.read()
        content_str = content.decode('utf-8-sig')  # utf-8-sig để xử lý BOM
        
        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(content_str))
        rows = list(csv_reader)
        
        if not rows:
            raise HTTPException(status_code=400, detail="File CSV không có dữ liệu")
        
        logger.info(f"[CSV] Đã đọc được {len(rows)} dòng từ CSV")
        
        # Kiểm tra các cột bắt buộc
        required_columns = ['CTCT', 'minAgeMonths', 'maxAgeMonth', 'domain', 'item', 'cirtialItem']
        first_row_keys = set(rows[0].keys())
        missing_columns = [col for col in required_columns if col not in first_row_keys]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"File CSV thiếu các cột: {', '.join(missing_columns)}"
            )
        
        # Lấy danh sách domains từ API
        logger.info("[CSV] Đang lấy thông tin domains từ API...")
        domain_mapping = fetch_domains()
        
        # Tính toán level cho tất cả các dòng dựa trên file đầu vào (trước khi xử lý)
        logger.info("[CSV] Đang tính toán level cho các dòng...")
        item_counters = defaultdict(int)
        level_by_index = {}  # Map từ index sang level
        
        for idx, row in enumerate(rows):
            item = row.get('item', '').strip()
            item_counters[item] += 1
            level_by_index[idx] = item_counters[item]
        
        # Xử lý từng dòng
        processed_data = []
        errors = []
        
        logger.info("[CSV] Bắt đầu xử lý từng dòng...")
        
        for idx, row in enumerate(rows, start=1):
            try:
                # Lấy các giá trị từ CSV
                ctct = row.get('CTCT', '').strip()
                min_age = row.get('minAgeMonths', '').strip()
                max_age = row.get('maxAgeMonth', '').strip()
                domain_name = row.get('domain', '').strip()
                item = row.get('item', '').strip()
                cirtial_item = row.get('cirtialItem', '').strip()
                
                # Lấy level đã tính trước đó (dựa trên index trong rows ban đầu)
                level = level_by_index.get(idx - 1, 1)  # idx - 1 vì enumerate bắt đầu từ 1
                
                # Match domain với ID
                domain_id = match_domain_id(domain_name, domain_mapping)
                if not domain_id:
                    errors.append(f"Dòng {idx}: Không tìm thấy domain ID cho '{domain_name}'")
                
                # Generate title từ item
                logger.info(f"[CSV] Dòng {idx}: Đang generate title cho item '{item[:50]}...'")
                title = generate_title(item)
                
                # Generate description từ cirtialItem
                logger.info(f"[CSV] Dòng {idx}: Đang generate description cho cirtialItem '{cirtial_item[:50]}...'")
                description = generate_description(cirtial_item)
                
                # Tạo dữ liệu đã xử lý
                processed_row = {
                    "CTCT": ctct,
                    "minAgeMonths": min_age,
                    "maxAgeMonth": max_age,
                    "domain": domain_name,
                    "domainId": domain_id,
                    "item": item,
                    "cirtialItem": cirtial_item,
                    "title": title,
                    "description": description,
                    "level": level
                }
                
                processed_data.append(processed_row)
                
            except Exception as e:
                error_msg = f"Dòng {idx}: Lỗi khi xử lý - {str(e)}"
                logger.error(f"[CSV] {error_msg}")
                errors.append(error_msg)
        
        logger.info(f"[CSV] Hoàn thành xử lý {len(processed_data)} dòng")
        
        return CSVProcessResponse(
            message=f"Đã xử lý thành công {len(processed_data)} dòng",
            total_rows=len(processed_data),
            processed_data=processed_data,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[CSV] Lỗi khi xử lý file CSV: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý file CSV: {str(exc)}"
        )

