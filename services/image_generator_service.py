"""
Image Generation Client for 9router Gateway
Gọi model sinh ảnh (vd: cl/google/gemini-2.5-flash-image) qua 9router OpenAI-compatible API
Lưu file vào ổ đĩa và trả về đường dẫn tĩnh.
"""

import os
import time
import base64
import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("ImageGeneratorClient")

NINEROUTER_URL = os.getenv("NINEROUTER_URL", "http://127.0.0.1:20128/v1")
IMAGE_MODEL = os.getenv("SOCIAL_STORY_IMAGE_MODEL", "cl/google/gemini-2.5-flash-image")
STATIC_UPLOAD_DIR = os.getenv("STATIC_UPLOAD_DIR", "uploads/social_stories")


class ImageGeneratorClient:
    """Client giao tiếp với 9router để tạo tranh minh họa sách truyện thiếu nhi."""

    @classmethod
    async def generate_storybook_illustration(
        cls,
        prompt: str,
        story_id: str,
        page_number: int,
        child_name: str = "Bé",
        child_gender: Optional[str] = None,
    ) -> Optional[str]:
        """
        Sinh 1 tranh minh họa sách thiếu nhi, lưu file PNG và trả về URL file tĩnh.
        """
        # Tạo prompt chuẩn tranh màu nước thiếu nhi ấm áp
        gender_desc = "cute boy" if child_gender and child_gender.lower() in ["male", "nam", "trai"] else (
            "cute girl" if child_gender and child_gender.lower() in ["female", "nu", "gai"] else "cute child"
        )
        
        full_prompt = (
            f"Children's book watercolor illustration, cute picture book style, soft pastel colors, "
            f"friendly and comforting atmosphere. A {gender_desc} named {child_name}. "
            f"Scene: {prompt}. "
            f"No scary elements, reassuring, warm lighting, high quality storybook art."
        )

        logger.info(f"[ImageGeneratorClient] Calling 9router model: {IMAGE_MODEL} for story {story_id} page {page_number}")

        # Thư mục lưu trữ
        save_dir = os.path.join(STATIC_UPLOAD_DIR, story_id)
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"page_{page_number}.png"
        file_path = os.path.join(save_dir, file_name)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 1. Thử gọi qua endpoint chat completions của 9router với model image
                chat_payload = {
                    "model": IMAGE_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Generate an illustration: {full_prompt}"
                        }
                    ],
                }

                response = await client.post(
                    f"{NINEROUTER_URL}/chat/completions",
                    json=chat_payload,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        # Kiểm tra nếu có image url hoặc base64 trong response content
                        content = message.get("content", "")
                        
                        # Trường hợp response trả về base64 data URI
                        if "data:image" in content and "base64," in content:
                            b64_data = content.split("base64,")[1].split('"')[0].split(")")[0]
                            with open(file_path, "wb") as f:
                                f.write(base64.b64decode(b64_data))
                            logger.info(f"[ImageGeneratorClient] Saved base64 image to {file_path}")
                            return f"/media/social_stories/{story_id}/{file_name}"

                # 2. Thử endpoint /images/generations chuẩn OpenAI
                img_payload = {
                    "model": IMAGE_MODEL,
                    "prompt": full_prompt,
                    "n": 1,
                    "size": "512x512",
                    "response_format": "b64_json"
                }

                img_resp = await client.post(
                    f"{NINEROUTER_URL}/images/generations",
                    json=img_payload,
                    headers={"Content-Type": "application/json"}
                )

                if img_resp.status_code == 200:
                    img_data = img_resp.json()
                    items = img_data.get("data", [])
                    if items:
                        item = items[0]
                        if "b64_json" in item:
                            with open(file_path, "wb") as f:
                                f.write(base64.b64decode(item["b64_json"]))
                            return f"/media/social_stories/{story_id}/{file_name}"
                        elif "url" in item:
                            # Tải từ temp URL về máy chủ để lưu vĩnh viễn vào DB
                            temp_url = item["url"]
                            down_resp = await client.get(temp_url)
                            if down_resp.status_code == 200:
                                with open(file_path, "wb") as f:
                                    f.write(down_resp.content)
                                return f"/media/social_stories/{story_id}/{file_name}"

                logger.error(
                    f"[ImageGeneratorClient] 9router returned error status {img_resp.status_code}: {img_resp.text}"
                )
                raise RuntimeError(
                    f"9router Image Generation failed with status {img_resp.status_code}: {img_resp.text}"
                )
        except Exception as e:
            logger.error(f"[ImageGeneratorClient] ❌ Error generating image: {e}")
            raise RuntimeError(f"Không thể tạo tranh minh họa từ 9router: {e}") from e
