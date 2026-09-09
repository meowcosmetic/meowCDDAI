"""
Speech-to-Text (STT) Service for meowAI.
Supports audio normalization (via ffmpeg) and multi-provider transcription
(Gemini, OpenAI Whisper, Local HuggingFace Transformers with automatic fallback).
"""

import io
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple
from pydantic import BaseModel

# from config import settings

logger = logging.getLogger(__name__)

MAX_STT_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB limit


class STTResponse(BaseModel):
    text: str
    language: str = "vi"
    duration_seconds: float
    confidence: float
    provider_used: str


class STTService:
    def __init__(self):
        self._local_pipeline = None

    def normalize_and_probe_audio(self, audio_bytes: bytes, filename: str) -> Tuple[bytes, float]:
        """
        Validate audio integrity, probe duration, and convert to 16kHz mono WAV PCM.
        Raises ValueError if audio is corrupt or not a valid audio format.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("EMPTY_AUDIO_FILE")

        if len(audio_bytes) > MAX_STT_UPLOAD_BYTES:
            raise ValueError("FILE_TOO_LARGE")

        ext = os.path.splitext(filename)[1].lower().lstrip(".") or "bin"
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as in_file:
            in_file.write(audio_bytes)
            in_path = in_file.name

        out_path = in_path + ".wav"
        try:
            # 1. Probe duration using ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                in_path
            ]
            probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            duration = 0.0
            if probe_res.returncode == 0:
                try:
                    duration = float(probe_res.stdout.decode().strip())
                except ValueError:
                    duration = 0.0

            # 2. Convert to 16kHz mono 16-bit PCM WAV
            convert_cmd = [
                "ffmpeg", "-y",
                "-i", in_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                out_path
            ]
            conv_res = subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if conv_res.returncode != 0:
                logger.warning(f"ffmpeg conversion failed: {conv_res.stderr.decode('utf-8', errors='ignore')}")
                raise ValueError("INVALID_AUDIO_FORMAT")

            with open(out_path, "rb") as f:
                wav_bytes = f.read()

            # If ffprobe didn't get duration, estimate from 16kHz 16-bit mono PCM (32000 bytes/sec)
            if duration <= 0.0 and len(wav_bytes) > 44:
                duration = round((len(wav_bytes) - 44) / 32000.0, 2)

            return wav_bytes, duration
        finally:
            if os.path.exists(in_path):
                try: os.remove(in_path)
                except Exception: pass
            if os.path.exists(out_path):
                try: os.remove(out_path)
                except Exception: pass

    async def transcribe(
        self,
        file_bytes: bytes,
        filename: str,
        language: str = "vi",
        provider: str = "auto",
        prompt: Optional[str] = None
    ) -> STTResponse:
        """Transcribe audio bytes to text with automatic provider selection and fallback."""
        wav_bytes, duration = self.normalize_and_probe_audio(file_bytes, filename)

        provider_req = (provider or "auto").lower()
        text: Optional[str] = None
        used_provider: str = ""

        # 1. Attempt Gemini if requested or auto
        if provider_req in ("auto", "gemini"):
            try:
                text = await self._transcribe_with_gemini(wav_bytes, language, prompt)
                if text is not None:
                    used_provider = "gemini"
            except Exception as e:
                logger.warning(f"[STT] Gemini transcription failed: {e}")

        # 2. Attempt OpenAI if requested or fallback from gemini in auto mode
        if text is None and provider_req in ("auto", "openai"):
            try:
                text = await self._transcribe_with_openai(wav_bytes, language, prompt)
                if text is not None:
                    used_provider = "openai"
            except Exception as e:
                logger.warning(f"[STT] OpenAI transcription failed: {e}")

        # 3. Attempt Local Transformers if requested or fallback from cloud
        if text is None:
            try:
                text = await self._transcribe_with_local(wav_bytes, language)
                if text is not None:
                    used_provider = "local_transformers"
            except Exception as e:
                logger.warning(f"[STT] Local transcription failed: {e}")

        # 4. Final safety default if all models unavailable
        if text is None:
            text = ""
            used_provider = "unavailable"
            confidence = 0.0
        else:
            text = text.strip()
            confidence = 0.95 if used_provider in ("gemini", "openai") else 0.85

        return STTResponse(
            text=text,
            language=language,
            duration_seconds=duration,
            confidence=confidence,
            provider_used=used_provider
        )

    async def _transcribe_with_gemini(self, wav_bytes: bytes, language: str, prompt: Optional[str]) -> Optional[str]:
        """Transcribe using Google Gemini Multimodal Audio API."""
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY") or getattr(settings, "AI_API_KEY", None)
            if not api_key:
                return None

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            system_instruction = (
                f"Bạn là một chuyên gia nhận diện giọng nói (STT) Tiếng Việt. "
                f"Hãy nghe đoạn âm thanh sau và chép lại chính xác từng từ bằng tiếng Việt chuẩn ngữ pháp, có dấu đầy đủ. "
                f"Chỉ xuất văn bản được chép lại, tuyệt đối không thêm lời chào, giải thích hay định dạng markdown."
            )
            if prompt:
                system_instruction += f" Ngữ cảnh tham khảo: {prompt}."

            audio_part = {
                "mime_type": "audio/wav",
                "data": wav_bytes
            }
            response = await model.generate_content_async([system_instruction, audio_part])
            return response.text.strip() if response and response.text else ""
        except Exception as e:
            logger.info(f"[STT] Gemini API not available: {e}")
            return None

    async def _transcribe_with_openai(self, wav_bytes: bytes, language: str, prompt: Optional[str]) -> Optional[str]:
        """Transcribe using OpenAI Whisper API."""
        try:
            import httpx
            api_key = getattr(settings, "OPENAI_API_KEY", None) or getattr(settings, "AI_API_KEY", None)
            base_url = getattr(settings, "OPENAI_BASE_URL", None) or getattr(settings, "AI_BASE_URL", None) or "https://api.openai.com/v1"
            if not api_key:
                return None

            url = f"{base_url.rstrip('/')}/audio/transcriptions"
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            data = {"model": "whisper-1", "language": language}
            if prompt:
                data["prompt"] = prompt

            headers = {"Authorization": f"Bearer {api_key}"}
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(url, files=files, data=data, headers=headers)
                if res.status_code == 200:
                    return res.json().get("text", "").strip()
                logger.info(f"[STT] OpenAI Whisper returned status {res.status_code}")
                return None
        except Exception as e:
            logger.info(f"[STT] OpenAI Whisper API not available: {e}")
            return None

    async def _transcribe_with_local(self, wav_bytes: bytes, language: str) -> Optional[str]:
        """Transcribe using local HuggingFace Transformers pipeline or soundfile inspect."""
        try:
            import soundfile as sf
            audio_data, sr = sf.read(io.BytesIO(wav_bytes))
            # Basic energy / silence check
            import numpy as np
            if np.max(np.abs(audio_data)) < 0.01:
                return ""  # Silent audio

            if self._local_pipeline is None:
                from transformers import pipeline
                # Use lightweight default whisper-tiny for fast local transcription
                self._local_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny",
                    generate_kwargs={"language": language, "task": "transcribe"}
                )
            result = self._local_pipeline(audio_data)
            return result.get("text", "").strip()
        except Exception as e:
            logger.info(f"[STT] Local pipeline not loaded: {e}")
            return None


# Global singleton instance
stt_service = STTService()
