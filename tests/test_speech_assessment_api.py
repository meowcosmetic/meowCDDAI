"""
Tests for Speech Assessment (Pronunciation Scoring) API in meowAI.
Covers happy paths, unhappy paths, boundary values, empty payloads, and invalid files (Rule 3).
"""

import io
import math
import struct
import wave
import sys
import os
from pathlib import Path

# Add meowAI root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def generate_dummy_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a valid 16-bit mono PCM WAV in-memory."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        total_samples = int(duration_seconds * sample_rate)
        for i in range(total_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            wav_file.writeframes(struct.pack('<h', val))
    return buf.getvalue()


def test_assess_empty_file_fails():
    """Unhappy Path: Upload 0-byte file must return 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    data = {"reference_text": "con cá"}
    response = client.post("/v1/speech/assess-pronunciation", files=files, data=data)
    assert response.status_code == 400
    detail = response.json().get("detail", {})
    assert detail.get("error") == "EMPTY_AUDIO_FILE"


def test_assess_empty_reference_text_fails():
    """Unhappy Path: Empty reference text must return 400."""
    wav_bytes = generate_dummy_wav(duration_seconds=1.0)
    files = {"file": ("test.wav", wav_bytes, "audio/wav")}
    data = {"reference_text": "   "}
    response = client.post("/v1/speech/assess-pronunciation", files=files, data=data)
    assert response.status_code == 400
    detail = response.json().get("detail", {})
    assert detail.get("error") == "EMPTY_REFERENCE_TEXT"


def test_assess_corrupt_non_audio_file_fails():
    """Unhappy Path: Corrupt non-audio file disguised as wav must return 400."""
    garbage_bytes = b"This is plain text not an audio waveform."
    files = {"file": ("fake.wav", garbage_bytes, "audio/wav")}
    data = {"reference_text": "quả táo"}
    response = client.post("/v1/speech/assess-pronunciation", files=files, data=data)
    assert response.status_code == 400
    detail = response.json().get("detail", {})
    assert detail.get("error") in ("INVALID_AUDIO_FORMAT", "EMPTY_AUDIO_FILE")


def test_assess_file_too_large_fails():
    """Boundary Test: File exceeding 15 MB must return 413."""
    from services.stt_service import MAX_STT_UPLOAD_BYTES
    fake_huge_bytes = b"x" * (MAX_STT_UPLOAD_BYTES + 100)
    files = {"file": ("huge.wav", fake_huge_bytes, "audio/wav")}
    data = {"reference_text": "bông hoa"}
    response = client.post("/v1/speech/assess-pronunciation", files=files, data=data)
    assert response.status_code == 413
    detail = response.json().get("detail", {})
    assert detail.get("error") == "FILE_TOO_LARGE"


def test_assess_valid_audio_success():
    """Happy Path: Upload valid WAV audio with reference text."""
    wav_bytes = generate_dummy_wav(duration_seconds=1.0)
    files = {"file": ("test_child.wav", wav_bytes, "audio/wav")}
    data = {"reference_text": "con cá", "language": "vi"}
    response = client.post("/v1/speech/assess-pronunciation", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert "overall_score" in res_data
    assert "word_details" in res_data
    assert len(res_data["word_details"]) == 2
    assert "pedagogical_feedback" in res_data
    assert res_data["reference_text"] == "con cá"
