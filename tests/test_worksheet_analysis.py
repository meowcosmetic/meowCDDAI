import io
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app

client = TestClient(app)

def create_dummy_worksheet_image(image_type: str = "tracing") -> bytes:
    """Tạo ảnh bài tập giả lập trong bộ nhớ để kiểm thử"""
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    if image_type == "tracing":
        # Vẽ một đường nét lượn sóng hoặc zíc zắc
        points = [(30, 150), (80, 70), (140, 220), (200, 80), (270, 150)]
        draw.line(points, fill=(20, 30, 180), width=4)
    elif image_type == "coloring":
        # Vẽ một hình tròn khuôn và tô màu bên trong
        draw.ellipse([60, 60, 240, 240], outline=(0, 0, 0), width=3)
        draw.ellipse([80, 80, 220, 220], fill=(230, 40, 50))
    else:
        # Tự do
        draw.rectangle([50, 50, 250, 250], outline=(0, 0, 0), width=2)
        draw.line([(60, 60), (240, 240)], fill=(0, 150, 50), width=3)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_analyze_worksheet_empty_file_fails():
    """Negative: Upload file 0 byte trả về 400"""
    response = client.post(
        "/api/v1/vision/analyze-worksheet",
        files={"file": ("empty.png", b"", "image/png")},
        data={"worksheet_type": "TRACING"}
    )
    assert response.status_code == 400
    assert "rong" in response.json()["detail"].lower()

def test_analyze_worksheet_corrupted_image_fails():
    """Negative: Upload file bytes rác không phải ảnh -> 400"""
    response = client.post(
        "/api/v1/vision/analyze-worksheet",
        files={"file": ("corrupted.png", b"not_a_real_image_bytes_12345", "image/png")},
        data={"worksheet_type": "TRACING"}
    )
    assert response.status_code == 400

def test_analyze_tracing_worksheet_success():
    """Happy path: Phân tích bài tập đồ nét tracing hợp lệ"""
    img_bytes = create_dummy_worksheet_image("tracing")
    response = client.post(
        "/api/v1/vision/analyze-worksheet",
        files={"file": ("worksheet_tracing.png", img_bytes, "image/png")},
        data={"worksheet_type": "TRACING", "child_id": "child-001"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["child_id"] == "child-001"
    assert data["worksheet_type"] == "TRACING"
    
    metrics = data["metrics"]
    assert "overall_motor_score" in metrics
    assert "stability_score" in metrics
    assert "boundary_compliance_score" in metrics
    assert "completion_rate" in metrics
    assert metrics["overall_motor_score"] > 0
    assert len(data["technical_observations"]) > 0
    assert len(data["pedagogical_suggestions"]) > 0
    assert "disclaimer" in data

def test_analyze_coloring_worksheet_success():
    """Happy path: Phân tích bài tập tô màu coloring hợp lệ"""
    img_bytes = create_dummy_worksheet_image("coloring")
    response = client.post(
        "/api/v1/vision/analyze-worksheet",
        files={"file": ("coloring.png", img_bytes, "image/png")},
        data={"worksheet_type": "COLORING"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["worksheet_type"] == "COLORING"
    assert data["metrics"]["boundary_compliance_score"] > 0
