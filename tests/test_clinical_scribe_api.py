import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_medical_summary_happy_path():
    """Kiểm tra kịch bản sinh báo cáo y khoa chuẩn DSM-5 đầy đủ 4 nguồn dữ liệu"""
    payload = {
        "child_id": "child-001",
        "child_name": "Bé Nam",
        "child_age": 5,
        "evaluation_period_days": 30,
        "standard_framework": "DSM_5",
        "test_scores_summary": "M-CHAT-R: Nguy cơ trung bình (5 điểm); ASQ-3: Vùng giám sát giao tiếp",
        "recent_meltdowns_count": 1,
        "top_triggers": ["Tiếng máy mài", "Người lạ chạm vào người"],
        "comfort_items": ["Chiếc xe ô tô đỏ quen thuộc"],
        "target_skills": ["Giao tiếp mắt", "Chỉ ngón tay", "Chia sẻ đồ chơi"],
        "mastered_skills": ["Nhận biết màu sắc cơ bản"]
    }
    response = client.post("/api/v1/clinical/generate-medical-summary", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Kiểm tra cấu trúc DSM-5
    assert data["child_id"] == "child-001"
    assert data["standard_framework"] == "DSM_5"
    assert "criterion_a" in data
    assert "criterion_b" in data
    assert "standardized_percentiles" in data
    assert "doctor_recommendation_draft" in data

    # Kiểm tra Trục A & B
    assert "LEVEL_1" in data["criterion_a"]["severity_level"]
    assert "social_emotional_reciprocity" in data["criterion_a"]["sub_criteria_findings"]
    assert "Tiếng máy mài" in data["criterion_b"]["clinical_evidence"]
    assert "Chiếc xe ô tô đỏ quen thuộc" in data["criterion_b"]["clinical_evidence"]

def test_generate_medical_summary_high_meltdowns():
    """Kiểm tra kịch bản trẻ có nhiều đợt bùng nổ (Meltdowns >= 3) nâng mức độ Trục B"""
    payload = {
        "child_id": "child-002",
        "child_name": "Bé Minh",
        "child_age": 6,
        "recent_meltdowns_count": 4,
        "top_triggers": ["Tiếng chuông báo", "Thay đổi giáo viên đột ngột"],
        "comfort_items": ["Gấu bông Pooh"]
    }
    response = client.post("/api/v1/clinical/generate-medical-summary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["criterion_b"]["severity_level"] == "MODERATE_TO_HIGH"
    assert "4 đợt khủng hoảng hành vi" in data["criterion_b"]["clinical_evidence"]

def test_generate_medical_summary_minimal_data():
    """Kiểm tra trẻ mới nhập học, chưa có bài test hay triggers -> sinh báo cáo an toàn"""
    payload = {
        "child_id": "child-003",
        "child_name": "Bé An",
        "child_age": 4
    }
    response = client.post("/api/v1/clinical/generate-medical-summary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["criterion_a"]["severity_level"] == "LEVEL_1_REQUIRING_SUPPORT"
    assert "doctor_recommendation_draft" in data

def test_negative_empty_payload():
    """Rule 3: Kiểm thử Negative payload rỗng {}"""
    response = client.post("/api/v1/clinical/generate-medical-summary", json={})
    assert response.status_code == 422

def test_negative_empty_child_id():
    """Rule 3: Kiểm thử Negative child_id rỗng"""
    payload = {
        "child_id": "   ",
        "child_name": "Bé Nam"
    }
    response = client.post("/api/v1/clinical/generate-medical-summary", json=payload)
    assert response.status_code == 400
    assert "child_id cannot be empty" in response.json()["detail"]
