import pytest
from fastapi.testclient import TestClient
from main import app
from services.child_harness_service import child_harness_service

client = TestClient(app)


def test_get_demo_child_context():
    """Kiểm tra lấy context cá nhân hóa của trẻ demo"""
    response = client.get("/api/v1/harness/demo_child_01/context")
    assert response.status_code == 200
    data = response.json()
    assert data["child_id"] == "demo_child_01"
    assert "Bé Minh Nam" in data["child_name"]
    assert "khủng long T-Rex" in data["dominant_interests"]
    assert "tiếng chuông to đột ngột > 70dB" in data["triggers_to_avoid"]
    assert "NGỮ CẢNH HỒ SƠ BÉ" in data["prompt_injection_snippet"]
    assert "khủng long" in data["prompt_injection_snippet"]


def test_create_and_update_profile():
    """Kiểm tra khởi tạo và cập nhật hồ sơ Harness cho bé mới"""
    child_id = "test_child_new_01"
    update_payload = {
        "child_name": "Bé Bắp",
        "age": 4,
        "gender": "Nữ",
        "diagnosis": "Chậm phát triển ngôn ngữ",
        "dominant_interests": ["chú thỏ trắng", "búp bê", "màu hồng"],
        "triggers_to_avoid": ["người lạ đột ngột chạm vào", "tiếng sấm sét"],
        "target_skills": ["gọi tên người thân", "nói câu 2-3 từ"]
    }
    response = client.post(f"/api/v1/harness/{child_id}/profile", json=update_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["child_id"] == child_id
    assert data["child_name"] == "Bé Bắp"
    assert data["age"] == 4
    assert "chú thỏ trắng" in data["dominant_interests"]
    assert "tiếng sấm sét" in data["triggers_to_avoid"]
    assert "chú thỏ trắng" in data["prompt_injection_snippet"]


def test_add_memory_interest_and_trigger():
    """Kiểm tra thêm mẩu trí nhớ (sở thích và trigger)"""
    child_id = "test_child_new_01"

    # 1. Thêm sở thích mới
    resp1 = client.post(
        f"/api/v1/harness/{child_id}/add-memory",
        json={"memory_type": "interest", "content": "siêu nhân gao"}
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "siêu nhân gao" in data1["dominant_interests"]
    assert "siêu nhân gao" in data1["prompt_injection_snippet"]

    # 2. Thêm trigger mới
    resp2 = client.post(
        f"/api/v1/harness/{child_id}/add-memory",
        json={"memory_type": "trigger", "content": "tiếng còi hú inh ỏi"}
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "tiếng còi hú inh ỏi" in data2["triggers_to_avoid"]
    assert "tiếng còi hú inh ỏi" in data2["prompt_injection_snippet"]


def test_sync_test_result_trigger():
    """Kiểm tra trigger tự động đồng bộ kết quả bài test vào Memory Harness"""
    child_id = "test_child_screening_01"

    sync_payload = {
        "test_name": "ASQ-3 48 Tháng",
        "category_scores": [
            {
                "categoryName": "Giao tiếp",
                "score": 15.0,
                "cutoff": 30.0,
                "status": "FAIL",
                "zone": "BLACK"
            },
            {
                "categoryName": "Vận động thô",
                "score": 50.0,
                "cutoff": 35.0,
                "status": "PASS",
                "zone": "WHITE"
            }
        ],
        "sensory_triggers": ["sợ tiếng chuông báo cháy", "khó chịu với vải len thô"]
    }

    response = client.post(f"/api/v1/harness/{child_id}/sync-test-result", json=sync_payload)
    assert response.status_code == 200
    data = response.json()

    # Xác minh domain FAIL nạp vào target_skills
    assert any("Giao tiếp" in s for s in data["target_skills"])
    # Xác minh domain PASS nạp vào mastered_skills
    assert any("Vận động thô" in s for s in data["mastered_skills"])
    # Xác minh sensory triggers nạp vào triggers_to_avoid
    assert "sợ tiếng chuông báo cháy" in data["triggers_to_avoid"]
    # Xác minh prompt injection snippet được cập nhật
    assert "Giao tiếp" in data["prompt_injection_snippet"]
    assert "sợ tiếng chuông báo cháy" in data["prompt_injection_snippet"]


def test_negative_cases():
    """Kiểm thử dữ liệu rỗng và không hợp lệ (Rule 3)"""
    # 1. child_id rỗng
    resp1 = client.get("/api/v1/harness/%20%20/context")
    assert resp1.status_code in [400, 404]

    # 2. Add memory với content rỗng
    resp2 = client.post(
        "/api/v1/harness/demo_child_01/add-memory",
        json={"memory_type": "interest", "content": "   "}
    )
    assert resp2.status_code == 400
