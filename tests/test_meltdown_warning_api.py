import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_safe_zone_telemetry():
    """Kiểm tra kịch bản trẻ tương tác vui vẻ, bình tĩnh (SAFE_ZONE)"""
    payload = {
        "child_id": "child-001",
        "child_name": "Bé Nam",
        "window_seconds": 30,
        "tap_events_count": 15,
        "consecutive_fails": 0,
        "idle_seconds": 2,
        "erratic_touch_count": 1,
        "average_latency_ms": 450,
        "comfort_item": "Chiếc xe ô tô đỏ"
    }
    response = client.post("/api/v1/telemetry/evaluate-frustration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "SAFE_ZONE"
    assert data["frustration_score"] < 45.0
    assert data["calm_mode_recommended"] is False
    assert data["calm_mode_action"] == "CONTINUE_NORMAL"

def test_moderate_frustration_telemetry():
    """Kiểm tra kịch bản bắt đầu bồn chồn (MODERATE_FRUSTRATION)"""
    payload = {
        "child_id": "child-001",
        "child_name": "Bé Nam",
        "window_seconds": 30,
        "tap_events_count": 60,
        "consecutive_fails": 3,
        "idle_seconds": 12,
        "erratic_touch_count": 4,
        "average_latency_ms": 300,
        "comfort_item": "Gấu Pooh"
    }
    response = client.post("/api/v1/telemetry/evaluate-frustration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "MODERATE_FRUSTRATION"
    assert 40.0 <= data["frustration_score"] < 75.0
    assert data["calm_mode_recommended"] is False
    assert "Gấu Pooh" in data["comfort_item_advice"]

def test_high_meltdown_risk_triggers_calm_mode():
    """Kiểm tra kịch bản bùng nổ khủng hoảng (Rage tap + error spike + freeze) kích hoạt Calm Mode"""
    payload = {
        "child_id": "child-001",
        "child_name": "Bé Nam",
        "window_seconds": 30,
        "tap_events_count": 180,  # 6 taps/sec -> Rage tap cực mạnh
        "consecutive_fails": 5,    # 5 lần sai liên tiếp
        "idle_seconds": 22,        # Freeze 22s
        "erratic_touch_count": 9,
        "average_latency_ms": 150,
        "comfort_item": "Chiếc xe ô tô đỏ"
    }
    response = client.post("/api/v1/telemetry/evaluate-frustration", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "HIGH_MELTDOWN_RISK"
    assert data["frustration_score"] >= 75.0
    assert data["calm_mode_recommended"] is True
    assert data["calm_mode_action"] == "TRIGGER_CALM_MODE"
    assert data["breathing_pattern"] == "4_4_4_BOX_BREATHING"
    assert "Chiếc xe ô tô đỏ" in data["de_escalation_message"]

def test_negative_empty_payload():
    """Rule 3: Kiểm thử Negative với payload rỗng {}"""
    response = client.post("/api/v1/telemetry/evaluate-frustration", json={})
    assert response.status_code == 422

def test_negative_empty_child_id():
    """Rule 3: Kiểm thử Negative với child_id rỗng"""
    payload = {
        "child_id": "   ",
        "child_name": "Bé Nam"
    }
    response = client.post("/api/v1/telemetry/evaluate-frustration", json=payload)
    assert response.status_code == 400
    assert "child_id cannot be empty" in response.json()["detail"]

def test_boundary_negative_values():
    """Rule 3: Kiểm thử Boundary với giá trị âm không hợp lệ"""
    payload = {
        "child_id": "child-001",
        "window_seconds": -10,  # Không được phép <= 0
        "tap_events_count": -5
    }
    response = client.post("/api/v1/telemetry/evaluate-frustration", json=payload)
    assert response.status_code == 422
