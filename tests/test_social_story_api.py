import pytest
from fastapi.testclient import TestClient
from main import app
from services.social_story_service import SocialStoryService

client = TestClient(app)


def test_get_templates():
    """Kiểm tra lấy danh sách 8 tình huống can thiệp mẫu."""
    response = client.get("/api/v1/stories/templates")
    assert response.status_code == 200
    templates = response.json()
    assert isinstance(templates, list)
    assert len(templates) == 8

    categories = [t["category"] for t in templates]
    assert "DENTAL_VISIT" in categories
    assert "VACCINATION" in categories
    assert "HAIRCUT" in categories
    assert "SHARING_TOYS" in categories
    assert "SUPERMARKET_CROWD" in categories
    assert "NEW_SCHOOL" in categories
    assert "ANGER_REGULATION" in categories
    assert "WAITING_TURN" in categories

    for t in templates:
        assert "title" in t
        assert "description" in t
        assert "target_behavior" in t


def test_generate_story_dental_visit():
    """Kiểm tra sinh câu chuyện nha sĩ kết hợp sở thích và chuẩn Carol Gray."""
    payload = {
        "child_name": "Nam",
        "child_age": 5,
        "situation_category": "DENTAL_VISIT",
        "interests": ["Khủng Long T-Rex"],
        "comfort_item": "Chiếc xe ô tô đỏ",
        "triggers_to_avoid": ["tiếng chuông to"],
    }
    response = client.post("/api/v1/stories/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["child_name"] == "Nam"
    assert data["situation_category"] == "DENTAL_VISIT"
    assert "Khủng Long" in data["companion_character"]
    assert "ô tô đỏ" in data["comfort_item_used"].lower()

    # Kiểm tra 4 trụ cột can thiệp lâm sàng
    assert "target_issue" in data and len(data["target_issue"]) > 5
    assert "specific_trigger" in data and len(data["specific_trigger"]) > 5
    assert "sensory_reason" in data and len(data["sensory_reason"]) > 5
    assert "coping_skill" in data and len(data["coping_skill"]) > 5

    # Kiểm tra tỷ lệ Carol Gray
    assert data["carol_gray_ratio"] >= 2.0
    assert data["is_pedagogically_valid"] is True

    # Kiểm tra 5 trang truyện
    pages = data["pages"]
    assert len(pages) == 5
    sentence_types = [p["sentence_type"] for p in pages]
    assert "DESCRIPTIVE" in sentence_types
    assert "PERSPECTIVE" in sentence_types
    assert "DIRECTIVE" in sentence_types
    assert "AFFIRMATIVE" in sentence_types

    for p in pages:
        assert "scene_tag" in p
        assert p["scene_tag"].startswith("SCENE_")
        assert "parent_coaching_tip" in p
        assert len(p["parent_coaching_tip"]) > 10

    # Kiểm tra Cấp độ 2: Cẩm nang thực chiến giảm mẫn cảm 5 bước
    assert "action_protocol" in data
    protocol = data["action_protocol"]
    assert "timer_duration_minutes" in protocol
    assert len(protocol["steps"]) == 5
    for s in protocol["steps"]:
        assert "step_number" in s
        assert "title" in s
        assert "guide" in s
        assert "is_completed" in s

    # Kiểm tra câu hỏi thấu cảm
    comp_q = data["comprehension_question"]
    assert "question" in comp_q
    assert len(comp_q["options"]) == 2
    assert comp_q["correct_option_index"] == 0
    assert "explanation" in comp_q


def test_generate_story_custom_situation():
    """Kiểm tra sinh câu chuyện tình huống riêng biệt (CUSTOM)."""
    payload = {
        "child_name": "Bé Bi",
        "child_age": 6,
        "situation_category": "CUSTOM",
        "custom_context": "Bé chuẩn bị chuyển sang căn nhà mới ở thành phố khác",
        "interests": ["Siêu nhân Gao"],
        "comfort_item": "Chiếc gối ôm màu xanh",
    }
    response = client.post("/api/v1/stories/generate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["situation_category"] == "CUSTOM"
    assert data["carol_gray_ratio"] >= 2.0
    assert data["is_pedagogically_valid"] is True
    assert "Siêu nhân" in data["companion_character"]
    assert len(data["pages"]) == 5


def test_evaluate_carol_gray_formula():
    """Kiểm tra thẩm định công thức tỷ lệ Carol Gray."""
    # 3 Non-directive (2 Descriptive, 1 Perspective) + 1 Directive => Ratio = 3.0 (Valid)
    valid_pages = [
        {"sentence_type": "DESCRIPTIVE"},
        {"sentence_type": "DESCRIPTIVE"},
        {"sentence_type": "PERSPECTIVE"},
        {"sentence_type": "DIRECTIVE"},
    ]
    eval_res = SocialStoryService.evaluate_carol_gray_formula(valid_pages)
    assert eval_res["carol_gray_ratio"] == 3.0
    assert eval_res["is_pedagogically_valid"] is True

    # 1 Descriptive + 2 Directive => Ratio = 0.5 (Invalid, quá nhiều lệnh áp đặt)
    invalid_pages = [
        {"sentence_type": "DESCRIPTIVE"},
        {"sentence_type": "DIRECTIVE"},
        {"sentence_type": "DIRECTIVE"},
    ]
    eval_res2 = SocialStoryService.evaluate_carol_gray_formula(invalid_pages)
    assert eval_res2["carol_gray_ratio"] == 0.5
    assert eval_res2["is_pedagogically_valid"] is False


def test_generate_story_empty_situation_error():
    """Negative test: Kiểm tra từ chối khi situation_category rỗng."""
    payload = {
        "child_name": "Nam",
        "situation_category": "",
    }
    response = client.post("/api/v1/stories/generate", json=payload)
    assert response.status_code == 400
