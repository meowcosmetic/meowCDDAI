"""
Social Story Generation Service - Dual-Tier System (Carol Gray Standard + Active Desensitization Protocol)
Cung cấp hệ thống can thiệp 2 cấp độ:
- Cấp độ 1: Truyện tranh xã hội thụ động (Visual Story Guide) chuẩn Carol Gray Formula Ratio >= 2.0
- Cấp độ 2: Cẩm nang thực chiến giảm mẫn cảm 5 bước (Active Desensitization Action Protocol)
"""

from typing import Dict, Any, List, Optional


class SocialStoryService:
    """Service tạo và thẩm định hệ thống can thiệp xã hội 2 cấp độ."""

    TEMPLATES: Dict[str, Dict[str, Any]] = {
        "DENTAL_VISIT": {
            "title": "Đi Khám Răng Thân Thiện",
            "icon": "medical_services",
            "category": "DENTAL_VISIT",
            "description": "Làm quen với phòng khám nha khoa, vượt qua nỗi sợ tiếng máy mài và há miệng hợp tác.",
            "target_behavior": "Bé ngồi yên trên ghế và há miệng cho bác sĩ đếm răng",
            "target_issue": "Bé sợ đau và sợ người lạ đưa dụng cụ kim loại vào miệng",
            "specific_trigger": "Tiếng máy mài răng kêu 'ro ro' và ánh đèn rọi sáng",
            "sensory_reason": "Thính giác và thị giác của bé rất nhạy cảm với âm thanh tần số cao và ánh sáng mạnh",
            "coping_skill": "Giơ bàn tay làm dấu 'Cho con nghỉ 5 giây' và cầm chặt đồ chơi quen thuộc hít thở sâu",
            "timer_minutes": 10,
        },
        "VACCINATION": {
            "title": "Bé Dũng Cảm Đi Tiêm Phòng",
            "icon": "vaccines",
            "category": "VACCINATION",
            "description": "Hiểu rằng thuốc vắc-xin giúp cơ thể khỏe mạnh, vượt qua nỗi sợ kim tiêm.",
            "target_behavior": "Bé ngồi trong lòng mẹ, thả lỏng cánh tay và hít thở đều",
            "target_issue": "Bé hoảng loạn khi nhìn thấy mũi kim tiêm và sợ cảm giác đau buốt",
            "specific_trigger": "Nhìn thấy y tá cầm kim tiêm và mùi cồn sát khuẩn",
            "sensory_reason": "Bé lo sợ cảm giác đau kéo dài và chưa hình dung được vắc-xin bảo vệ cơ thể",
            "coping_skill": "Tựa đầu vào vai mẹ, nhắm mắt đếm từ 1 đến 3 và cùng bạn đồng hành thổi bong bóng",
            "timer_minutes": 5,
        },
        "HAIRCUT": {
            "title": "Bé Đi Cắt Tóc Gọn Gàng",
            "icon": "content_cut",
            "category": "HAIRCUT",
            "description": "Chiếc áo choàng kỳ diệu và vượt qua cảm giác sợ kéo sát tai.",
            "target_behavior": "Bé ngồi yên trên ghế xoay để bác thợ cắt tóc",
            "target_issue": "Bé sợ người lạ chạm vào đầu và sợ bị mất tóc",
            "specific_trigger": "Cây kéo đưa sát tai phát tiếng 'lách cách' và tóc vụn rơi vào cổ",
            "sensory_reason": "Vùng da đầu và cổ của bé rất nhạy cảm xúc giác, bé lo cắt tóc là bị thương",
            "coping_skill": "Đeo tai nghe nhạc êm ái, ngồi yên và dùng chổi lông mềm phủi tóc vụn",
            "timer_minutes": 12,
        },
        "SHARING_TOYS": {
            "title": "Cùng Nhau Chơi Đồ Chơi Thật Vui",
            "icon": "toys",
            "category": "SHARING_TOYS",
            "description": "Học cách luân phiên và kiểm soát phản ứng khi bạn chạm vào đồ chơi.",
            "target_behavior": "Con chơi một lúc rồi đổi đồ chơi cho bạn mượn",
            "target_issue": "Bé nổi giận, la hét hoặc đánh bạn khi bị bạn lấy đồ chơi",
            "specific_trigger": "Khoảnh khắc bạn tiến lại gần và cầm vào món đồ bé đang chơi",
            "sensory_reason": "Bé nghĩ món đồ bị mất vĩnh viễn và chưa hiểu khái niệm chia sẻ theo lượt",
            "coping_skill": "Dùng đồng hồ cát tính giờ, nói câu: 'Bạn chơi một lúc rồi đổi cho mình nhé'",
            "timer_minutes": 5,
        },
        "SUPERMARKET_CROWD": {
            "title": "Bé Cùng Mẹ Đi Siêu Thị Đông Vui",
            "icon": "shopping_cart",
            "category": "SUPERMARKET_CROWD",
            "description": "Làm quen với nơi đông người, âm thanh và hàng quán rực rỡ.",
            "target_behavior": "Con nắm chặt tay mẹ không chạy một mình",
            "target_issue": "Bé bị choáng ngợp giác quan, hoảng loạn muốn bỏ chạy hoặc khóc thét",
            "specific_trigger": "Tiếng loa phát thanh to bất ngờ và đám đông chen chúc xung quanh",
            "sensory_reason": "Quá tải thính giác và thị giác do có quá nhiều kích thích không gian cùng lúc",
            "coping_skill": "Nắm chặt tay mẹ, đeo tai nghe chống ồn hoặc ôm chặt gấu bông quen thuộc",
            "timer_minutes": 15,
        },
        "NEW_SCHOOL": {
            "title": "Ngày Đầu Tiên Đi Lớp Mới",
            "icon": "school",
            "category": "NEW_SCHOOL",
            "description": "Làm quen với trường mới, giải tỏa nỗi lo âu chia ly khi xa ba mẹ.",
            "target_behavior": "Con vào lớp chào cô và đợi mẹ đón vào buổi chiều",
            "target_issue": "Bé sợ bị bỏ rơi và khóc to không chịu buông tay mẹ vào lớp",
            "specific_trigger": "Khoảnh khắc mẹ quay bước đi sau khi đưa bé đến cửa lớp",
            "sensory_reason": "Bé chưa có khái niệm thời gian rõ ràng và chưa hiểu quy luật 'buổi chiều mẹ sẽ đón'",
            "coping_skill": "Đeo vòng tay hẹn giờ có ảnh gia đình, chào cô và tham gia góc đồ chơi yêu thích",
            "timer_minutes": 20,
        },
        "ANGER_REGULATION": {
            "title": "Làm Dịu Cơn Giận Trong Con",
            "icon": "sentiment_satisfied_alt",
            "category": "ANGER_REGULATION",
            "description": "Nhận biết khi tức giận và học kỹ năng điều hòa an toàn không làm đau bạn.",
            "target_behavior": "Con khoanh tay hít thở sâu không đánh bạn",
            "target_issue": "Bé cắn, đánh hoặc đập đầu khi không vừa ý",
            "specific_trigger": "Khi bị từ chối điều bé muốn hoặc khi bị gián đoạn hoạt động yêu thích",
            "sensory_reason": "Kém kỹ năng diễn đạt cảm xúc bằng lời nên giải tỏa căng thẳng qua xung động cơ thể",
            "coping_skill": "Khoanh hai tay ôm chặt ngực, hít thở 3 nhịp 'Hít hoa - Thổi nến' hoặc bóp bóng xốp",
            "timer_minutes": 5,
        },
        "WAITING_TURN": {
            "title": "Bé Kiên Nhẫn Chờ Đến Lượt",
            "icon": "hourglass_top",
            "category": "WAITING_TURN",
            "description": "Hiểu về quy luật xếp hàng và luân phiên thứ tự một cách bình tĩnh.",
            "target_behavior": "Con xếp hàng kiên nhẫn chờ đến lượt mình",
            "target_issue": "Bé chen lấn, đẩy bạn phía trước hoặc la hét vì không được chơi ngay",
            "specific_trigger": "Bắt buộc phải đứng yên trong hàng nhìn người khác đang chơi",
            "sensory_reason": "Khó khăn trong việc ức chế xung động tức thời và cảm giác chờ đợi gây khó chịu",
            "coping_skill": "Đếm bước chân tại chỗ, hát thầm bài hát yêu thích và vỗ tay nhịp nhàng",
            "timer_minutes": 5,
        },
    }

    @classmethod
    def get_templates(cls) -> List[Dict[str, Any]]:
        """Trả về danh sách 8 template tình huống chuẩn với 4 trụ cột can thiệp."""
        return list(cls.TEMPLATES.values())

    @classmethod
    def evaluate_carol_gray_formula(cls, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Thẩm định công thức tỷ lệ Carol Gray (1991):
        Ratio = (Descriptive + Perspective + Affirmative) / Directive >= 2.0
        """
        counts = {
            "DESCRIPTIVE": 0,
            "PERSPECTIVE": 0,
            "DIRECTIVE": 0,
            "AFFIRMATIVE": 0,
        }
        for page in pages:
            stype = page.get("sentence_type", "DESCRIPTIVE").upper()
            if stype in counts:
                counts[stype] += 1
            else:
                counts["DESCRIPTIVE"] += 1

        directive_count = counts["DIRECTIVE"]
        non_directive_count = counts["DESCRIPTIVE"] + counts["PERSPECTIVE"] + counts["AFFIRMATIVE"]

        if directive_count == 0:
            ratio = float(non_directive_count) if non_directive_count > 0 else 1.0
        else:
            ratio = round(non_directive_count / directive_count, 2)

        is_valid = ratio >= 2.0

        return {
            "counts": counts,
            "directive_count": directive_count,
            "non_directive_count": non_directive_count,
            "carol_gray_ratio": ratio,
            "is_pedagogically_valid": is_valid,
            "standard": "Carol Gray Social Story Ratio Standard (>= 2.0)",
        }

    @classmethod
    def generate_social_story(cls, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sinh hệ thống can thiệp xã hội 2 cấp độ:
        - Cấp độ 1: 5 trang sách tranh chuẩn Carol Gray (Ratio >= 2.0)
        - Cấp độ 2: Cẩm nang thực chiến giảm mẫn cảm 5 bước kèm Visual Timer
        """
        child_name = request_data.get("child_name", "Bé")
        if not child_name or not child_name.strip():
            child_name = "Bé"

        child_age = request_data.get("child_age", 5)
        situation_cat = request_data.get("situation_category", "DENTAL_VISIT").upper()
        raw_context = request_data.get("custom_context") or ""
        custom_context = raw_context.strip() if isinstance(raw_context, str) else ""

        interests = request_data.get("interests", []) or []
        comfort_item = request_data.get("comfort_item", "") or ""
        triggers_to_avoid = request_data.get("triggers_to_avoid", []) or []

        # Bạn đồng hành
        companion = "Bạn Gấu Nhỏ"
        if interests:
            companion = f"Bạn {interests[0]}"
        elif request_data.get("favorite_characters"):
            companion = f"Bạn {request_data['favorite_characters'][0]}"

        # Vật an ủi
        comfort_phrase = "vật an ủi quen thuộc"
        if comfort_item and comfort_item.strip():
            comfort_phrase = comfort_item.strip()
        elif "xe" in str(interests).lower() or "ô tô" in str(interests).lower():
            comfort_phrase = "chiếc ô tô đồ chơi quen thuộc"
        elif "khủng long" in str(interests).lower():
            comfort_phrase = "chú khủng long nhỏ trong tay"

        template = cls.TEMPLATES.get(situation_cat)

        # 4 Trụ Cột Can Thiệp Lâm Sàng
        target_issue = request_data.get("target_issue") or (template["target_issue"] if template else f"{child_name} cảm thấy bối rối trước tình huống mới")
        specific_trigger = request_data.get("specific_trigger") or (template["specific_trigger"] if template else (custom_context if custom_context else "Khi đối mặt với sự thay đổi môi trường đột ngột"))
        sensory_reason = request_data.get("sensory_reason") or (template["sensory_reason"] if template else f"Giác quan của {child_name} rất nhạy cảm và cần thời gian thích nghi")
        coping_skill = request_data.get("coping_skill") or (template["coping_skill"] if template else f"Nắm chặt tay mẹ, ôm {comfort_phrase} và cùng {companion} hít thở sâu 3 nhịp")

        # CẤP ĐỘ 1: Xây dựng các trang truyện chuẩn Carol Gray
        pages = cls._build_four_pillar_pages(
            name=child_name,
            companion=companion,
            comfort=comfort_phrase,
            issue=target_issue,
            trigger=specific_trigger,
            reason=sensory_reason,
            skill=coping_skill,
            category=situation_cat,
        )

        # CẤP ĐỘ 2: Xây dựng cẩm nang thực chiến giảm mẫn cảm 5 bước
        timer_minutes = template.get("timer_minutes", 10) if template else 10
        action_protocol = cls._build_action_protocol(
            cat=situation_cat,
            name=child_name,
            companion=companion,
            comfort=comfort_phrase,
            trigger=specific_trigger,
            skill=coping_skill,
            minutes=timer_minutes,
        )

        # Câu hỏi thấu cảm xoay quanh Coping Skill
        comp_q = {
            "question": f"Khi gặp tình huống ({specific_trigger}), {child_name} sẽ dùng kỹ năng dũng cảm nào?",
            "options": [
                f"{coping_skill}",
                "La hét, cắn hoặc bỏ chạy mất",
            ],
            "correct_option_index": 0,
            "explanation": f"Chính xác! Áp dụng kỹ năng ({coping_skill}) giúp {child_name} luôn làm chủ tình huống và an toàn!",
        }

        # Đánh giá tỷ lệ Carol Gray
        formula_eval = cls.evaluate_carol_gray_formula(pages)
        title = f"{companion} Cùng {child_name}: {template['title'] if template else 'Kỹ Năng Dũng Cảm'}"

        return {
            "child_name": child_name,
            "child_age": child_age,
            "situation_category": situation_cat,
            "title": title,
            "target_issue": target_issue,
            "specific_trigger": specific_trigger,
            "sensory_reason": sensory_reason,
            "coping_skill": coping_skill,
            "companion_character": companion,
            "comfort_item_used": comfort_phrase,
            "carol_gray_ratio": formula_eval["carol_gray_ratio"],
            "is_pedagogically_valid": formula_eval["is_pedagogically_valid"],
            "formula_metrics": formula_eval,
            "pages": pages,
            "action_protocol": action_protocol,
            "comprehension_question": comp_q,
        }

    @classmethod
    def _build_action_protocol(
        cls, cat: str, name: str, companion: str, comfort: str, trigger: str, skill: str, minutes: int
    ) -> Dict[str, Any]:
        """
        Xây dựng Cẩm Nang Thực Chiến Giảm Mẫn Cảm 5 Bước (Desensitization Protocol):
        1. Đếm ngược (Countdown)
        2. Khảo sát không gian thực tế (Field Visit)
        3. Nghe thử âm thanh & Video mô phỏng (Sensory Preview)
        4. Giới hạn thời gian với Đồng hồ trực quan (Finite Visual Timer)
        5. Trấn an & Trao phần thưởng yêu thích từ Harness (Reward Protocol)
        """
        location_desc = "tiệm cắt tóc" if "HAIR" in cat else ("phòng khám nha khoa" if "DENTAL" in cat else ("phòng tiêm chủng" if "VACCIN" in cat else "địa điểm diễn ra"))
        sound_desc = "tiếng kéo lách cách và tông đơ" if "HAIR" in cat else ("tiếng máy mài 'ro ro'" if "DENTAL" in cat else ("âm thanh sinh hoạt phòng khám" if "VACCIN" in cat else "âm thanh môi trường"))

        steps = [
            {
                "step_number": 1,
                "title": "Thông báo & Đếm ngược ngày (Countdown)",
                "guide": f"Thông báo trước 3 ngày rằng {name} sẽ đến {location_desc}. Nhắc con mỗi sáng khi thức dậy: 'Còn 3 ngày... còn 2 ngày... ngày mai mình sẽ đi nhé con!'.",
                "tag": "PREDICTABILITY",
                "is_completed": False,
            },
            {
                "step_number": 2,
                "title": f"Khảo sát không gian {location_desc} (Field Familiarization)",
                "guide": f"Dắt {name} đi dạo ngang qua {location_desc} trước 1 ngày. Cho con đứng nhìn biển hiệu, nhìn chiếc ghế ngồi từ xa và nói: 'Đây là nơi ngày mai {companion} sẽ đồng hành cùng con'.",
                "tag": "SPATIAL_AWARENESS",
                "is_completed": False,
            },
            {
                "step_number": 3,
                "title": f"Làm quen âm thanh & Xem video mô phỏng (Sensory Preview)",
                "guide": f"Bật thử file âm thanh {sound_desc} ở mức âm lượng nhỏ trong điện thoại cho {name} nghe trước ở nhà. Cho con xem video ngắn mô phỏng để não bộ làm quen với trigger ({trigger}).",
                "tag": "DESENSITIZATION",
                "is_completed": False,
            },
            {
                "step_number": 4,
                "title": f"Đặt giới hạn thời gian {minutes} phút (Visual Sensory Timer)",
                "guide": f"Thông báo rõ cho {name}: 'Con chỉ cần ngồi trên ghế đúng {minutes} phút là xong'. Bật Đồng Hồ Đếm Giờ Trực Quan trên app để con thấy kim đồng hồ tiến về đích từng giây.",
                "tag": "VISUAL_TIMER",
                "is_completed": False,
            },
            {
                "step_number": 5,
                "title": f"Trấn an & Trao thưởng ({comfort}) (Positive Reinforcement)",
                "guide": f"Ngay khi bước xuống ghế, ôm chặt {name} vào lòng, khen con là em bé dũng cảm và trao ngay phần thưởng con yêu thích ({comfort}) để củng cố hành vi tích cực!",
                "tag": "REWARD_PROTOCOL",
                "is_completed": False,
            },
        ]

        return {
            "protocol_name": f"Kế Hoạch Can Thiệp Thực Chiến 5 Bước Cho {name}",
            "timer_duration_minutes": minutes,
            "steps": steps,
        }

    @classmethod
    def _build_four_pillar_pages(
        cls,
        name: str,
        companion: str,
        comfort: str,
        issue: str,
        trigger: str,
        reason: str,
        skill: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """Xây dựng 5 trang sách tranh thụ động chuẩn Carol Gray (Ratio >= 2.0)."""
        cat = category.upper()
        scene_1 = "SCENE_DENTAL_CHAIR" if "DENTAL" in cat else ("SCENE_VACCINE_CLINIC" if "VACCIN" in cat else ("SCENE_BARBER_SHOP" if "HAIR" in cat else "SCENE_SITUATION_START"))
        scene_2 = "SCENE_DRILL_SOUND_TRIGGER" if "DENTAL" in cat else ("SCENE_NEEDLE_TRIGGER" if "VACCIN" in cat else ("SCENE_SCISSORS_TRIGGER" if "HAIR" in cat else "SCENE_TRIGGER_MOMENT"))
        scene_3 = "SCENE_DOCTOR_SMILE" if "DENTAL" in cat else ("SCENE_NURSE_CARE" if "VACCIN" in cat else ("SCENE_BARBER_SMILE" if "HAIR" in cat else "SCENE_KIND_SURROUNDING"))
        scene_4 = "SCENE_DEEP_BREATHE"

        return [
            {
                "page_number": 1,
                "sentence_type": "DESCRIPTIVE",
                "text": f"Đôi khi {name} có thể cảm thấy lo lắng hoặc không thoải mái vì {issue.lower()}.",
                "scene_tag": scene_1,
                "illustration_prompt": f"Gentle illustration acknowledging child feelings of {issue}, pastel colors",
                "parent_coaching_tip": f"Ba mẹ hãy thừa nhận cảm xúc của con: 'Mẹ biết con thấy lo, và cảm thấy thế là hoàn toàn bình thường con nhé'.",
            },
            {
                "page_number": 2,
                "sentence_type": "DESCRIPTIVE",
                "text": f"Khoảnh khắc kích hoạt sự lo lắng là khi: {trigger}.",
                "scene_tag": scene_2,
                "illustration_prompt": f"Clear illustration of the trigger moment: {trigger}, 2D pastel art",
                "parent_coaching_tip": f"Nhắc con nhận biết đúng khoảnh khắc này để chuẩn bị sẵn sàng tâm lý, không bị bất ngờ.",
            },
            {
                "page_number": 3,
                "sentence_type": "PERSPECTIVE",
                "text": f"{reason}. Những người xung quanh luôn yêu thương và mong muốn điều an toàn nhất cho {name}.",
                "scene_tag": scene_3,
                "illustration_prompt": f"Empathetic explanation of sensory sensitivity and warm friendly helpers, pastel",
                "parent_coaching_tip": f"Giải thích cho con hiểu vì sao cơ thể con phản ứng như vậy và người lớn luôn ở đây bảo vệ con.",
            },
            {
                "page_number": 4,
                "sentence_type": "DIRECTIVE",
                "text": f"Khi tình huống đó diễn ra, {name} có thể dùng kỹ năng dũng cảm: {skill}.",
                "scene_tag": scene_4,
                "illustration_prompt": f"Child successfully performing coping skill {skill} with cheerful buddy, pastel",
                "parent_coaching_tip": f"Luyện tập kỹ năng này cùng con 3 lần ngay tại nhà trước khi sự kiện diễn ra để tạo phản xạ tự nhiên.",
            },
            {
                "page_number": 5,
                "sentence_type": "AFFIRMATIVE",
                "text": f"Biết cách sử dụng kỹ năng này, {name} và {companion} sẽ luôn an toàn và vượt qua thật êm đẹp. Con là em bé vô cùng dũng cảm!",
                "scene_tag": "SCENE_BRAVE_STAR_REWARD",
                "illustration_prompt": f"Child celebrating proud victory with a golden medal of bravery, pastel style",
                "parent_coaching_tip": f"Sau khi con làm được, hãy ôm con thật chặt và tặng con phần thưởng khen ngợi xứng đáng.",
            },
        ]
