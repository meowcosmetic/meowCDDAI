import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChildHarnessProfile(BaseModel):
    child_id: str
    child_name: str = "Bé"
    age: Optional[int] = None
    gender: Optional[str] = None
    diagnosis: Optional[str] = "Phát triển ngôn ngữ"
    dominant_interests: List[str] = Field(default_factory=list)
    triggers_to_avoid: List[str] = Field(default_factory=list)
    mastered_skills: List[str] = Field(default_factory=list)
    target_skills: List[str] = Field(default_factory=list)
    prompt_injection_snippet: str = ""
    memory_notes: List[Dict[str, Any]] = Field(default_factory=list)
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChildHarnessService:
    """
    Dịch vụ quản lý lớp vỏ trí nhớ cá nhân hóa dài hạn (Child Memory Harness) cho từng bé.
    Cung cấp:
    - 4 tầng trí nhớ (Tĩnh, Sở thích, Yếu tố nhạy cảm, Kỹ năng).
    - Bộ sinh Prompt Injection Snippet tự động.
    - Cơ chế đồng bộ từ bài test sàng lọc (ASQ-3/CDD/M-CHAT).
    """

    def __init__(self):
        # Lưu trữ in-memory cache cho profiles
        self._profiles: Dict[str, ChildHarnessProfile] = {}
        # Khởi tạo một số dữ liệu mẫu cho demo
        self._init_demo_profiles()

    def _init_demo_profiles(self):
        sample = ChildHarnessProfile(
            child_id="demo_child_01",
            child_name="Bé Minh Nam",
            age=5,
            gender="Nam",
            diagnosis="Chậm nói kèm rối loạn phổ tự kỷ nhẹ",
            dominant_interests=["khủng long T-Rex", "màu xanh lá cây", "xe cứu hỏa", "bà ngoại"],
            triggers_to_avoid=["tiếng chuông to đột ngột > 70dB", "giao diện nhấp nháy đỏ chói", "ép buộc nhìn thẳng vào mắt"],
            mastered_skills=["nhận biết màu sắc cơ bản", "chào ba mẹ", "phát âm âm đơn"],
            target_skills=["luyện thanh điệu (dấu hỏi, ngã, nặng)", "sửa ngọng phụ âm l/n", "diễn đạt nhu cầu tự lập"],
            memory_notes=[
                {"type": "interest", "text": "Bé rất thích mô hình khủng long bạo chúa T-Rex", "created_at": datetime.now().isoformat()},
                {"type": "trigger", "text": "Bé sợ tiếng còi xe cấp cứu quá lớn", "created_at": datetime.now().isoformat()}
            ]
        )
        sample.prompt_injection_snippet = self.generate_prompt_snippet(sample)
        self._profiles[sample.child_id] = sample

    def get_or_create_profile(
        self,
        child_id: str,
        child_name: Optional[str] = None,
        age: Optional[int] = None,
        diagnosis: Optional[str] = None,
    ) -> ChildHarnessProfile:
        clean_id = child_id.strip()
        if clean_id in self._profiles:
            profile = self._profiles[clean_id]
            if child_name and child_name.strip():
                profile.child_name = child_name.strip()
            if age is not None and age > 0:
                profile.age = age
            if diagnosis and diagnosis.strip():
                profile.diagnosis = diagnosis.strip()
            profile.prompt_injection_snippet = self.generate_prompt_snippet(profile)
            profile.updated_at = datetime.now().isoformat()
            return profile

        new_profile = ChildHarnessProfile(
            child_id=clean_id,
            child_name=child_name.strip() if child_name else "Bé",
            age=age,
            diagnosis=diagnosis.strip() if diagnosis else "Phát triển ngôn ngữ",
        )
        new_profile.prompt_injection_snippet = self.generate_prompt_snippet(new_profile)
        self._profiles[clean_id] = new_profile
        return new_profile

    def add_memory(
        self,
        child_id: str,
        memory_type: str,
        content: str,
    ) -> ChildHarnessProfile:
        """
        Thêm một mẩu trí nhớ (sở thích, trigger hoặc ghi chú) vào hồ sơ bé.
        """
        if not content or not content.strip():
            raise ValueError("Nội dung ghi chú không được để trống")

        profile = self.get_or_create_profile(child_id)
        clean_content = content.strip()
        m_type = memory_type.lower().strip()

        if m_type in ["interest", "so_thich"]:
            if clean_content.lower() not in [i.lower() for i in profile.dominant_interests]:
                profile.dominant_interests.append(clean_content)
        elif m_type in ["trigger", "nhay_cam", "yeu_to_can_tranh"]:
            if clean_content.lower() not in [t.lower() for t in profile.triggers_to_avoid]:
                profile.triggers_to_avoid.append(clean_content)
        elif m_type in ["target_skill", "ky_nang_muc_tieu"]:
            if clean_content.lower() not in [s.lower() for s in profile.target_skills]:
                profile.target_skills.append(clean_content)
        elif m_type in ["mastered_skill", "ky_nang_da_dat"]:
            if clean_content.lower() not in [s.lower() for s in profile.mastered_skills]:
                profile.mastered_skills.append(clean_content)

        profile.memory_notes.append({
            "type": m_type,
            "text": clean_content,
            "created_at": datetime.now().isoformat(),
        })

        profile.prompt_injection_snippet = self.generate_prompt_snippet(profile)
        profile.updated_at = datetime.now().isoformat()
        return profile

    def sync_test_result(
        self,
        child_id: str,
        test_name: str,
        category_scores: List[Dict[str, Any]],
        sensory_triggers: Optional[List[str]] = None,
    ) -> ChildHarnessProfile:
        """
        Trigger tự động đồng bộ kết quả bài test (ASQ-3, CDD Test) vào Child Memory Harness.
        - Lĩnh vực FAIL/RED/BLACK -> nạp vào target_skills.
        - Lĩnh vực PASS/WHITE -> nạp vào mastered_skills.
        - Sensory triggers -> nạp vào triggers_to_avoid.
        """
        profile = self.get_or_create_profile(child_id)

        for cat in category_scores:
            cat_name = cat.get("categoryName") or cat.get("name") or "Kỹ năng"
            status = str(cat.get("status", "")).upper()
            zone = str(cat.get("zone", "")).upper()

            is_intervention = status in ["FAIL", "RED", "INTERVENTION"] or zone in ["BLACK", "GRAY"]
            is_mastered = status in ["PASS", "WHITE", "SUCCESS"] or zone in ["WHITE"]

            if is_intervention:
                skill_desc = f"{cat_name} (cần can thiệp theo test {test_name})"
                if not any(cat_name.lower() in s.lower() for s in profile.target_skills):
                    profile.target_skills.append(skill_desc)
            elif is_mastered:
                skill_desc = f"{cat_name} (đạt chuẩn trong test {test_name})"
                if not any(cat_name.lower() in s.lower() for s in profile.mastered_skills):
                    profile.mastered_skills.append(skill_desc)

        if sensory_triggers:
            for trigger in sensory_triggers:
                if trigger and trigger.strip():
                    t_clean = trigger.strip()
                    if not any(t_clean.lower() in s.lower() for s in profile.triggers_to_avoid):
                        profile.triggers_to_avoid.append(t_clean)

        profile.memory_notes.append({
            "type": "test_sync",
            "text": f"Đồng bộ kết quả bài test {test_name}",
            "created_at": datetime.now().isoformat(),
        })

        profile.prompt_injection_snippet = self.generate_prompt_snippet(profile)
        profile.updated_at = datetime.now().isoformat()
        return profile

    def generate_prompt_snippet(self, profile: ChildHarnessProfile) -> str:
        """
        Sinh ra đoạn System Prompt Context súc tích để tiêm vào bất kỳ AI Agent nào.
        """
        parts = [f"NGỮ CẢNH HỒ SƠ BÉ: {profile.child_name}"]
        if profile.age:
            parts.append(f"({profile.age} tuổi)")
        if profile.diagnosis:
            parts.append(f"- Chẩn đoán: {profile.diagnosis}.")

        if profile.dominant_interests:
            interests_str = ", ".join(profile.dominant_interests[:5])
            parts.append(f"Sở thích đặc biệt: {interests_str} (hãy ưu tiên lồng ghép hình ảnh/ví dụ này để tạo động lực và sự tập trung).")

        if profile.triggers_to_avoid:
            triggers_str = ", ".join(profile.triggers_to_avoid[:4])
            parts.append(f"CẢNH BÁO YẾU TỐ NHẠY CẢM CẦN TRÁNH: {triggers_str} (tuyệt đối không dùng để tránh gây kích động/khủng hoảng cho bé).")

        if profile.target_skills:
            targets_str = ", ".join(profile.target_skills[:3])
            parts.append(f"Mục tiêu rèn luyện trọng tâm: {targets_str}.")

        parts.append("Hãy giao tiếp bằng giọng điệu ấm áp, kiên nhẫn, câu ngắn gọn và khuyến khích tích cực.")
        return " ".join(parts)

    def get_context(self, child_id: str) -> Dict[str, Any]:
        """
        Lấy context đầy đủ của bé để các module khác gọi dùng.
        """
        profile = self.get_or_create_profile(child_id)
        return {
            "child_id": profile.child_id,
            "child_name": profile.child_name,
            "age": profile.age,
            "diagnosis": profile.diagnosis,
            "dominant_interests": profile.dominant_interests,
            "triggers_to_avoid": profile.triggers_to_avoid,
            "mastered_skills": profile.mastered_skills,
            "target_skills": profile.target_skills,
            "prompt_injection_snippet": profile.prompt_injection_snippet,
            "memory_notes_count": len(profile.memory_notes),
            "updated_at": profile.updated_at,
        }


# Singleton instance
child_harness_service = ChildHarnessService()
