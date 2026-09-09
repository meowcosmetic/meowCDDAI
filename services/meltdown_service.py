"""
Service phân tích nhịp điệu tương tác cảm ứng (Touch Telemetry Dynamics)
và cảnh báo sớm nguy cơ bùng nổ khủng hoảng hành vi (Early Meltdown Warning System).
"""

from typing import Dict, Any, Optional

def evaluate_frustration_telemetry(
    window_seconds: int = 30,
    tap_events_count: int = 0,
    consecutive_fails: int = 0,
    idle_seconds: int = 0,
    erratic_touch_count: int = 0,
    average_latency_ms: int = 0,
    child_name: str = "Bé",
    comfort_item: Optional[str] = None
) -> Dict[str, Any]:
    """
    Tính toán Frustration Score (0 - 100) theo thuật toán Cửa Sổ Trượt 30s.
    """
    # 1. Bảo vệ giá trị biên (Negative & Boundary Check)
    safe_window = max(1, window_seconds)
    safe_taps = max(0, tap_events_count)
    safe_fails = max(0, consecutive_fails)
    safe_idle = max(0, idle_seconds)
    safe_erratic = max(0, erratic_touch_count)

    # 2. Chuẩn hóa 4 chỉ số thành phần (0 - 100)
    # Tần suất tap mỗi giây (Ngưỡng nguy hiểm >= 5 taps/giây)
    tap_rate_per_sec = safe_taps / safe_window
    s_rage = min(100.0, (tap_rate_per_sec / 5.0) * 100.0)

    # Chuỗi sai liên tiếp (Ngưỡng nguy hiểm >= 5 lần)
    s_error = min(100.0, (safe_fails / 5.0) * 100.0)

    # Độ trễ bất động / đứng hình (Hesitation freeze >= 15s)
    idle_excess = max(0.0, float(safe_idle) - 8.0)
    s_freeze = min(100.0, (idle_excess / 15.0) * 100.0)

    # Thao tác rung giật / trượt loạn xạ ngoài vùng hợp lệ (Ngưỡng >= 10 lần)
    s_jitter = min(100.0, (safe_erratic / 10.0) * 100.0)

    # 3. Trọng số tổng hợp (Weighted Frustration Score)
    # Rage-tap: 35%, Error-spike: 30%, Freeze: 20%, Jitter: 15%
    raw_score = (0.35 * s_rage) + (0.30 * s_error) + (0.20 * s_freeze) + (0.15 * s_jitter)
    frustration_score = round(max(0.0, min(100.0, raw_score)), 1)

    # 4. Phân loại mức độ nguy cơ (Risk Tiers)
    if frustration_score >= 75.0:
        risk_level = "HIGH_MELTDOWN_RISK"
        calm_mode_recommended = True
        calm_mode_action = "TRIGGER_CALM_MODE"
        breathing_pattern = "4_4_4_BOX_BREATHING"
        item_text = f" chiếc {comfort_item}" if comfort_item else " món đồ chơi quen thuộc"
        de_escalation_message = (
            f"{child_name} đang có dấu hiệu quá tải giác quan nghiêm trọng (Chỉ số ức chế: {frustration_score}%). "
            f"Hệ thống đã tự động kích hoạt Chế Độ Thư Giãn Calm Mode. Hãy cho con ôm{item_text} và nghỉ ngơi 5 phút."
        )
    elif frustration_score >= 40.0:
        risk_level = "MODERATE_FRUSTRATION"
        calm_mode_recommended = False
        calm_mode_action = "PROVIDE_HINT_AND_SLOW_TEMPO"
        breathing_pattern = "OPTIONAL"
        de_escalation_message = (
            f"{child_name} bắt đầu bồn chồn hoặc gặp khó khăn. Đề xuất hiển thị thêm gợi ý trợ giúp và giảm âm lượng nhạc nền."
        )
    else:
        risk_level = "SAFE_ZONE"
        calm_mode_recommended = False
        calm_mode_action = "CONTINUE_NORMAL"
        breathing_pattern = "NONE"
        de_escalation_message = f"{child_name} đang tương tác vui vẻ, điều hòa cảm xúc tốt."

    comfort_advice = (
        f"Hãy cho con cầm {comfort_item} để xoa dịu xúc giác."
        if comfort_item and frustration_score >= 40.0
        else "Duy trì không gian học tập yên tĩnh, thoáng mát."
    )

    return {
        "frustration_score": frustration_score,
        "risk_level": risk_level,
        "calm_mode_recommended": calm_mode_recommended,
        "calm_mode_action": calm_mode_action,
        "breathing_pattern": breathing_pattern,
        "sub_scores": {
            "rage_tap_score": round(s_rage, 1),
            "error_spike_score": round(s_error, 1),
            "hesitation_freeze_score": round(s_freeze, 1),
            "jitter_score": round(s_jitter, 1),
        },
        "de_escalation_message": de_escalation_message,
        "comfort_item_advice": comfort_advice,
    }
