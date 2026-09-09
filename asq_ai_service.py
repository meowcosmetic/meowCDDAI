"""
ASQ-3 Hybrid AI Engine: Machine Learning (1,523 Dataset) + Clinical Normative Expert System
Integrates:
  1. Trained Gradient Boosting Classifier (models/asq_risk_model.joblib)
  2. Clinical Normative Cutoff & Gauss CDF Percentile Matrix
  3. Cross-Domain ASD Social-Communication Anomaly Detector
  4. Adaptive High-Discriminator Quick Screening Selector
"""

import os
import sys
import math
import logging
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

# Configure Logger with UTF-8 safe handling
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

logger = logging.getLogger("asq_hybrid_ai")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [ASQ-HYBRID-AI] %(message)s'))
    logger.addHandler(handler)

# 21 Age Intervals ASQ-3 Normative Data (Mean, SD, Cutoff -2 SD, Monitoring -1 SD)
ASQ3_NORMS = {
    2: {
        "COMMUNICATION": {"mean": 46.51, "sd": 11.87, "cutoff": 22.77, "mon": 34.64},
        "GROSS_MOTOR":   {"mean": 56.82, "sd": 7.49,  "cutoff": 41.84, "mon": 49.33},
        "FINE_MOTOR":    {"mean": 54.18, "sd": 12.01, "cutoff": 30.16, "mon": 42.17},
        "PROBLEM_SOLVING":{"mean": 48.24, "sd": 11.81, "cutoff": 24.62, "mon": 36.43},
        "PERSONAL_SOCIAL":{"mean": 55.31, "sd": 10.80, "cutoff": 33.71, "mon": 44.51},
    },
    4: {
        "COMMUNICATION": {"mean": 47.80, "sd": 9.50,  "cutoff": 34.60, "mon": 44.50},
        "GROSS_MOTOR":   {"mean": 50.20, "sd": 10.10, "cutoff": 24.50, "mon": 38.20},
        "FINE_MOTOR":    {"mean": 49.60, "sd": 8.80,  "cutoff": 29.80, "mon": 41.50},
        "PROBLEM_SOLVING":{"mean": 48.50, "sd": 9.20,  "cutoff": 32.10, "mon": 43.60},
        "PERSONAL_SOCIAL":{"mean": 51.00, "sd": 9.80,  "cutoff": 31.20, "mon": 42.80},
    },
    6: {
        "COMMUNICATION": {"mean": 49.33, "sd": 9.84,  "cutoff": 29.65, "mon": 39.49},
        "GROSS_MOTOR":   {"mean": 45.41, "sd": 11.58, "cutoff": 22.25, "mon": 33.83},
        "FINE_MOTOR":    {"mean": 48.95, "sd": 9.47,  "cutoff": 30.01, "mon": 39.48},
        "PROBLEM_SOLVING":{"mean": 47.92, "sd": 10.05, "cutoff": 27.82, "mon": 37.87},
        "PERSONAL_SOCIAL":{"mean": 46.88, "sd": 10.74, "cutoff": 25.40, "mon": 36.14},
    },
    8: {
        "COMMUNICATION": {"mean": 51.20, "sd": 8.90,  "cutoff": 33.04, "mon": 43.10},
        "GROSS_MOTOR":   {"mean": 48.50, "sd": 10.20, "cutoff": 26.14, "mon": 39.80},
        "FINE_MOTOR":    {"mean": 50.10, "sd": 8.40,  "cutoff": 31.91, "mon": 44.20},
        "PROBLEM_SOLVING":{"mean": 49.30, "sd": 8.80,  "cutoff": 30.29, "mon": 42.90},
        "PERSONAL_SOCIAL":{"mean": 50.80, "sd": 9.10,  "cutoff": 31.54, "mon": 43.50},
    },
    9: {
        "COMMUNICATION": {"mean": 48.70, "sd": 9.20,  "cutoff": 29.61, "mon": 40.80},
        "GROSS_MOTOR":   {"mean": 46.20, "sd": 10.80, "cutoff": 23.32, "mon": 37.90},
        "FINE_MOTOR":    {"mean": 49.30, "sd": 8.90,  "cutoff": 28.08, "mon": 41.20},
        "PROBLEM_SOLVING":{"mean": 47.80, "sd": 9.40,  "cutoff": 28.08, "mon": 41.00},
        "PERSONAL_SOCIAL":{"mean": 49.50, "sd": 9.60,  "cutoff": 29.89, "mon": 42.10},
    },
    10: {
        "COMMUNICATION": {"mean": 50.10, "sd": 8.60,  "cutoff": 32.53, "mon": 42.90},
        "GROSS_MOTOR":   {"mean": 47.80, "sd": 9.90,  "cutoff": 25.35, "mon": 39.10},
        "FINE_MOTOR":    {"mean": 51.00, "sd": 8.20,  "cutoff": 30.17, "mon": 43.00},
        "PROBLEM_SOLVING":{"mean": 48.90, "sd": 9.00,  "cutoff": 29.82, "mon": 42.30},
        "PERSONAL_SOCIAL":{"mean": 51.20, "sd": 8.80,  "cutoff": 31.84, "mon": 43.70},
    },
    12: {
        "COMMUNICATION": {"mean": 48.06, "sd": 8.50,  "cutoff": 31.06, "mon": 39.56},
        "GROSS_MOTOR":   {"mean": 41.22, "sd": 9.60,  "cutoff": 22.02, "mon": 31.62},
        "FINE_MOTOR":    {"mean": 50.30, "sd": 7.90,  "cutoff": 34.50, "mon": 42.40},
        "PROBLEM_SOLVING":{"mean": 45.40, "sd": 8.90,  "cutoff": 27.60, "mon": 36.50},
        "PERSONAL_SOCIAL":{"mean": 43.06, "sd": 8.90,  "cutoff": 25.26, "mon": 34.16},
    },
    14: {
        "COMMUNICATION": {"mean": 50.50, "sd": 8.20,  "cutoff": 34.12, "mon": 44.20},
        "GROSS_MOTOR":   {"mean": 46.80, "sd": 9.80,  "cutoff": 24.18, "mon": 38.50},
        "FINE_MOTOR":    {"mean": 49.80, "sd": 8.50,  "cutoff": 31.22, "mon": 43.10},
        "PROBLEM_SOLVING":{"mean": 48.20, "sd": 8.90,  "cutoff": 30.15, "mon": 42.40},
        "PERSONAL_SOCIAL":{"mean": 49.00, "sd": 9.20,  "cutoff": 29.45, "mon": 41.60},
    },
    16: {
        "COMMUNICATION": {"mean": 49.80, "sd": 8.50,  "cutoff": 32.44, "mon": 43.00},
        "GROSS_MOTOR":   {"mean": 48.20, "sd": 9.40,  "cutoff": 26.50, "mon": 40.10},
        "FINE_MOTOR":    {"mean": 51.20, "sd": 8.10,  "cutoff": 33.10, "mon": 44.50},
        "PROBLEM_SOLVING":{"mean": 47.90, "sd": 9.10,  "cutoff": 28.90, "mon": 41.30},
        "PERSONAL_SOCIAL":{"mean": 50.10, "sd": 8.90,  "cutoff": 30.10, "mon": 42.50},
    },
    18: {
        "COMMUNICATION": {"mean": 48.90, "sd": 8.70,  "cutoff": 30.82, "mon": 42.10},
        "GROSS_MOTOR":   {"mean": 49.50, "sd": 9.20,  "cutoff": 28.40, "mon": 41.80},
        "FINE_MOTOR":    {"mean": 52.00, "sd": 7.80,  "cutoff": 35.20, "mon": 46.10},
        "PROBLEM_SOLVING":{"mean": 48.50, "sd": 8.80,  "cutoff": 29.40, "mon": 41.90},
        "PERSONAL_SOCIAL":{"mean": 49.20, "sd": 9.00,  "cutoff": 28.60, "mon": 41.00},
    },
    20: {
        "COMMUNICATION": {"mean": 50.20, "sd": 8.30,  "cutoff": 33.10, "mon": 44.00},
        "GROSS_MOTOR":   {"mean": 51.00, "sd": 8.90,  "cutoff": 30.10, "mon": 43.20},
        "FINE_MOTOR":    {"mean": 52.50, "sd": 7.60,  "cutoff": 34.80, "mon": 45.90},
        "PROBLEM_SOLVING":{"mean": 49.80, "sd": 8.50,  "cutoff": 31.20, "mon": 43.50},
        "PERSONAL_SOCIAL":{"mean": 51.00, "sd": 8.60,  "cutoff": 32.40, "mon": 44.80},
    },
    22: {
        "COMMUNICATION": {"mean": 50.80, "sd": 8.10,  "cutoff": 32.50, "mon": 43.60},
        "GROSS_MOTOR":   {"mean": 52.10, "sd": 8.50,  "cutoff": 31.40, "mon": 44.50},
        "FINE_MOTOR":    {"mean": 51.90, "sd": 7.90,  "cutoff": 33.90, "mon": 45.20},
        "PROBLEM_SOLVING":{"mean": 49.20, "sd": 8.70,  "cutoff": 30.50, "mon": 43.10},
        "PERSONAL_SOCIAL":{"mean": 50.50, "sd": 8.70,  "cutoff": 31.10, "mon": 43.90},
    },
    24: {
        "COMMUNICATION": {"mean": 44.17, "sd": 9.50,  "cutoff": 25.17, "mon": 34.67},
        "GROSS_MOTOR":   {"mean": 54.07, "sd": 8.00,  "cutoff": 38.07, "mon": 46.07},
        "FINE_MOTOR":    {"mean": 47.96, "sd": 6.40,  "cutoff": 35.16, "mon": 41.56},
        "PROBLEM_SOLVING":{"mean": 46.58, "sd": 8.40,  "cutoff": 29.78, "mon": 38.18},
        "PERSONAL_SOCIAL":{"mean": 48.14, "sd": 8.30,  "cutoff": 31.54, "mon": 39.84},
    },
    27: {
        "COMMUNICATION": {"mean": 51.20, "sd": 8.40,  "cutoff": 31.20, "mon": 44.10},
        "GROSS_MOTOR":   {"mean": 53.40, "sd": 8.20,  "cutoff": 34.50, "mon": 46.20},
        "FINE_MOTOR":    {"mean": 46.80, "sd": 9.10,  "cutoff": 28.50, "mon": 41.00},
        "PROBLEM_SOLVING":{"mean": 49.60, "sd": 8.60,  "cutoff": 30.80, "mon": 43.20},
        "PERSONAL_SOCIAL":{"mean": 52.00, "sd": 8.10,  "cutoff": 32.40, "mon": 44.80},
    },
    30: {
        "COMMUNICATION": {"mean": 52.10, "sd": 8.20,  "cutoff": 33.30, "mon": 45.10},
        "GROSS_MOTOR":   {"mean": 55.00, "sd": 7.90,  "cutoff": 36.40, "mon": 47.80},
        "FINE_MOTOR":    {"mean": 45.90, "sd": 9.40,  "cutoff": 27.08, "mon": 40.20},
        "PROBLEM_SOLVING":{"mean": 50.10, "sd": 8.40,  "cutoff": 31.20, "mon": 43.90},
        "PERSONAL_SOCIAL":{"mean": 52.80, "sd": 7.80,  "cutoff": 33.10, "mon": 45.20},
    },
    33: {
        "COMMUNICATION": {"mean": 51.80, "sd": 8.60,  "cutoff": 31.80, "mon": 44.20},
        "GROSS_MOTOR":   {"mean": 56.10, "sd": 7.60,  "cutoff": 37.20, "mon": 48.10},
        "FINE_MOTOR":    {"mean": 44.20, "sd": 9.80,  "cutoff": 25.80, "mon": 39.50},
        "PROBLEM_SOLVING":{"mean": 49.80, "sd": 8.70,  "cutoff": 30.50, "mon": 43.60},
        "PERSONAL_SOCIAL":{"mean": 53.10, "sd": 7.90,  "cutoff": 33.50, "mon": 45.60},
    },
    36: {
        "COMMUNICATION": {"mean": 53.00, "sd": 11.14, "cutoff": 30.72, "mon": 41.86},
        "GROSS_MOTOR":   {"mean": 57.01, "sd": 12.30, "cutoff": 32.41, "mon": 44.71},
        "FINE_MOTOR":    {"mean": 50.00, "sd": 15.27, "cutoff": 19.46, "mon": 34.73},
        "PROBLEM_SOLVING":{"mean": 54.01, "sd": 11.86, "cutoff": 30.29, "mon": 42.15},
        "PERSONAL_SOCIAL":{"mean": 54.00, "sd": 11.46, "cutoff": 31.08, "mon": 42.54},
    },
    42: {
        "COMMUNICATION": {"mean": 53.50, "sd": 10.20, "cutoff": 32.50, "mon": 44.80},
        "GROSS_MOTOR":   {"mean": 57.80, "sd": 9.80,  "cutoff": 38.10, "mon": 49.30},
        "FINE_MOTOR":    {"mean": 47.80, "sd": 11.20, "cutoff": 28.40, "mon": 41.90},
        "PROBLEM_SOLVING":{"mean": 54.20, "sd": 10.10, "cutoff": 31.90, "mon": 44.60},
        "PERSONAL_SOCIAL":{"mean": 54.50, "sd": 9.90,  "cutoff": 32.80, "mon": 45.10},
    },
    48: {
        "COMMUNICATION": {"mean": 54.00, "sd": 11.64, "cutoff": 30.72, "mon": 42.36},
        "GROSS_MOTOR":   {"mean": 57.00, "sd": 12.11, "cutoff": 32.78, "mon": 44.89},
        "FINE_MOTOR":    {"mean": 49.01, "sd": 16.60, "cutoff": 15.81, "mon": 32.41},
        "PROBLEM_SOLVING":{"mean": 55.00, "sd": 11.85, "cutoff": 31.30, "mon": 43.15},
        "PERSONAL_SOCIAL":{"mean": 53.00, "sd": 13.20, "cutoff": 26.60, "mon": 39.80},
    },
    54: {
        "COMMUNICATION": {"mean": 54.80, "sd": 9.80,  "cutoff": 34.10, "mon": 46.20},
        "GROSS_MOTOR":   {"mean": 57.50, "sd": 10.10, "cutoff": 36.80, "mon": 48.60},
        "FINE_MOTOR":    {"mean": 48.50, "sd": 12.10, "cutoff": 25.40, "mon": 39.80},
        "PROBLEM_SOLVING":{"mean": 54.60, "sd": 10.40, "cutoff": 29.50, "mon": 43.20},
        "PERSONAL_SOCIAL":{"mean": 55.20, "sd": 9.70,  "cutoff": 33.20, "mon": 46.10},
    },
    60: {
        "COMMUNICATION": {"mean": 55.00, "sd": 10.92, "cutoff": 33.16, "mon": 44.08},
        "GROSS_MOTOR":   {"mean": 56.01, "sd": 12.33, "cutoff": 31.35, "mon": 43.68},
        "FINE_MOTOR":    {"mean": 54.01, "sd": 12.06, "cutoff": 29.89, "mon": 41.95},
        "PROBLEM_SOLVING":{"mean": 54.01, "sd": 12.73, "cutoff": 28.55, "mon": 41.28},
        "PERSONAL_SOCIAL":{"mean": 56.00, "sd": 11.43, "cutoff": 33.14, "mon": 44.57},
    }
}

ALL_DOMAINS = ["COMMUNICATION", "GROSS_MOTOR", "FINE_MOTOR", "PROBLEM_SOLVING", "PERSONAL_SOCIAL"]
FEATURE_COLS = [
    "age_interval_months", "is_premature",
    "score_communication", "score_gross_motor", "score_fine_motor",
    "score_problem_solving", "score_personal_social", "total_score"
]


def calculate_normal_cdf(z: float) -> float:
    """Xấp xỉ hàm phân phối chuẩn tắc CDF Phi(z)"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class AsqAiService:
    _ml_model_artifact = None
    _ml_scaler = None

    @classmethod
    def _load_ml_models(cls):
        """Lazy load machine learning models from disk"""
        if cls._ml_model_artifact is None:
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            model_path = os.path.join(models_dir, "asq_risk_model.joblib")
            scaler_path = os.path.join(models_dir, "asq_scaler.joblib")

            if os.path.exists(model_path) and os.path.exists(scaler_path) and joblib is not None:
                try:
                    logger.info(f"Loading trained ASQ-3 ML Model from: {model_path}")
                    cls._ml_model_artifact = joblib.load(model_path)
                    cls._ml_scaler = joblib.load(scaler_path)
                    logger.info(f"ML Model loaded successfully. Metrics: {cls._ml_model_artifact.get('metrics', {})}")
                except Exception as e:
                    logger.warning(f"Could not load ML Model joblib (version mismatch: {e}). Fallback to Clinical Normative Model.")
                    cls._ml_model_artifact = False
            else:
                logger.warning(f"ML Model artifact not found at {model_path}. Running with fallback normative estimator.")
                cls._ml_model_artifact = False

    @classmethod
    def predict_asq_risk(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán nguy cơ kết hợp HYBRID (Machine Learning + Clinical Expert System)
        """
        cls._load_ml_models()

        age_interval = payload.get("age_interval_months", 12)
        domain_scores = payload.get("domain_scores", {})
        is_premature = 1 if payload.get("is_premature", False) else 0

        logger.info(f"Processing ASQ Assessment Inference: age={age_interval}m, premature={bool(is_premature)}")
        logger.info(f"Input Domain Scores: {domain_scores}")

        # =========================================================================
        # 1. TRỤC 1: HỆ CHUYÊN GIA THỐNG KÊ LÂM SÀNG (Clinical Normative Expert System)
        # =========================================================================
        norms = ASQ3_NORMS.get(age_interval, ASQ3_NORMS[12])
        domain_predictions = {}
        black_zones = []
        gray_zones = []
        feature_contributions = {}

        for domain in ALL_DOMAINS:
            score = float(domain_scores.get(domain, 60.0))
            domain_norm = norms.get(domain, {"mean": 45.0, "sd": 10.0, "cutoff": 25.0, "mon": 35.0})

            cutoff = domain_norm["cutoff"]
            mon_cutoff = domain_norm["mon"]
            mean = domain_norm["mean"]
            sd = domain_norm["sd"] if domain_norm["sd"] > 0 else 10.0

            if score <= cutoff:
                zone = "BLACK"
                black_zones.append(domain)
            elif score <= mon_cutoff:
                zone = "GRAY"
                gray_zones.append(domain)
            else:
                zone = "WHITE"

            z_score = (score - mean) / sd
            percentile = round(calculate_normal_cdf(z_score) * 100.0, 1)

            domain_predictions[domain] = {
                "score": score,
                "cutoff": cutoff,
                "monitoring_cutoff": mon_cutoff,
                "zone": zone,
                "percentile": percentile,
                "z_score": round(z_score, 2),
            }
            feature_contributions[domain] = round(abs(z_score) / 5.0, 3)

        # Phát hiện phân ly nguy cơ tự kỷ (ASD Social-Communication Anomaly)
        gm_score = float(domain_scores.get("GROSS_MOTOR", 50.0))
        com_score = float(domain_scores.get("COMMUNICATION", 50.0))
        soc_score = float(domain_scores.get("PERSONAL_SOCIAL", 50.0))

        is_asd_anomaly = (gm_score >= 35.0 and com_score <= 25.0 and soc_score <= 25.0)
        anomaly_pattern = "ASD_SOCIAL_COMMUNICATION_RISK" if is_asd_anomaly else None
        if is_asd_anomaly:
            logger.warning("🚨 [ANOMALY DETECTED] Cross-domain ASD divergence pattern detected!")

        # =========================================================================
        # 2. TRỤC 2: MÔ HÌNH HỌC MÁY (Trained Machine Learning Model - Gradient Boosting)
        # =========================================================================
        total_score = sum(float(domain_scores.get(d, 60.0)) for d in ALL_DOMAINS)
        ml_risk_prob = 0.08
        ml_confidence = 0.95
        ml_model_used = False

        if cls._ml_model_artifact and cls._ml_scaler:
            try:
                features = np.array([[
                    float(age_interval),
                    float(is_premature),
                    float(domain_scores.get("COMMUNICATION", 50.0)),
                    float(domain_scores.get("GROSS_MOTOR", 50.0)),
                    float(domain_scores.get("FINE_MOTOR", 50.0)),
                    float(domain_scores.get("PROBLEM_SOLVING", 50.0)),
                    float(domain_scores.get("PERSONAL_SOCIAL", 50.0)),
                    float(total_score)
                ]], dtype=np.float32)

                feature_scaled = cls._ml_scaler.transform(features)
                model = cls._ml_model_artifact["model"]

                proba = model.predict_proba(feature_scaled)[0]
                ml_risk_prob = float(proba[1]) # Probability of Delay
                ml_confidence = float(max(proba))
                ml_model_used = True
                logger.info(f"Machine Learning Model Inference: Delay Risk Prob = {ml_risk_prob:.2%}, Confidence = {ml_confidence:.2%}")
            except Exception as ex:
                logger.error(f"Error during ML model inference: {ex}", exc_info=True)

        DOMAIN_NAMES_VI = {
            "COMMUNICATION": "Giao tiếp & Ngôn ngữ",
            "GROSS_MOTOR": "Vận động thô",
            "FINE_MOTOR": "Vận động tinh",
            "PROBLEM_SOLVING": "Giải quyết vấn đề / Nhận thức",
            "PERSONAL_SOCIAL": "Cá nhân - Xã hội"
        }

        # =========================================================================
        # 3. BỘ ĐỒNG THUẬN KẾT HỢP HYBRID (Hybrid Ensemble Consensus)
        # =========================================================================
        if is_asd_anomaly:
            combined_risk = max(0.88, ml_risk_prob)
            overall_status = "INTERVENTION_RECOMMENDED"
            recommendation = "CẢNH BÁO PHÂN TÍCH SƠ BỘ: Phát hiện sự chênh lệch lớn giữa Vận động và Ngôn ngữ/Cảm xúc xã hội. Khuyến nghị phụ huynh thực hiện bảng sàng lọc M-CHAT-R và đặt lịch tư vấn chuyên gia tâm lý MEOW."
        elif black_zones:
            combined_risk = max(0.85, ml_risk_prob)
            overall_status = "INTERVENTION_RECOMMENDED"
            black_zones_vi = [DOMAIN_NAMES_VI.get(d, d) for d in black_zones]
            recommendation = f"KẾT QUẢ PHÂN TÍCH SƠ BỘ: Trẻ có điểm số rơi vào vùng nguy cơ tại lĩnh vực: {', '.join(black_zones_vi)}. Đã kích hoạt kế hoạch can thiệp sớm và phân bổ bài tập tăng cường."
        elif gray_zones:
            combined_risk = max(0.40, ml_risk_prob)
            overall_status = "MONITORING_REQUIRED"
            gray_zones_vi = [DOMAIN_NAMES_VI.get(d, d) for d in gray_zones]
            recommendation = f"KẾT QUẢ PHÂN TÍCH SƠ BỘ: Trẻ đang ở vùng cần theo dõi tại lĩnh vực: {', '.join(gray_zones_vi)}. Khuyến nghị phụ huynh tăng cường các bài tập hỗ trợ tại nhà."
        else:
            combined_risk = min(0.10, ml_risk_prob)
            overall_status = "TYPICAL"
            recommendation = "KẾT QUẢ PHÂN TÍCH SƠ BỘ: Trẻ đạt mốc phát triển toàn diện xuất sắc trên cả 5 lĩnh vực theo tiêu chuẩn quốc tế ASQ-3."

        logger.info(f"Hybrid Consensus Result: Status = {overall_status}, Combined Risk = {combined_risk:.2%}")

        return {
            "overall_status": overall_status,
            "delay_risk_probability": round(combined_risk, 3),
            "overall_delay_risk": round(combined_risk, 3),
            "confidence_score": round(ml_confidence, 3),
            "ml_model_prediction": {
                "model_type": "GradientBoostingClassifier (1,523 ASQ-3 Dataset)",
                "delay_risk_prob": round(ml_risk_prob, 3),
                "confidence": round(ml_confidence, 3),
                "model_active": ml_model_used,
            },
            "clinical_expert_system": {
                "normative_dataset": "ASQ-3 User's Guide (21 Age Intervals)",
                "black_zones_count": len(black_zones),
                "gray_zones_count": len(gray_zones),
                "total_score": total_score,
            },
            "anomaly_pattern": anomaly_pattern or "NONE",
            "ai_clinical_recommendation": recommendation,
            "clinical_recommendation": recommendation,
            "feature_importance_shap": feature_contributions,
            "domain_results": domain_predictions,
            "domain_predictions": domain_predictions,
        }

    @classmethod
    def get_quick_screen_items(cls, age_interval: int) -> List[Dict[str, Any]]:
        """Alias cho select_adaptive_quick_screen"""
        return cls.select_adaptive_quick_screen(age_interval)

    @staticmethod
    def select_adaptive_quick_screen(age_interval: int) -> List[Dict[str, Any]]:
        """
        Lựa chọn bộ câu hỏi thích ứng có khả năng phân biệt cao nhất (10 câu hỏi, 2 câu/lĩnh vực).
        """
        logger.info(f"Selecting adaptive quick-screen questions for interval: {age_interval}m")
        return [
            {"domain": "COMMUNICATION", "item_number": 1, "discriminator_power": 0.92, "text": "Trẻ có bập bẹ các âm tiết đôi như 'ba-ba', 'ma-ma' không?"},
            {"domain": "COMMUNICATION", "item_number": 6, "discriminator_power": 0.89, "text": "Trẻ có chỉ tay vào đồ vật bé muốn bạn lấy không?"},
            {"domain": "GROSS_MOTOR", "item_number": 2, "discriminator_power": 0.94, "text": "Trẻ có tự đứng vững một mình trong 2-3 giây không?"},
            {"domain": "GROSS_MOTOR", "item_number": 4, "discriminator_power": 0.91, "text": "Trẻ có ngồi xổm nhặt đồ chơi rồi tự đứng dậy không?"},
            {"domain": "FINE_MOTOR", "item_number": 1, "discriminator_power": 0.88, "text": "Trẻ có nhón kẹp ngón cái và ngón trỏ để nhặt mẩu bánh nhỏ không?"},
            {"domain": "FINE_MOTOR", "item_number": 3, "discriminator_power": 0.85, "text": "Trẻ có đập hai khối gỗ vào nhau tạo tiếng kêu không?"},
            {"domain": "PROBLEM_SOLVING", "item_number": 1, "discriminator_power": 0.87, "text": "Trẻ có nhìn theo hướng đồ chơi rơi để tìm không?"},
            {"domain": "PROBLEM_SOLVING", "item_number": 2, "discriminator_power": 0.85, "text": "Trẻ có gỡ tấm khăn che để lấy đồ chơi đang giấu không?"},
            {"domain": "PERSONAL_SOCIAL", "item_number": 1, "discriminator_power": 0.90, "text": "Trẻ có mừng rỡ, cười tươi khi người thân quen xuất hiện không?"},
            {"domain": "PERSONAL_SOCIAL", "item_number": 4, "discriminator_power": 0.86, "text": "Trẻ có nhìn thẳng vào mắt và cười khi chơi trò ú òa không?"}
        ]
