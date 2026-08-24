"""
Unit Tests for ASQ-3 AI ML Service and Decision Support Engine
"""

import pytest
from asq_ai_service import AsqAiService, calculate_normal_cdf


def test_calculate_normal_cdf():
    # z=0 -> 50%
    assert abs(calculate_normal_cdf(0.0) - 0.5) < 0.01
    # z=1.96 -> ~97.5%
    assert abs(calculate_normal_cdf(1.96) - 0.975) < 0.01
    # z=-1.96 -> ~2.5%
    assert abs(calculate_normal_cdf(-1.96) - 0.025) < 0.01


def test_predict_asq_risk_typical():
    payload = {
        "age_interval_months": 12,
        "domain_scores": {
            "COMMUNICATION": 55.0,
            "GROSS_MOTOR": 50.0,
            "FINE_MOTOR": 55.0,
            "PROBLEM_SOLVING": 50.0,
            "PERSONAL_SOCIAL": 55.0
        }
    }
    result = AsqAiService.predict_asq_risk(payload)
    assert result["overall_status"] == "TYPICAL"
    assert result["overall_delay_risk"] < 0.15
    assert result["anomaly_pattern"] == "NONE"
    for domain in ["COMMUNICATION", "GROSS_MOTOR", "FINE_MOTOR", "PROBLEM_SOLVING", "PERSONAL_SOCIAL"]:
        assert result["domain_predictions"][domain]["zone"] == "WHITE"


def test_predict_asq_risk_gross_motor_delay():
    payload = {
        "age_interval_months": 12,
        "domain_scores": {
            "COMMUNICATION": 50.0,
            "GROSS_MOTOR": 15.0, # Cutoff is 22.02 -> Black zone
            "FINE_MOTOR": 50.0,
            "PROBLEM_SOLVING": 45.0,
            "PERSONAL_SOCIAL": 45.0
        }
    }
    result = AsqAiService.predict_asq_risk(payload)
    assert result["overall_status"] == "INTERVENTION_RECOMMENDED"
    assert result["overall_delay_risk"] > 0.80
    assert result["domain_predictions"]["GROSS_MOTOR"]["zone"] == "BLACK"
    assert "GROSS_MOTOR" in result["clinical_recommendation"]


def test_predict_asq_risk_asd_anomaly_pattern():
    payload = {
        "age_interval_months": 12,
        "domain_scores": {
            "COMMUNICATION": 15.0,
            "GROSS_MOTOR": 55.0, # High gross motor
            "FINE_MOTOR": 50.0,
            "PROBLEM_SOLVING": 45.0,
            "PERSONAL_SOCIAL": 10.0 # Very low social
        }
    }
    result = AsqAiService.predict_asq_risk(payload)
    assert result["overall_status"] == "INTERVENTION_RECOMMENDED"
    assert result["anomaly_pattern"] == "ASD_SOCIAL_COMMUNICATION_RISK"
    assert "M-CHAT-R" in result["clinical_recommendation"]


def test_get_quick_screen_items():
    items = AsqAiService.get_quick_screen_items(12)
    assert len(items) >= 10
    domains = {item["domain"] for item in items}
    assert "COMMUNICATION" in domains
    assert "GROSS_MOTOR" in domains
    assert "FINE_MOTOR" in domains
    assert "PROBLEM_SOLVING" in domains
    assert "PERSONAL_SOCIAL" in domains
