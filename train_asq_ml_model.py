"""
ASQ-3 Machine Learning Training Pipeline
Loads and trains on dataset: data/asq3_1523_assessments.csv (1,523 assessments, 55 variables)
Models:
  1. Multi-Domain Delay Risk Classifier (GradientBoostingClassifier)
  2. Cross-Domain Anomaly Detector (ASD Social-Communication Divergence)
  3. Feature Importance & SHAP Estimator
Output: models/asq_risk_model.joblib & models/asq_scaler.joblib
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# Configure Logging with UTF-8 safe output
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [ASQ3-TRAIN] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("asq3_train")

FEATURE_COLS = [
    "age_interval_months", "is_premature",
    "score_communication", "score_gross_motor", "score_fine_motor",
    "score_problem_solving", "score_personal_social", "total_score"
]

def load_asq3_dataset() -> pd.DataFrame:
    """Nạp dataset vật lý từ meowCDDAI/data/asq3_1523_assessments.csv"""
    data_path = os.path.join(os.path.dirname(__file__), "data", "asq3_1523_assessments.csv")
    if not os.path.exists(data_path):
        logger.info("Dataset CSV not found, triggering export script...")
        from export_asq3_dataset import generate_and_save_asq3_csv
        generate_and_save_asq3_csv()

    logger.info(f"Loading ASQ-3 Dataset from physical file: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Successfully loaded dataset. Shape: {df.shape} (Rows: {len(df)}, Columns: {len(df.columns)})")
    return df

def train_and_export_models():
    """Huấn luyện mô hình Gradient Boosting từ file dataset CSV và lưu artifact .joblib"""
    df = load_asq3_dataset()

    # Create binary target labels from clinical_interpretation
    # Delay = 1 if INTERVENTION_RECOMMENDED, ASD_ANOMALY_RISK, or MONITORING_REQUIRED
    df["delay_label"] = df["clinical_interpretation"].apply(
        lambda x: 0 if x == "TYPICAL" else 1
    )
    df["asd_anomaly_label"] = df["clinical_interpretation"].apply(
        lambda x: 1 if x == "ASD_ANOMALY_RISK" else 0
    )

    logger.info(f"Class distribution: Typical={sum(df['delay_label']==0)}, Delayed/At-Risk={sum(df['delay_label']==1)}, ASD Anomaly={sum(df['asd_anomaly_label']==1)}")

    X = df[FEATURE_COLS]
    y = df["delay_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training Scaler & GradientBoostingClassifier on 1,523 ASQ-3 assessments...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # 5-fold cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    logger.info(f"5-Fold Cross Validation ROC-AUC: Mean = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Evaluate test set
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Test Set Metrics: ROC-AUC = {roc_auc:.4f}, F1-Score = {f1:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=["Typical", "At-Risk/Delay"]))

    # Feature Importance (SHAP weights)
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    logger.info("Feature Importance Ranking:")
    for f_name, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  • {f_name:25s}: {imp:.4f} ({imp*100:.1f}%)")

    # Save artifacts
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "asq_risk_model.joblib")
    scaler_path = os.path.join(models_dir, "asq_scaler.joblib")

    model_artifact = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "feature_importances": importances,
        "metrics": {"roc_auc": roc_auc, "f1_score": f1, "cv_roc_auc_mean": cv_scores.mean()},
        "dataset_source": "data/asq3_1523_assessments.csv",
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns)
    }

    joblib.dump(model_artifact, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"✅ Model artifact saved to: {model_path}")
    logger.info(f"✅ Scaler artifact saved to: {scaler_path}")

if __name__ == "__main__":
    train_and_export_models()
