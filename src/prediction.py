"""
src/prediction.py
------------------
Inference engine module for single-student and batch predictions,
risk probability scoring, and key risk factor diagnostics.
"""

import os
import sys
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

from src.data_processing import engineer_features, clean_data
from src.train import FEATURE_COLS

def load_saved_models(models_dir: str = "models") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads saved Joblib model artifacts for regression and classification."""
    reg_path = os.path.join(models_dir, "regression_model.joblib")
    cls_path = os.path.join(models_dir, "classifier_model.joblib")
    
    if not os.path.exists(reg_path) or not os.path.exists(cls_path):
        from src.train import train_and_evaluate_models
        train_and_evaluate_models(models_dir=models_dir)
        
    reg_artifact = joblib.load(reg_path)
    cls_artifact = joblib.load(cls_path)
    
    return reg_artifact, cls_artifact

def diagnose_risk_factors(student_data: Dict[str, Any]) -> List[str]:
    """Identifies specific risk triggers based on domain thresholds."""
    factors = []
    
    attendance = student_data.get("attendance", 100)
    study_hours = student_data.get("study_hours", 20)
    previous_marks = student_data.get("previous_marks", 100)
    assignment_score = student_data.get("assignment_score", 100)
    sleep_hours = student_data.get("sleep_hours", 8)
    
    if attendance < 70.0:
        factors.append(f"Low Attendance ({attendance:.1f}%)")
    if study_hours < 8.0:
        factors.append(f"Low Weekly Study Time ({study_hours:.1f} hrs/week)")
    if previous_marks < 50.0:
        factors.append(f"Low Previous Academic Marks ({previous_marks:.1f}/100)")
    if assignment_score < 55.0:
        factors.append(f"Low Assignment Scores ({assignment_score:.1f}/100)")
    if sleep_hours < 6.0:
        factors.append(f"Insufficient Sleep ({sleep_hours:.1f} hrs/day)")
        
    if not factors:
        factors.append("No major negative factors identified")
        
    return factors

def predict_single_student(student_dict: Dict[str, Any], models_dir: str = "models") -> Dict[str, Any]:
    """
    Runs prediction for a single student dictionary.
    Returns predicted final mark, risk category, risk probability, and key factors.
    """
    reg_artifact, cls_artifact = load_saved_models(models_dir)
    
    df_single = pd.DataFrame([student_dict])
    df_clean = clean_data(df_single)
    df_proc = engineer_features(df_clean)
    
    feature_cols = cls_artifact["feature_names"]
    X_single = df_proc[feature_cols]
    
    # 1. Regression Prediction
    reg_model = reg_artifact["model"]
    scaler_r = reg_artifact.get("scaler")
    if scaler_r:
        X_reg_scaled = scaler_r.transform(X_single)
        pred_mark = float(reg_model.predict(X_reg_scaled)[0])
    else:
        pred_mark = float(reg_model.predict(X_single)[0])
    pred_mark = float(np.clip(pred_mark, 0.0, 100.0))
    
    # 2. Classification Prediction
    cls_model = cls_artifact["model"]
    scaler_c = cls_artifact.get("scaler")
    if scaler_c:
        X_cls_scaled = scaler_c.transform(X_single)
        pred_risk_code = int(cls_model.predict(X_cls_scaled)[0])
        if hasattr(cls_model, "predict_proba"):
            risk_prob = float(cls_model.predict_proba(X_cls_scaled)[0][1])
        else:
            risk_prob = 1.0 if pred_risk_code == 1 else 0.0
    else:
        pred_risk_code = int(cls_model.predict(X_single)[0])
        if hasattr(cls_model, "predict_proba"):
            risk_prob = float(cls_model.predict_proba(X_single)[0][1])
        else:
            risk_prob = 1.0 if pred_risk_code == 1 else 0.0

    # Risk level categorization based on probability
    if risk_prob >= 0.65:
        risk_level = "HIGH RISK"
    elif risk_prob >= 0.35:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK (SAFE)"

    contributing_factors = diagnose_risk_factors(student_dict)
    
    return {
        "predicted_final_marks": round(pred_mark, 1),
        "at_risk_code": pred_risk_code,
        "risk_probability": round(risk_prob * 100, 1),
        "risk_level": risk_level,
        "contributing_factors": contributing_factors
    }

def batch_predict(df: pd.DataFrame, models_dir: str = "models") -> pd.DataFrame:
    """Runs batch predictions for an entire DataFrame."""
    reg_artifact, cls_artifact = load_saved_models(models_dir)
    
    df_clean = clean_data(df)
    df_proc = engineer_features(df_clean)
    feature_cols = cls_artifact["feature_names"]
    X = df_proc[feature_cols]
    
    # Regression
    reg_model = reg_artifact["model"]
    scaler_r = reg_artifact.get("scaler")
    X_reg = scaler_r.transform(X) if scaler_r else X
    pred_marks = np.clip(reg_model.predict(X_reg), 0.0, 100.0)
    
    # Classification
    cls_model = cls_artifact["model"]
    scaler_c = cls_artifact.get("scaler")
    X_cls = scaler_c.transform(X) if scaler_c else X
    
    if hasattr(cls_model, "predict_proba"):
        risk_probs = cls_model.predict_proba(X_cls)[:, 1]
    else:
        risk_probs = cls_model.predict(X_cls)
        
    res_df = df_proc.copy()
    res_df["Predicted_Final_Marks"] = np.round(pred_marks, 1)
    res_df["Risk_Probability_%"] = np.round(risk_probs * 100, 1)
    res_df["Predicted_Risk_Status"] = np.where(risk_probs >= 0.5, "At-Risk", "Safe")
    
    return res_df
