"""
src/train.py
------------
Machine Learning training module for regression (predicting final marks)
and classification (predicting at-risk status).
"""

import os
import sys
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from src.data_processing import load_raw_data, preprocess_pipeline

FEATURE_COLS = [
    "attendance", "study_hours", "assignment_score", "previous_marks",
    "sleep_hours", "participation", "academic_history_score", "effort_index"
]

def train_and_evaluate_models(data_path: str = "data/students.csv", models_dir: str = "models") -> Dict[str, Any]:
    """
    Main training function:
    1. Loads dataset and applies preprocessing & feature engineering.
    2. Trains Linear Regression & Random Forest Regressor.
    3. Trains Logistic Regression & Random Forest Classifier.
    4. Computes performance metrics and saves best models via joblib.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    raw_df = load_raw_data(data_path)
    df = preprocess_pipeline(raw_df)
    
    X = df[FEATURE_COLS]
    y_reg = df["final_marks"]
    y_cls = df["at_risk"]
    
    # -------------------------------------------------------------
    # 1. REGRESSION MODEL TRAINING (Final Marks Prediction)
    # -------------------------------------------------------------
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)
    
    # Train Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_r_scaled, y_train_r)
    pred_lin = lin_reg.predict(X_test_r_scaled)
    
    # Train Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)  # Tree models don't require scaling
    pred_rf_reg = rf_reg.predict(X_test_r)
    
    reg_metrics = {
        "Linear Regression": {
            "MAE": round(mean_absolute_error(y_test_r, pred_lin), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test_r, pred_lin)), 3),
            "R2": round(r2_score(y_test_r, pred_lin), 3)
        },
        "Random Forest Regressor": {
            "MAE": round(mean_absolute_error(y_test_r, pred_rf_reg), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test_r, pred_rf_reg)), 3),
            "R2": round(r2_score(y_test_r, pred_rf_reg), 3)
        }
    }
    
    # Select best regression model based on R2 score
    best_reg_name = "Linear Regression" if reg_metrics["Linear Regression"]["R2"] >= reg_metrics["Random Forest Regressor"]["R2"] else "Random Forest Regressor"
    best_reg_model = lin_reg if best_reg_name == "Linear Regression" else rf_reg
    
    # Save best regression model + scaler metadata
    reg_artifact = {
        "model_name": best_reg_name,
        "model": best_reg_model,
        "scaler": scaler_r if best_reg_name == "Linear Regression" else None,
        "feature_names": FEATURE_COLS
    }
    joblib.dump(reg_artifact, os.path.join(models_dir, "regression_model.joblib"))

    # -------------------------------------------------------------
    # 2. CLASSIFICATION MODEL TRAINING (At-Risk Prediction)
    # -------------------------------------------------------------
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    # Train Logistic Regression
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_c_scaled, y_train_c)
    pred_log = log_reg.predict(X_test_c_scaled)
    
    # Train Random Forest Classifier
    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cls.fit(X_train_c, y_train_c)
    pred_rf_cls = rf_cls.predict(X_test_c)
    
    # Calculate cross validation accuracy
    cv_log = cross_val_score(log_reg, scaler_c.transform(X), y_cls, cv=5).mean()
    cv_rf = cross_val_score(rf_cls, X, y_cls, cv=5).mean()

    cls_metrics = {
        "Logistic Regression": {
            "Accuracy": round(accuracy_score(y_test_c, pred_log), 3),
            "Precision": round(precision_score(y_test_c, pred_log, zero_division=0), 3),
            "Recall": round(recall_score(y_test_c, pred_log, zero_division=0), 3),
            "F1": round(f1_score(y_test_c, pred_log, zero_division=0), 3),
            "CV Accuracy": round(cv_log, 3),
            "Confusion Matrix": confusion_matrix(y_test_c, pred_log).tolist()
        },
        "Random Forest Classifier": {
            "Accuracy": round(accuracy_score(y_test_c, pred_rf_cls), 3),
            "Precision": round(precision_score(y_test_c, pred_rf_cls, zero_division=0), 3),
            "Recall": round(recall_score(y_test_c, pred_rf_cls, zero_division=0), 3),
            "F1": round(f1_score(y_test_c, pred_rf_cls, zero_division=0), 3),
            "CV Accuracy": round(cv_rf, 3),
            "Confusion Matrix": confusion_matrix(y_test_c, pred_rf_cls).tolist()
        }
    }
    
    # Select best classifier model based on F1 Score
    best_cls_name = "Random Forest Classifier" if cls_metrics["Random Forest Classifier"]["F1"] >= cls_metrics["Logistic Regression"]["F1"] else "Logistic Regression"
    best_cls_model = rf_cls if best_cls_name == "Random Forest Classifier" else log_reg
    
    # Save best classification artifact
    cls_artifact = {
        "model_name": best_cls_name,
        "model": best_cls_model,
        "scaler": scaler_c if best_cls_name == "Logistic Regression" else None,
        "feature_names": FEATURE_COLS,
        "feature_importances": rf_cls.feature_importances_.tolist()
    }
    joblib.dump(cls_artifact, os.path.join(models_dir, "classifier_model.joblib"))

    return {
        "regression_metrics": reg_metrics,
        "classification_metrics": cls_metrics,
        "best_reg_model": best_reg_name,
        "best_cls_model": best_cls_name,
        "feature_importances": dict(zip(FEATURE_COLS, rf_cls.feature_importances_))
    }

if __name__ == "__main__":
    results = train_and_evaluate_models()
    print("=== Training Completed Successfully ===")
    print("Best Regression Model:", results["best_reg_model"])
    print("Regression Metrics:", results["regression_metrics"])
    print("\nBest Classifier Model:", results["best_cls_model"])
    print("Classification Metrics:", results["classification_metrics"])
