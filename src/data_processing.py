"""
src/data_processing.py
----------------------
Data loading, cleaning, validation, missing value imputation,
outlier handling, and feature engineering module.
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "student_id", "attendance", "study_hours", "assignment_score",
    "previous_marks", "sleep_hours", "participation", "final_marks"
]

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset and checks for required columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at path: {file_path}")
    
    df = pd.read_csv(file_path)
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded CSV is missing required columns: {missing_cols}")
    
    return df

def get_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a summary of data shape, missing values, and data types."""
    missing_counts = df.isnull().sum()
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": missing_counts[missing_counts > 0].to_dict(),
        "total_missing_cells": int(missing_counts.sum())
    }

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    1. Fixes invalid/outlier values (e.g. attendance > 100 or negative study hours).
    2. Imputes missing numerical values using column medians.
    """
    df_clean = df.copy()

    # Step 1: Outlier and Boundary Cleaning
    # Attendance must be between 0% and 100%
    df_clean.loc[df_clean["attendance"] > 100, "attendance"] = 100.0
    df_clean.loc[df_clean["attendance"] < 0, "attendance"] = 0.0

    # Study hours cannot be negative
    df_clean.loc[df_clean["study_hours"] < 0, "study_hours"] = np.nan

    # Sleep hours reasonable bounds (3 to 12)
    df_clean.loc[(df_clean["sleep_hours"] < 3) | (df_clean["sleep_hours"] > 12), "sleep_hours"] = np.nan

    # Step 2: Missing Value Imputation (Median for numerical attributes)
    numerical_cols = ["attendance", "study_hours", "assignment_score", "previous_marks", "sleep_hours"]
    for col in numerical_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    return df_clean

def engineer_features(df: pd.DataFrame, risk_threshold: float = 50.0) -> pd.DataFrame:
    """
    Creates new domain-specific features and binary target for risk classification:
    - effort_index: Composite interaction of study hours and attendance.
    - academic_history: Weighted blend of previous marks and assignment scores.
    - at_risk: Binary indicator (1 if final_marks < risk_threshold else 0).
    """
    df_fe = df.copy()

    # Domain Feature 1: Academic History Score
    df_fe["academic_history_score"] = np.round(
        0.6 * df_fe["previous_marks"] + 0.4 * df_fe["assignment_score"], 2
    )

    # Domain Feature 2: Effort Index
    df_fe["effort_index"] = np.round(
        (df_fe["study_hours"] / 25.0) * (df_fe["attendance"] / 100.0) * 100, 2
    )

    # Target Feature for Classification: At-Risk Status
    df_fe["at_risk"] = (df_fe["final_marks"] < risk_threshold).astype(int)

    return df_fe

def preprocess_pipeline(df: pd.DataFrame, risk_threshold: float = 50.0) -> pd.DataFrame:
    """Master pipeline function combining cleaning and feature engineering."""
    df_clean = clean_data(df)
    df_processed = engineer_features(df_clean, risk_threshold=risk_threshold)
    return df_processed
