"""
src/analysis.py
---------------
Exploratory Data Analysis (EDA) statistics and numerical analytics helper module.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

NUMERICAL_COLS = [
    "attendance", "study_hours", "assignment_score", 
    "previous_marks", "sleep_hours", "participation", "final_marks"
]

def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates mean, median, std, min, max, and quantiles for numerical features."""
    cols_to_calc = [c for c in NUMERICAL_COLS if c in df.columns]
    
    stats_df = pd.DataFrame({
        "Mean": df[cols_to_calc].mean(),
        "Median": df[cols_to_calc].median(),
        "Std Dev": df[cols_to_calc].std(),
        "Min": df[cols_to_calc].min(),
        "25%": df[cols_to_calc].quantile(0.25),
        "75%": df[cols_to_calc].quantile(0.75),
        "Max": df[cols_to_calc].max()
    })
    
    return np.round(stats_df, 2)

def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Pearson correlation matrix for numerical features."""
    cols_to_calc = [c for c in NUMERICAL_COLS if c in df.columns]
    if "effort_index" in df.columns:
        cols_to_calc.append("effort_index")
    if "academic_history_score" in df.columns:
        cols_to_calc.append("academic_history_score")
        
    return np.round(df[cols_to_calc].corr(), 2)

def calculate_risk_summary(df: pd.DataFrame, risk_threshold: float = 50.0) -> Dict[str, Any]:
    """Provides high-level breakdown of student risk categories."""
    if "at_risk" in df.columns:
        at_risk_series = df["at_risk"]
    else:
        at_risk_series = (df["final_marks"] < risk_threshold).astype(int)
        
    total_students = len(df)
    at_risk_count = int(at_risk_series.sum())
    safe_count = total_students - at_risk_count
    
    return {
        "total_students": total_students,
        "at_risk_count": at_risk_count,
        "safe_count": safe_count,
        "at_risk_percentage": round((at_risk_count / total_students) * 100, 1),
        "safe_percentage": round((safe_count / total_students) * 100, 1)
    }

def get_feature_correlations_with_target(df: pd.DataFrame, target_col: str = "final_marks") -> pd.Series:
    """Returns correlation of each feature specifically with the target variable."""
    corr_matrix = calculate_correlations(df)
    if target_col in corr_matrix.columns:
        return corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    return pd.Series(dtype=float)
