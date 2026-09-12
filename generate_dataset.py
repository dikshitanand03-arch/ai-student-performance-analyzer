"""
generate_dataset.py
-------------------
Script to generate a realistic synthetic dataset for student performance analysis.
Includes realistic features, underlying correlated targets, controlled missing values,
and mild outliers to demonstrate robust data preprocessing.
"""

import os
import numpy as np
import pandas as pd

def generate_student_data(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    student_ids = np.arange(1001, 1001 + n_samples)
    
    # Feature distributions with realistic spread
    attendance = np.random.normal(75, 15, n_samples).clip(35, 100)
    study_hours = np.random.normal(10, 5, n_samples).clip(1, 25)
    previous_marks = np.random.normal(60, 16, n_samples).clip(20, 98)
    assignment_score = (0.5 * previous_marks + 0.5 * np.random.normal(60, 15, n_samples)).clip(15, 100)
    sleep_hours = np.random.normal(7.0, 1.2, n_samples).clip(4.0, 10.0)
    participation = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.25, 0.35, 0.15, 0.10])
    
    # Calculate target (final_marks) with domain-realistic formula + random noise
    base_marks = (
        0.35 * previous_marks +
        0.25 * assignment_score +
        0.25 * (attendance / 100.0 * 100.0) +
        0.15 * (study_hours / 25.0 * 100.0)
    )
    noise = np.random.normal(0, 4.0, n_samples)
    final_marks = np.round((base_marks + noise).clip(10, 100), 1)

    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance": np.round(attendance, 1),
        "study_hours": np.round(study_hours, 1),
        "assignment_score": np.round(assignment_score, 1),
        "previous_marks": np.round(previous_marks, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "participation": participation,
        "final_marks": final_marks
    })

    # Inject minor missing values (NaN) to test missing value imputation
    nan_mask_study = np.random.choice(df.index, size=12, replace=False)
    nan_mask_sleep = np.random.choice(df.index, size=8, replace=False)
    df.loc[nan_mask_study, "study_hours"] = np.nan
    df.loc[nan_mask_sleep, "sleep_hours"] = np.nan

    # Inject mild outliers to test data cleaning
    df.loc[5, "attendance"] = 150.0  # Invalid attendance > 100%
    df.loc[12, "study_hours"] = -4.0  # Invalid negative study hours

    return df

def main():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "students.csv")

    df = generate_student_data(n_samples=600)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} records.")

if __name__ == "__main__":
    main()
