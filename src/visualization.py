"""
src/visualization.py
--------------------
Visualization generation module using Matplotlib and Seaborn for Streamlit.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

PRIMARY_COLOR = "#3182ce"
SECONDARY_COLOR = "#e53e3e"
ACCENT_COLOR = "#38a169"
PALETTE = ["#3182ce", "#e53e3e"]

def plot_attendance_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Plots Attendance % vs Final Marks, hue-coded by At-Risk category."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    hue_col = "at_risk" if "at_risk" in df.columns else None
    sns.scatterplot(
        data=df,
        x="attendance",
        y="final_marks",
        hue=hue_col,
        palette=PALETTE if hue_col else None,
        alpha=0.8,
        s=50,
        ax=ax
    )
    
    ax.axhline(50, color="gray", linestyle="--", alpha=0.6, label="Risk Threshold (50)")
    ax.set_title("Attendance % vs Final Marks", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Attendance (%)", fontsize=10)
    ax.set_ylabel("Final Marks", fontsize=10)
    if hue_col:
        ax.legend(title="At Risk", labels=["No", "Yes"])
    plt.tight_layout()
    return fig

def plot_study_hours_vs_marks(df: pd.DataFrame) -> plt.Figure:
    """Plots Weekly Study Hours vs Final Marks with trend line."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.regplot(
        data=df,
        x="study_hours",
        y="final_marks",
        scatter_kws={'alpha': 0.6, 'color': PRIMARY_COLOR},
        line_kws={'color': '#1a365d', 'linewidth': 2},
        ax=ax
    )
    
    ax.set_title("Study Hours vs Final Marks (Linear Fit)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Study Hours / Week", fontsize=10)
    ax.set_ylabel("Final Marks", fontsize=10)
    plt.tight_layout()
    return fig

def plot_marks_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots KDE and histogram of Final Marks."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sns.histplot(
        df["final_marks"],
        kde=True,
        color=PRIMARY_COLOR,
        bins=25,
        ax=ax
    )
    
    mean_val = df["final_marks"].mean()
    median_val = df["final_marks"].median()
    
    ax.axvline(mean_val, color=SECONDARY_COLOR, linestyle="-", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color=ACCENT_COLOR, linestyle="--", linewidth=1.5, label=f"Median: {median_val:.1f}")
    
    ax.set_title("Distribution of Final Marks", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Final Marks", fontsize=10)
    ax.set_ylabel("Student Count", fontsize=10)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Renders correlation heatmap for key features."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

def plot_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    """Plots breakdown of At-Risk vs Safe students."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if "at_risk" in df.columns:
        counts = df["at_risk"].value_counts().rename(index={0: "Safe (Not At Risk)", 1: "At Risk"})
    else:
        counts = (df["final_marks"] < 50).astype(int).value_counts().rename(index={0: "Safe (Not At Risk)", 1: "At Risk"})
        
    colors = ["#38a169", "#e53e3e"]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    ax.set_title("Student Risk Categorization Count", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Count", fontsize=10)
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_names: list, importances: list) -> plt.Figure:
    """Plots feature importances for classification model."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    sorted_indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_imp = [importances[i] for i in sorted_indices]
    
    ax.barh(sorted_features, sorted_imp, color=PRIMARY_COLOR)
    ax.set_title("Model Feature Importance", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm: np.ndarray, labels: list = ["Safe", "At-Risk"]) -> plt.Figure:
    """Plots confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title("Classifier Confusion Matrix", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    plt.tight_layout()
    return fig
