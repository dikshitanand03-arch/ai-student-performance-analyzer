"""
app.py
------
Streamlit Interactive Web Application for AI-Powered Student Performance Analyzer.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import (
    load_raw_data, get_data_quality_summary, preprocess_pipeline, REQUIRED_COLUMNS
)
from src.analysis import (
    calculate_summary_statistics, calculate_correlations, calculate_risk_summary
)
from src.visualization import (
    plot_attendance_vs_marks, plot_study_hours_vs_marks, plot_marks_distribution,
    plot_correlation_heatmap, plot_risk_distribution, plot_feature_importance,
    plot_confusion_matrix
)
from src.train import train_and_evaluate_models
from src.prediction import (
    predict_single_student, batch_predict, load_saved_models
)
from src.ai_explainer import generate_ai_explanation

# Configure Page Setup
st.set_page_config(
    page_title="AI Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #3182CE;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card-risk {
        background-color: #FFF5F5;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #E53E3E;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #E53E3E;
        font-weight: bold;
    }
    .risk-safe {
        color: #38A169;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_dataset() -> pd.DataFrame:
    """Loads or generates default dataset."""
    default_path = os.path.join("data", "students.csv")
    if not os.path.exists(default_path):
        from generate_dataset import main as gen_main
        gen_main()
    return load_raw_data(default_path)


def main():
    # Sidebar Setup
    st.sidebar.image("https://img.icons8.com/illustrations/200/education.png", width=120)
    st.sidebar.title("🎓 Analyzer Control")
    st.sidebar.markdown("---")
    
    # 1. Dataset Source Selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Use Default Dataset (students.csv)", "Upload Custom CSV File"]
    )
    
    raw_df = None
    if data_source == "Upload Custom CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
                if missing_cols:
                    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
                    st.stop()
                st.sidebar.success("Custom CSV loaded successfully!")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.stop()
        else:
            st.info("Please upload a CSV file to proceed.")
            st.stop()
    else:
        raw_df = load_default_dataset()
        
    # 2. Risk Threshold Control
    risk_threshold = st.sidebar.slider(
        "At-Risk Threshold (Marks)",
        min_value=35.0,
        max_value=70.0,
        value=50.0,
        step=1.0,
        help="Students with predicted final marks below this threshold will be flagged as At-Risk."
    )

    # 3. Optional OpenAI Key
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 AI Explanation Settings")
    openai_key_input = st.sidebar.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Provide your OpenAI key for LLM advisory notes. If left empty, fallback heuristic advice is used."
    )
    
    # Preprocess Data
    df_processed = preprocess_pipeline(raw_df, risk_threshold=risk_threshold)
    data_quality = get_data_quality_summary(raw_df)
    risk_summary = calculate_risk_summary(df_processed, risk_threshold=risk_threshold)
    
    # Header Banner
    st.markdown('<div class="main-title">🎓 Student Performance & Risk Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Machine Learning Analytics, Performance Forecasting, and Risk Detection</div>', unsafe_allow_html=True)
    
    # High-level Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#4A5568;">Total Students</h4>
            <h2 style="margin:0; color:#2B6CB0;">{risk_summary['total_students']}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_marks = df_processed["final_marks"].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color:#4A5568;">Average Final Mark</h4>
            <h2 style="margin:0; color:#2B6CB0;">{avg_marks:.1f} / 100</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card-risk">
            <h4 style="margin:0; color:#4A5568;">At-Risk Students</h4>
            <h2 style="margin:0; color:#E53E3E;">{risk_summary['at_risk_count']}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card-risk">
            <h4 style="margin:0; color:#4A5568;">At-Risk Rate</h4>
            <h2 style="margin:0; color:#E53E3E;">{risk_summary['at_risk_percentage']}%</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview & Quality",
        "📈 Exploratory Data Analysis",
        "🤖 ML Models & Evaluation",
        "🎯 Individual Student Diagnostics"
    ])

    # -------------------------------------------------------------
    # TAB 1: DATA OVERVIEW & QUALITY
    # -------------------------------------------------------------
    with tab1:
        st.subheader("📋 Dataset Inspection & Preprocessing Summary")
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("##### Raw Dataset Preview")
            st.dataframe(raw_df.head(10), use_container_width=True)
            
        with col_right:
            st.markdown("##### Data Quality Audit")
            st.json({
                "Total Records": data_quality["num_rows"],
                "Total Columns": data_quality["num_columns"],
                "Missing Cells Imputed": data_quality["total_missing_cells"],
                "Missing Columns": data_quality["missing_values"] if data_quality["missing_values"] else "None (Clean)"
            })
            
        st.markdown("---")
        st.subheader("📐 Summary Statistics (Cleaned Data)")
        stats_df = calculate_summary_statistics(df_processed)
        st.dataframe(stats_df, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------
    with tab2:
        st.subheader("📈 Statistical Relationships & Visualizations")
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.pyplot(plot_attendance_vs_marks(df_processed))
        with col_eda2:
            st.pyplot(plot_study_hours_vs_marks(df_processed))
            
        st.markdown("<br>", unsafe_allow_dict=True)
        col_eda3, col_eda4 = st.columns(2)
        with col_eda3:
            st.pyplot(plot_marks_distribution(df_processed))
        with col_eda4:
            corr_matrix = calculate_correlations(df_processed)
            st.pyplot(plot_correlation_heatmap(corr_matrix))

    # -------------------------------------------------------------
    # TAB 3: MACHINE LEARNING MODELS & EVALUATION
    # -------------------------------------------------------------
    with tab3:
        st.subheader("🤖 Model Benchmarking & Performance Comparison")
        
        # Ensure models are trained and load live metrics
        models_dir = "models"
        train_results = train_and_evaluate_models(models_dir=models_dir)
        reg_art, cls_art = load_saved_models(models_dir)
        
        reg_metrics = train_results["regression_metrics"]
        cls_metrics = train_results["classification_metrics"]
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### 1. Regression Models (Predicting Final Marks)")
            st.info(f"Selected Best Regressor: **{train_results['best_reg_model']}**")
            
            reg_df = pd.DataFrame(reg_metrics).T[["MAE", "RMSE", "R2"]]
            reg_df.columns = ["MAE (lower is better)", "RMSE (lower is better)", "R² Score (higher is better)"]
            st.table(reg_df)
            
        with col_m2:
            st.markdown("#### 2. Risk Classification Models (At-Risk Status)")
            st.info(f"Selected Best Classifier: **{train_results['best_cls_model']}**")
            
            cls_df = pd.DataFrame({
                m_name: {
                    "Accuracy": m_vals["Accuracy"],
                    "Precision": m_vals["Precision"],
                    "Recall": m_vals["Recall"],
                    "F1-Score": m_vals["F1"],
                    "5-Fold CV Accuracy": m_vals["CV Accuracy"]
                } for m_name, m_vals in cls_metrics.items()
            }).T
            st.table(cls_df)

        st.markdown("---")
        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            if "feature_importances" in cls_art:
                st.pyplot(plot_feature_importance(
                    cls_art["feature_names"], cls_art["feature_importances"]
                ))
        with col_imp2:
            # Display live Confusion Matrix for selected classifier
            best_cls_name = train_results['best_cls_model']
            cm_matrix = np.array(cls_metrics[best_cls_name]["Confusion Matrix"])
            st.pyplot(plot_confusion_matrix(cm_matrix))

    # -------------------------------------------------------------
    # TAB 4: INDIVIDUAL STUDENT DIAGNOSTICS & AI ADVISORY
    # -------------------------------------------------------------
    with tab4:
        st.subheader("🎯 Individual Student Prediction & AI Intervention")
        
        analysis_mode = st.radio(
            "Select Input Mode:",
            ["Select Student from Dataset", "Manual Metric Sliders"],
            horizontal=True
        )
        
        student_input = {}
        
        if analysis_mode == "Select Student from Dataset":
            selected_id = st.selectbox(
                "Select Student ID:",
                options=df_processed["student_id"].values
            )
            student_row = df_processed[df_processed["student_id"] == selected_id].iloc[0]
            student_input = {
                "student_id": int(student_row["student_id"]),
                "attendance": float(student_row["attendance"]),
                "study_hours": float(student_row["study_hours"]),
                "assignment_score": float(student_row["assignment_score"]),
                "previous_marks": float(student_row["previous_marks"]),
                "sleep_hours": float(student_row["sleep_hours"]),
                "participation": int(student_row["participation"])
            }
        else:
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                att = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
                study = st.slider("Study Hours/Week", 0.0, 30.0, 10.0)
            with col_in2:
                prev = st.slider("Previous Marks (0-100)", 0.0, 100.0, 65.0)
                assign = st.slider("Assignment Score (0-100)", 0.0, 100.0, 70.0)
            with col_in3:
                sleep = st.slider("Sleep Hours/Day", 4.0, 10.0, 7.0)
                part = st.slider("Participation Scale (1-5)", 1, 5, 3)
                
            student_input = {
                "student_id": 9999,
                "attendance": att,
                "study_hours": study,
                "assignment_score": assign,
                "previous_marks": prev,
                "sleep_hours": sleep,
                "participation": part
            }

        # Run Live Prediction
        prediction_res = predict_single_student(student_input, models_dir="models")
        
        st.markdown("---")
        st.markdown("### 📊 Diagnostic Results")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric(
                label="Predicted Final Mark",
                value=f"{prediction_res['predicted_final_marks']} / 100"
            )
        with col_res2:
            st.metric(
                label="Risk Probability",
                value=f"{prediction_res['risk_probability']}%"
            )
        with col_res3:
            st.metric(
                label="Risk Category",
                value=prediction_res['risk_level']
            )

        st.markdown("##### 🚨 Primary Risk Contributing Factors")
        for factor in prediction_res["contributing_factors"]:
            if "No major" in factor:
                st.success(f"✅ {factor}")
            else:
                st.warning(f"⚠️ {factor}")
                
        st.markdown("<br>", unsafe_allow_dict=True)
        st.markdown("### 🤖 AI Mentor Explanation & Actionable Advice")
        
        with st.spinner("Generating AI explanation..."):
            ai_advice = generate_ai_explanation(
                student_input,
                prediction_res,
                api_key=openai_key_input
            )
            st.info(ai_advice)

if __name__ == "__main__":
    main()
