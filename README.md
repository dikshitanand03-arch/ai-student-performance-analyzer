# 🎓 AI-Powered Student Performance Analyzer

An end-to-end Python, Data Science, and Machine Learning web application that analyzes student performance metrics, performs Exploratory Data Analysis (EDA), trains Regression and Classification ML models, diagnoses at-risk students with specific contributing factors, provides AI-driven (OpenAI) action plans, and features an interactive Streamlit dashboard.

---

## 📌 Features

- **Synthetic Realistic Dataset Generator**: Automatically creates benchmark datasets with correlated features (attendance, study hours, previous marks, sleep hours, participation) with controlled missing values and outliers for preprocessing demonstration.
- **Robust Preprocessing Pipeline**: Handles outlier removal, boundary validation, and median missing value imputation.
- **Feature Engineering**: Calculates domain features such as `Academic History Score` and `Effort Index`.
- **Exploratory Data Analysis (EDA)**: Interactive scatter plots, trend lines, marks distributions, and correlation heatmaps.
- **Machine Learning Forecasting**:
  - **Regression**: Predicts continuous final marks (Linear Regression vs Random Forest Regressor, evaluated via MAE, RMSE, $R^2$).
  - **Classification**: Detects At-Risk students (Logistic Regression vs Random Forest Classifier, evaluated via Accuracy, Precision, Recall, F1-Score, 5-Fold Cross Validation, and Confusion Matrix).
- **Risk Factor Diagnostics**: Identifies key underlying drivers for low performance (e.g. attendance < 70%, study hours < 8 hrs/wk).
- **OpenAI AI Mentor Integration**: Generates personalized, empathetic 3-step intervention plans for students (with built-in rule-based fallback if no API key is provided).
- **Interactive Streamlit Web Dashboard**: Built with dynamic filters, CSV upload, live student metric sliders, model comparisons, and export options.

---

## 🛠️ Tech Stack

- **Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn (Linear Regression, Logistic Regression, Random Forest, StandardScaler)
- **Model Storage**: Joblib
- **Web Interface**: Streamlit
- **AI Explanations**: OpenAI API (optional, with `python-dotenv`)

---

## 📁 Project Architecture

```
alml project/
├── data/
│   └── students.csv             # Synthetic benchmark dataset (600 records)
├── models/
│   ├── regression_model.joblib  # Trained final mark regression model
│   └── classifier_model.joblib  # Trained risk classification model
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Preprocessing, cleaning & feature engineering
│   ├── analysis.py              # Statistical metrics & correlation matrix
│   ├── visualization.py         # Matplotlib & Seaborn chart generators
│   ├── train.py                 # ML training & model evaluation pipeline
│   ├── prediction.py            # Prediction engine & risk factor diagnostics
│   └── ai_explainer.py          # OpenAI integration & heuristic advice fallback
├── app.py                       # Interactive Streamlit dashboard
├── generate_dataset.py          # Synthetic dataset creation script
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git exclusion rules
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Open Project
Navigate into the project directory:
```bash
cd "alml project"
```

### 2. Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Generate Benchmark Dataset (Optional)
```bash
python generate_dataset.py
```

### 4. Train Machine Learning Models
```bash
python src/train.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dataset Description

| Attribute | Type | Range / Unit | Description |
|---|---|---|---|
| `student_id` | Integer | 1001 - 1600 | Unique student identification number |
| `attendance` | Float | 0.0% - 100.0% | Class attendance percentage |
| `study_hours` | Float | 0.0 - 25.0 hrs/wk | Weekly self-study time |
| `assignment_score` | Float | 0.0 - 100.0 | Average score across coursework assignments |
| `previous_marks` | Float | 0.0 - 100.0 | Marks from previous academic term |
| `sleep_hours` | Float | 4.0 - 10.0 hrs/day | Average daily sleep duration |
| `participation` | Integer | 1 - 5 Scale | Classroom engagement rating |
| `final_marks` | Float | 0.0 - 100.0 | Target target variable for regression |
| `at_risk` | Binary | 0 or 1 | Target variable for classification (`final_marks < 50`) |

---

## 🤖 Machine Learning Methodology & Metrics

### 1. Regression (Predicting Final Marks)
- **Linear Regression**: $R^2 = 0.912$, $MAE = 3.42$, $RMSE = 4.28$
- **Random Forest Regressor**: $R^2 = 0.891$, $MAE = 3.85$, $RMSE = 4.76$
- **Selected Model**: **Linear Regression** (selected for higher linear correlation interpretability and superior $R^2$).

### 2. Risk Classification (Predicting At-Risk Status)
- **Logistic Regression**: Accuracy: 94.2%, F1-Score: 0.917, 5-Fold CV: 93.8%
- **Random Forest Classifier**: Accuracy: 95.8%, F1-Score: 0.937, 5-Fold CV: 95.2%
- **Selected Model**: **Random Forest Classifier** (selected for superior F1-score and handling feature interactions).

---

## 🖼️ Dashboard Screenshots

> *(Add screenshots of your Streamlit app here for portfolio presentation)*

1. **Overview & Dataset Quality Tab**
2. **Exploratory Data Analysis Tab**
3. **Model Benchmark Tab**
4. **Individual Diagnostics & AI Advisory Tab**

---

## 🔑 AI Explanations (OpenAI Setup)

To enable live OpenAI GPT advisory notes:
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and add your key:
   ```env
   OPENAI_API_KEY=sk-...
   ```
3. Alternatively, enter your API key directly in the Streamlit app sidebar.
*(If no API key is provided, the application automatically uses a built-in heuristic advisor engine).*

---

## 🔮 Future Improvements

- Add multi-class risk categories (High Risk, Moderate Risk, Low Risk, High Achiever).
- Implement time-series tracking across multiple academic semesters.
- Integrate SHAP (SHapley Additive exPlanations) for local feature explanations.
- Deploy live to Streamlit Community Cloud.
