# Student Dropout Prediction System

A machine learning system for early detection of students at risk of academic failure,
designed to identify at-risk profiles before the first evaluations using only
socio-behavioral data.

---

## Project Overview

**Problem**: Traditional dropout detection relies on grade-based indicators,
which only become available after failure has already occurred.

**Approach**: Predict dropout risk using exclusively socio-behavioral variables
(family context, attendance, study habits, social behavior), enabling intervention
before the first exam.

**Key decision**: Deliberate exclusion of G1 and G2 grade variables to ensure
genuinely early detection capability.

---

## Model Performance

| Metric    | Value  |
|-----------|--------|
| Recall    | 80.77% |
| Precision | 44.68% |
| F1-Score  | 0.578  |
| AUC-ROC   | 0.734  |
| Threshold | 0.399  |

**Design rationale**: The model is optimized for recall rather than precision.
In an early warning context, the cost of missing an at-risk student
significantly exceeds the cost of a false alert.

---

## Dataset

- **Source**: Cortez & Silva (2008), UCI Machine Learning Repository
- **Size**: 395 students, 34 variables
- **Target**: Binary classification (fail if final grade < 10/20)
- **Features**: Academic history, family background, behavioral indicators

---

## Feature Engineering

Three composite variables were created to improve predictive power:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| parent_edu_mean | (Medu + Fedu) / 2 | Combined parental education level |
| total_alcohol | Dalc + Walc | Weekly alcohol consumption index |
| total_support | schoolsup + famsup | Cumulative academic support |
| high_risk | failures > 0 AND absences > 10 | Combined risk flag |

---

Project1-Dropout/

|-- app.py

|-- agent/

|   |-- agent.py

|   |-- alerter.py

|   |-- evaluator.py

|   |-- monitor.py

|   └-- researcher.py

|-- models/

|   |-- logistic_regression_final.pkl

|   |-- scaler.pkl

|   |-- feature_names.json

|   └-- config.json

|-- data/

|   |-- X_train.csv

|   |-- X_test.csv

|   |-- y_train.csv

|   └-- y_test.csv

└-- notebooks/

└-- 01_exploration.ipynb
---
## Autonomous Agent System

Beyond the prediction model, this project includes a multi-component
autonomous monitoring system:

**Agent** (`agent.py`)
Reads incoming student data, generates predictions, and sends
email alerts for at-risk profiles without human intervention.

**Evaluator** (`evaluator.py`)
Monitors model performance weekly. If recall drops below 75%,
automatically retrains the model on updated data and saves
the improved version.

**Researcher** (`researcher.py`)
Scrapes recent academic publications on dropout prediction,
analyzes trends and research gaps using an LLM, and generates
a structured research report to inform model improvements.

**Monitor** (`monitor.py`)
Orchestrates the full weekly routine: literature review,
model evaluation, and student analysis.

---

## Deployed Application

Live demo: https://student-dropout-prediction-fxj3vscls4vnvm9hpj2263.streamlit.app

The interface allows educators to input student profiles and receive:
- Risk probability score
- Alert classification (at-risk / stable)
- Identified risk factors
- Recommended interventions

---

## Technical Stack

- Python 3.11
- scikit-learn (Logistic Regression, StandardScaler)
- pandas, numpy
- Streamlit (deployment)
- SHAP (feature importance)
- schedule (automated agent)
- arxiv, groq (research agent)

---

## Limitations

- Dataset limited to 395 students from Portuguese secondary schools (2008)
- 19% of dropout cases not detected (atypical profiles)
- 55% false alert rate (accepted trade-off)
- Model requires revalidation before deployment in different educational contexts

---

## References

Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School
Student Performance. In Proceedings of 5th Annual Future Business Technology
Conference, Porto, Portugal.

---

## Author

Justin Kelem
Master AI/Data Science Candidate 2027
GitHub: justinkelem708-ops