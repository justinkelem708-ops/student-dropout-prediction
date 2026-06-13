---
language:
- en
- fr
license: cc-by-4.0
task_categories:
- text-classification
task_ids:
- binary-classification
tags:
- education
- dropout
- student-performance
- early-warning
pretty_name: Student Dropout Prediction Dataset
size_categories:
- n<1K
---
# Student Dropout Prediction Dataset

## Dataset Description

Preprocessed dataset for early dropout prediction in secondary schools.
Based on the UCI Student Performance Dataset (Cortez & Silva, 2008).

## Dataset Structure

| File | Description | Rows |
|------|-------------|------|
| X_train.csv | Training features | 316 |
| X_test.csv | Test features | 79 |
| y_train.csv | Training labels | 316 |
| y_test.csv | Test labels | 79 |
| student-mat.csv | Original math dataset | 395 |
| student-por.csv | Original portuguese dataset | 649 |

## Features

34 socio-behavioral variables including :
- Academic history (failures, absences, study time)
- Family context (parent education, family support)
- Behavioral indicators (social activity, alcohol consumption)

## Target Variable

Binary classification :
- 0 : Pass (final grade >= 10/20)
- 1 : Fail / Dropout risk (final grade < 10/20)

## Preprocessing

- Categorical variables encoded
- Feature engineering applied (parent_edu_mean, total_alcohol, high_risk, total_support)
- Train/test split : 80/20

## Model Performance

Logistic Regression trained on this dataset achieves :
- Recall : 80.77%
- Precision : 44.68%
- F1-Score : 0.578

## Source

Cortez, P., & Silva, A. (2008).
Using Data Mining to Predict Secondary School Student Performance.
UCI Machine Learning Repository.

## Related Resources

- Prediction app : https://student-dropout-prediction-fxj3vscls4vnvm9hpj2263.streamlit.app
- GitHub : https://github.com/justinkelem708-ops/student-dropout-prediction

## License

CC BY 4.0