# Credit Risk Analysis
### Predicting Credit Card Default Using Polynomial Logistic Regression

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.0-orange)
![Plotly](https://img.shields.io/badge/Plotly-Dash-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Project Overview
This project predicts whether a credit card client will default on their
next payment using real banking data from 30,000 clients.

The model achieves **AUC = 0.7551** and detects **60% of default cases**,
potentially preventing **NT$39,800,000** in losses.

---

## Dataset
- **Source:** UCI Machine Learning Repository via Kaggle
- **Link:** [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Size:** 30,000 clients, 24 features
- **Target:** Default payment next month (0 = No, 1 = Yes)
- **Default Rate:** 22.1%

---

## Project Pipeline

| Step | Description |
|------|-------------|
| 1 | Data Loading & Exploration (EDA) |
| 2 | Feature Engineering & Preprocessing |
| 3 | Assumption Testing (Shapiro-Wilk) |
| 4 | Model Building (Degree 1 vs Degree 2) |
| 5 | Evaluation (AUC, Confusion Matrix, ROC) |
| 6 | Interactive Dashboard (Plotly Dash) |

---

## Model Results

| Metric | Degree 1 (Linear) | Degree 2 (Best) |
|--------|-------------------|-----------------|
| Accuracy | 67.97% | 75.03% |
| AUC Score | 0.7081 | 0.7551 |
| Default Detection | — | 60.0% |

---

## Business Impact