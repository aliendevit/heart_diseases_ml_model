Trained and evaluated both Logistic Regression and Random Forest on a heart disease dataset using 5-fold stratified cross-validation.

Logistic Regression achieved 0.83 ± 0.05 accuracy, while Random Forest achieved 0.83 ± 0.05 accuracy with a slightly lower mean.

Since Random Forest did not significantly outperform Logistic Regression, I selected Logistic Regression as the final model for its simplicity and interpretability.
# Heart Disease Prediction – Logistic Regression (ML Week 1 Project)

## Overview

This project is a small end-to-end machine learning pipeline for predicting heart disease
(0 = no disease, 1 = disease) from tabular clinical data.

It includes:

- Exploratory Data Analysis (EDA) in Jupyter notebooks
- A Logistic Regression baseline model
- Cross-validation–based evaluation
- A clean Python package structure (`src/`)
- A training script that saves a model artifact
- A single-patient prediction script that simulates real-world inference

This project is part of my personal ML learning roadmap and will serve as a reference
template for future projects (vision, NLP, GenAI, etc.).

---

## Project Structure

```text
ml_week1_project/
├─ data/
│  └─ heart.csv
├─ models/
│  └─ heart_logreg_model.joblib
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_baseline_model.ipynb
│  ├─ 03_improved_model.ipynb
│  └─ 04_cross_validation.ipynb
└─ src/
   ├─ preprocess.py
   ├─ train_model.py
   └─ predict_single.py
## API (FastAPI)

The project also exposes the trained Logistic Regression model as a simple FastAPI service.

Run the API:

```bash
uvicorn src.api_app:app --reload
