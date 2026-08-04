# Feature Engineering Pipeline for Machine Learning

> **MLOps Assignment — IBM HR Analytics Employee Attrition & Performance Dataset**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

This project implements a **production-quality, reusable Feature Engineering Pipeline** designed to automatically transform raw human resources data into machine-learning-ready features. The pipeline is built using industry-standard tools — **Scikit-Learn Pipelines** and **ColumnTransformer** — ensuring zero data leakage, reproducibility, and easy deployment.

---

## 🎯 Objective

Design and implement a modular, end-to-end Feature Engineering Pipeline that:

- Loads and explores the IBM HR Attrition dataset
- Cleans the data (handles missing values and duplicates)
- Engineers features using imputation, scaling, and encoding
- Selects the most predictive features via **SelectKBest**
- Saves the pipeline for reuse on unseen data

---

## 📂 Dataset

| Attribute         | Value                                           |
|-------------------|-------------------------------------------------|
| **Name**          | IBM HR Analytics Employee Attrition & Performance |
| **Source**        | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Rows**          | 1,470                                           |
| **Columns**       | 35                                              |
| **Target**        | `Attrition` (Yes / No)                          |
| **Task**          | Binary Classification                           |

---

## 🗂️ Folder Structure

```
Feature_Engineering_Pipeline/
│
├── data/
│   └── employee_attrition.csv        ← Raw dataset (place here)
│
├── src/
│   ├── preprocessing.py              ← Step 1 & 2: Load, explore, clean
│   ├── feature_selection.py          ← Step 4: SelectKBest analysis
│   ├── pipeline.py                   ← Step 3, 5, 6, 7: Build & transform
│   ├── inference.py                  ← Step 8: Load & reuse pipeline
│   └── main.py                       ← Orchestrates all steps
│
├── models/
│   └── feature_pipeline.pkl          ← Saved fitted pipeline (joblib)
│
├── output/
│   ├── transformed_data.csv          ← Transformed feature matrix
│   ├── missing_value_heatmap.png     ← Missing value visualisation
│   ├── correlation_heatmap.png       ← Feature correlation matrix
│   └── feature_importance.png        ← SelectKBest bar chart
│
├── screenshots/                      ← Assignment screenshot captures
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone / Download the project

```bash
git clone <repo-url>
cd Feature_Engineering_Pipeline
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` from [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and save it as:

```
data/employee_attrition.csv
```

---

## ▶️ How to Run

```bash
cd src
python main.py
```

The script will walk through all 8 steps, pausing at each screenshot milestone and prompting you to press **Enter** to continue.

---

## 🔄 Pipeline Architecture

```
Raw DataFrame
      │
      ▼
ColumnTransformer
  ├── Numerical Branch
  │     ├── SimpleImputer (mean)
  │     └── StandardScaler
  └── Categorical Branch
        ├── SimpleImputer (most_frequent)
        └── OneHotEncoder
      │
      ▼
SelectKBest (f_classif, k=20)
      │
      ▼
Transformed Feature Matrix (1470 × 20)
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `output/transformed_data.csv` | Feature-engineered dataset (1470 × 21 incl. target) |
| `output/missing_value_heatmap.png` | Heatmap showing NaN locations |
| `output/correlation_heatmap.png` | Pearson correlation matrix of numerical features |
| `output/feature_importance.png` | SelectKBest F-score bar chart |
| `models/feature_pipeline.pkl` | Serialised fitted pipeline (joblib) |

---

## 📸 Required Screenshots

| # | Title |
|---|-------|
| 1 | Project Folder Structure |
| 2 | Dataset Loaded Successfully |
| 3 | Missing Value Analysis |
| 4 | Data Cleaning Completed |
| 5 | Numerical & Categorical Features |
| 6 | Feature Engineering Completed |
| 7 | Feature Selection Results |
| 8 | Pipeline Created Successfully |
| 9 | Original Shape vs Transformed Shape |
| 10 | Transformed Dataset Saved |
| 11 | Pipeline Saved Successfully |
| 12 | Pipeline Loaded Successfully |
| 13 | New Employee Data Transformed |
| 14 | Output Folder Contents |
| 15 | Final Project Structure |

---

## 🏗️ Module Descriptions

### `preprocessing.py`
- `load_data()` — reads CSV using `pd.read_csv`
- `display_dataset_info()` — shape, first 5 rows, dtypes, stats
- `identify_features()` — separates numerical and categorical columns
- `inject_missing_values()` — programmatically creates NaN for demo
- `clean_data()` — dedup + missing value display before/after
- `plot_missing_value_heatmap()` — seaborn heatmap
- `plot_correlation_heatmap()` — Pearson correlation matrix

### `feature_selection.py`
- `perform_feature_selection()` — SelectKBest with f_classif
- `get_selector_for_pipeline()` — returns selector for embedding in Pipeline

### `pipeline.py`
- `create_preprocessing_pipeline()` — builds ColumnTransformer + SelectKBest
- `transform_dataset()` — fits and transforms training data
- `save_pipeline()` — serialises pipeline with joblib
- `load_pipeline()` — deserialises pipeline from disk

### `inference.py`
- `create_sample_employees()` — generates 5 synthetic employee records
- `transform_new_data()` — applies loaded pipeline for inference

### `main.py`
- `main()` — orchestrates all 8 steps with screenshot prompts

---

## 🚀 Future Scope

| Enhancement | Description |
|-------------|-------------|
| **MLflow Integration** | Log pipeline parameters, metrics, and artefacts |
| **Model Training** | Extend pipeline with RandomForest / XGBoost classifier |
| **Hyperparameter Tuning** | Use `GridSearchCV` over pipeline params |
| **REST API Deployment** | Wrap inference in a FastAPI / Flask endpoint |
| **Docker Container** | Containerise the pipeline for reproducible deployment |
| **CI/CD Pipeline** | GitHub Actions for automated testing and retraining |
| **Monitoring** | Data drift detection using Evidently AI |
| **Feature Store** | Integrate with Feast for scalable feature management |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.13 | Core language |
| Pandas | ≥ 2.2 | Data manipulation |
| NumPy | ≥ 1.26 | Numerical operations |
| Scikit-Learn | ≥ 1.4 | Pipeline & transformers |
| Matplotlib | ≥ 3.8 | Plotting |
| Seaborn | ≥ 0.13 | Statistical visualisation |
| Joblib | ≥ 1.3 | Model serialisation |

---

## 👤 Author

**MLOps Assignment — Third Year Engineering**  
IBM HR Analytics Feature Engineering Pipeline

---

*Built with ❤️ for the MLOps Assignment*
