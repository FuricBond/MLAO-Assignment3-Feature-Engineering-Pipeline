"""
generate_dataset.py
===================
Generates a realistic synthetic IBM HR Analytics Employee Attrition
dataset and saves it to data/employee_attrition.csv.

Run this ONLY if you cannot download the original dataset from Kaggle.

Usage:
    cd Feature_Engineering_Pipeline
    python src/generate_dataset.py

Note:
    The original IBM dataset is available at:
    https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
    Download it and place it as data/employee_attrition.csv for best results.
"""

import os
import numpy as np
import pandas as pd

# Reproducible synthetic data
RANDOM_SEED = 42
N_ROWS = 1470   # same as original IBM dataset

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Categorical option pools (matching the original IBM HR dataset exactly)
# ---------------------------------------------------------------------------
BUSINESS_TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
EDUCATION_FIELDS = [
    "Human Resources", "Life Sciences", "Marketing",
    "Medical", "Other", "Technical Degree",
]
GENDERS = ["Female", "Male"]
JOB_ROLES = [
    "Healthcare Representative", "Human Resources",
    "Laboratory Technician", "Manager",
    "Manufacturing Director", "Research Director",
    "Research Scientist", "Sales Executive", "Sales Representative",
]
MARITAL_STATUSES = ["Divorced", "Married", "Single"]
OVERTIME = ["No", "Yes"]
OVER18 = ["Y"]

# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------
def generate_ibm_hr_dataset(n: int = N_ROWS) -> pd.DataFrame:
    """Generate n rows of synthetic IBM HR Attrition data."""

    # ── Numerical columns ─────────────────────────────────────────────────
    age              = rng.integers(18, 61, n)
    daily_rate       = rng.integers(102, 1500, n)
    distance         = rng.integers(1, 30, n)
    education        = rng.integers(1, 6, n)
    env_sat          = rng.integers(1, 5, n)
    hourly_rate      = rng.integers(30, 101, n)
    job_involve      = rng.integers(1, 5, n)
    job_level        = rng.integers(1, 6, n)
    job_sat          = rng.integers(1, 5, n)
    monthly_income   = rng.integers(1009, 20000, n)
    monthly_rate     = rng.integers(2094, 27000, n)
    num_companies    = rng.integers(0, 10, n)
    pct_hike         = rng.integers(11, 26, n)
    perf_rating      = rng.choice([3, 4], n, p=[0.85, 0.15])
    rel_sat          = rng.integers(1, 5, n)
    stock_level      = rng.integers(0, 4, n)
    total_work_years = rng.integers(0, 41, n)
    training_times   = rng.integers(0, 7, n)
    wlb              = rng.integers(1, 5, n)
    years_at_co      = rng.integers(0, 41, n)
    years_in_role    = rng.integers(0, 19, n)
    years_since_promo= rng.integers(0, 16, n)
    years_mgr        = rng.integers(0, 18, n)
    employee_count   = np.ones(n, dtype=int)
    employee_number  = np.arange(1, n + 1)
    standard_hours   = np.full(n, 80)

    # ── Categorical columns ───────────────────────────────────────────────
    business_travel = rng.choice(BUSINESS_TRAVEL, n, p=[0.10, 0.71, 0.19])
    department      = rng.choice(DEPARTMENTS, n, p=[0.06, 0.65, 0.29])
    edu_field       = rng.choice(EDUCATION_FIELDS, n,
                                  p=[0.04, 0.41, 0.10, 0.27, 0.05, 0.13])
    gender          = rng.choice(GENDERS, n, p=[0.40, 0.60])
    job_role        = rng.choice(JOB_ROLES, n)
    marital_status  = rng.choice(MARITAL_STATUSES, n, p=[0.23, 0.45, 0.32])
    overtime        = rng.choice(OVERTIME, n, p=[0.72, 0.28])
    over18          = rng.choice(OVER18, n)

    # ── Target: Attrition (~16% attrition rate to match original) ─────────
    attrition_prob  = 0.16
    attrition_raw   = rng.random(n)

    # Increase probability for overtime and low job satisfaction
    overtime_mask   = overtime == "Yes"
    low_sat_mask    = job_sat <= 2
    attrition_raw[overtime_mask]  -= 0.10   # overtime → more attrition
    attrition_raw[low_sat_mask]   -= 0.08

    attrition = np.where(attrition_raw < attrition_prob, "Yes", "No")

    # ── Assemble DataFrame (exact column order as original IBM HR) ─────────
    df = pd.DataFrame({
        "Age":                     age,
        "Attrition":               attrition,
        "BusinessTravel":          business_travel,
        "DailyRate":               daily_rate,
        "Department":              department,
        "DistanceFromHome":        distance,
        "Education":               education,
        "EducationField":          edu_field,
        "EmployeeCount":           employee_count,
        "EmployeeNumber":          employee_number,
        "EnvironmentSatisfaction": env_sat,
        "Gender":                  gender,
        "HourlyRate":              hourly_rate,
        "JobInvolvement":          job_involve,
        "JobLevel":                job_level,
        "JobRole":                 job_role,
        "JobSatisfaction":         job_sat,
        "MaritalStatus":           marital_status,
        "MonthlyIncome":           monthly_income,
        "MonthlyRate":             monthly_rate,
        "NumCompaniesWorked":      num_companies,
        "Over18":                  over18,
        "OverTime":                overtime,
        "PercentSalaryHike":       pct_hike,
        "PerformanceRating":       perf_rating,
        "RelationshipSatisfaction":rel_sat,
        "StandardHours":           standard_hours,
        "StockOptionLevel":        stock_level,
        "TotalWorkingYears":       total_work_years,
        "TrainingTimesLastYear":   training_times,
        "WorkLifeBalance":         wlb,
        "YearsAtCompany":          years_at_co,
        "YearsInCurrentRole":      years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager":    years_mgr,
    })
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base_dir, "data", "employee_attrition.csv")

    print("Generating synthetic IBM HR Attrition dataset ...")
    df = generate_ibm_hr_dataset(N_ROWS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print("[OK] Dataset saved -> " + out_path)
    print(f"   Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"   Attrition distribution:\n{df['Attrition'].value_counts()}")
