"""
inference.py
============
Handles STEP 8 — reusing the saved pipeline to transform new,
unseen employee records in a production-like manner.

Author  : MLOps Assignment
Dataset : IBM HR Analytics Employee Attrition & Performance
"""

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60
SUBSEP = "-" * 60


# ===========================================================================
# STEP 8 – TRANSFORM NEW DATA
# ===========================================================================


def create_sample_employees() -> pd.DataFrame:
    """
    Create a small DataFrame of synthetic new employee records that
    mimic the schema of the IBM HR Attrition dataset.

    These records have NOT been seen by the pipeline during training
    and simulate real-world inference requests.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 5 new employee records.
    """
    new_employees = {
        # Numerical features
        "Age":                       [28, 45, 33, 55, 24],
        "DailyRate":                 [802, 1250, 650, 980, 450],
        "DistanceFromHome":          [5, 20, 3, 15, 30],
        "Education":                 [3, 4, 2, 5, 1],
        "EnvironmentSatisfaction":   [3, 2, 4, 1, 3],
        "HourlyRate":                [65, 90, 55, 75, 40],
        "JobInvolvement":            [3, 2, 4, 3, 1],
        "JobLevel":                  [2, 4, 1, 5, 1],
        "JobSatisfaction":           [4, 1, 3, 2, 4],
        "MonthlyIncome":             [5000, 12000, 3500, 18000, 2500],
        "MonthlyRate":               [14000, 22000, 10000, 25000, 8000],
        "NumCompaniesWorked":        [2, 5, 1, 8, 0],
        "PercentSalaryHike":         [15, 20, 12, 25, 11],
        "PerformanceRating":         [3, 4, 3, 4, 3],
        "RelationshipSatisfaction":  [3, 2, 4, 1, 3],
        "StockOptionLevel":          [1, 2, 0, 3, 0],
        "TotalWorkingYears":         [5, 20, 8, 30, 1],
        "TrainingTimesLastYear":     [3, 2, 4, 1, 6],
        "WorkLifeBalance":           [3, 2, 4, 3, 1],
        "YearsAtCompany":            [3, 15, 5, 20, 1],
        "YearsInCurrentRole":        [2, 8, 3, 10, 0],
        "YearsSinceLastPromotion":   [1, 5, 2, 8, 0],
        "YearsWithCurrManager":      [2, 10, 3, 15, 0],
        # Categorical features
        "BusinessTravel":            ["Travel_Rarely", "Travel_Frequently",
                                      "Non-Travel", "Travel_Rarely",
                                      "Travel_Frequently"],
        "Department":                ["Sales", "Research & Development",
                                      "Human Resources", "Sales",
                                      "Research & Development"],
        "EducationField":            ["Life Sciences", "Medical",
                                      "Marketing", "Technical Degree",
                                      "Human Resources"],
        "Gender":                    ["Male", "Female", "Male",
                                      "Female", "Male"],
        "JobRole":                   ["Sales Executive",
                                      "Research Scientist",
                                      "Human Resources",
                                      "Manager",
                                      "Laboratory Technician"],
        "MaritalStatus":             ["Single", "Married", "Divorced",
                                      "Married", "Single"],
        "OverTime":                  ["Yes", "No", "No", "Yes", "Yes"],
    }
    df = pd.DataFrame(new_employees)
    return df


def transform_new_data(
    pipeline: Pipeline,
    new_data: pd.DataFrame,
) -> np.ndarray:
    """
    Apply the loaded pipeline to a new batch of employee records.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted preprocessing pipeline loaded from disk.
    new_data : pd.DataFrame
        New employee records with the same schema as training data.

    Returns
    -------
    np.ndarray
        Transformed feature matrix ready for model inference.
    """
    logger.info(
        "Transforming %d new employee records …", len(new_data)
    )

    print(f"\n{SEPARATOR}")
    print("  🔮  INFERENCE — NEW EMPLOYEE DATA")
    print(SEPARATOR)

    print(f"\n🔷 New Employee Records ({len(new_data)} rows):")
    print(SUBSEP)
    print(new_data.to_string())

    # Transform using the loaded pipeline (no fitting!)
    X_new_transformed: np.ndarray = pipeline.transform(new_data)

    print(f"\n{SEPARATOR}")
    print("  ✅  TRANSFORMATION COMPLETE")
    print(SEPARATOR)
    print(f"\n   Input  Shape : {new_data.shape}")
    print(f"   Output Shape : {X_new_transformed.shape}")

    print(f"\n🔷 Transformed Output (first 3 rows, first 10 features):")
    print(SUBSEP)

    # Pretty-print a slice of the output
    display_arr = X_new_transformed[:3, :10]
    display_df = pd.DataFrame(
        display_arr,
        columns=[f"Feature_{i+1}" for i in range(display_arr.shape[1])],
    )
    print(display_df.round(4).to_string())

    print(
        f"\n   Full output: {X_new_transformed.shape[0]} rows × "
        f"{X_new_transformed.shape[1]} selected features"
    )

    logger.info(
        "New data transformed. Output shape: %s", X_new_transformed.shape
    )
    return X_new_transformed
