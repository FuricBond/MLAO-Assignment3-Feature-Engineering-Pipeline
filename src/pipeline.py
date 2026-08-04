"""
pipeline.py
===========
Builds the complete Scikit-Learn preprocessing pipeline using
Pipeline and ColumnTransformer, incorporating:

  • Numerical pipeline  : SimpleImputer  → StandardScaler
  • Categorical pipeline: SimpleImputer  → OneHotEncoder
  • Feature selection   : SelectKBest (f_classif)

Author  : MLOps Assignment
Dataset : IBM HR Analytics Employee Attrition & Performance
"""

import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from feature_selection import get_selector_for_pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60
SUBSEP = "-" * 60


# ===========================================================================
# STEP 3 + STEP 5 – BUILD PREPROCESSING PIPELINE
# ===========================================================================


def _build_numerical_pipeline() -> Pipeline:
    """
    Construct the numerical feature transformation sub-pipeline.

    Steps
    -----
    1. SimpleImputer (strategy='mean')  — fills NaN with column mean.
    2. StandardScaler                   — zero mean, unit variance.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])


def _build_categorical_pipeline() -> Pipeline:
    """
    Construct the categorical feature transformation sub-pipeline.

    Steps
    -----
    1. SimpleImputer (strategy='most_frequent') — fills NaN with mode.
    2. OneHotEncoder (handle_unknown='ignore') — binary dummies.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ])


def create_preprocessing_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    k_best: int = 20,
) -> Pipeline:
    """
    Assemble the full end-to-end preprocessing pipeline:

        ColumnTransformer
          ├── numerical_pipeline   (SimpleImputer → StandardScaler)
          └── categorical_pipeline (SimpleImputer → OneHotEncoder)
        └── SelectKBest (f_classif, k=k_best)

    Parameters
    ----------
    numerical_features : list[str]
        List of numerical column names.
    categorical_features : list[str]
        List of categorical column names.
    k_best : int
        Number of top features to select via SelectKBest.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The complete, unfitted preprocessing pipeline.
    """
    logger.info(
        "Building pipeline — Numerical: %d, Categorical: %d, k_best: %d",
        len(numerical_features),
        len(categorical_features),
        k_best,
    )

    # Sub-pipelines
    num_pipeline = _build_numerical_pipeline()
    cat_pipeline = _build_categorical_pipeline()

    # ColumnTransformer merges both sub-pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_pipeline, numerical_features),
            ("categorical", cat_pipeline, categorical_features),
        ],
        remainder="drop",       # drop any unlisted columns
    )

    # Full pipeline = preprocessor + feature selection
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", get_selector_for_pipeline(k=k_best)),
    ])

    print(f"\n{SEPARATOR}")
    print("  🏗️   PIPELINE ARCHITECTURE")
    print(SEPARATOR)
    print("""
  ┌─────────────────────────────────────────────────────┐
  │               Full Preprocessing Pipeline            │
  │                                                     │
  │  ┌───────────────────────────────────────────────┐  │
  │  │            ColumnTransformer                   │  │
  │  │                                               │  │
  │  │  ┌──────────────────┐  ┌──────────────────┐  │  │
  │  │  │ Numerical Branch │  │Categorical Branch │  │  │
  │  │  │                  │  │                   │  │  │
  │  │  │ SimpleImputer    │  │ SimpleImputer     │  │  │
  │  │  │  (mean)          │  │  (most_frequent)  │  │  │
  │  │  │      ↓           │  │       ↓           │  │  │
  │  │  │ StandardScaler   │  │ OneHotEncoder     │  │  │
  │  │  └──────────────────┘  └──────────────────┘  │  │
  │  └───────────────────────────────────────────────┘  │
  │                        ↓                            │
  │  ┌───────────────────────────────────────────────┐  │
  │  │     SelectKBest (f_classif, k={k_best})        │  │
  │  └───────────────────────────────────────────────┘  │
  └─────────────────────────────────────────────────────┘
""".replace("{k_best}", str(k_best)))

    logger.info("Pipeline built successfully.")
    return full_pipeline


def get_feature_names_after_preprocessing(
    pipeline: Pipeline,
    numerical_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """
    Retrieve the feature names produced by the ColumnTransformer step
    (i.e., BEFORE SelectKBest filtering).

    Parameters
    ----------
    pipeline : Pipeline
        A fitted pipeline containing a 'preprocessor' ColumnTransformer step.
    numerical_features : list[str]
        Original numerical column names.
    categorical_features : list[str]
        Original categorical column names.

    Returns
    -------
    list[str]
        List of feature names after ColumnTransformer.
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names: list[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "numerical":
            feature_names.extend(cols)
        elif name == "categorical":
            ohe: OneHotEncoder = transformer.named_steps["encoder"]
            feature_names.extend(ohe.get_feature_names_out(cols).tolist())

    return feature_names


def get_selected_feature_names(
    pipeline: Pipeline,
    all_feature_names: list[str],
) -> list[str]:
    """
    Retrieve the feature names that were kept by SelectKBest.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted pipeline containing a 'feature_selection' SelectKBest step.
    all_feature_names : list[str]
        All feature names BEFORE SelectKBest (from ColumnTransformer).

    Returns
    -------
    list[str]
        Names of selected features.
    """
    selector: SelectKBest = pipeline.named_steps["feature_selection"]
    support: np.ndarray = selector.get_support()
    return [name for name, keep in zip(all_feature_names, support) if keep]


# ===========================================================================
# STEP 6 – TRANSFORM DATASET
# ===========================================================================


def transform_dataset(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
    output_path: str = "output/transformed_data.csv",
) -> Tuple[np.ndarray, list[str]]:
    """
    Fit the pipeline on X/y and transform the dataset.
    Save the resulting array as a CSV.

    Parameters
    ----------
    pipeline : Pipeline
        The unfitted preprocessing pipeline.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable (used by SelectKBest).
    numerical_features : list[str]
        Numerical column names.
    categorical_features : list[str]
        Categorical column names.
    output_path : str
        Path to save transformed_data.csv.

    Returns
    -------
    Tuple[np.ndarray, list[str]]
        (X_transformed, selected_feature_names)
    """
    logger.info("Fitting and transforming the dataset …")

    # Fit + transform
    X_transformed: np.ndarray = pipeline.fit_transform(X, y)

    # Retrieve names before and after selection
    all_names = get_feature_names_after_preprocessing(
        pipeline, numerical_features, categorical_features
    )
    selected_names = get_selected_feature_names(pipeline, all_names)

    # ── Display ──────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  📐  DATASET TRANSFORMATION RESULTS")
    print(SEPARATOR)
    print(f"\n   Original  Dataset Shape : {X.shape}")
    print(f"   Transformed Dataset Shape : {X_transformed.shape}")
    print(
        f"\n   Features BEFORE SelectKBest : {len(all_names)}"
        f"\n   Features AFTER  SelectKBest : {len(selected_names)}"
    )
    print(f"\n🔷 Selected Feature Names:")
    print(SUBSEP)
    for i, name in enumerate(selected_names, start=1):
        print(f"   {i:>2}. {name}")

    # ── Save transformed data ─────────────────────────────────────────────────
    _save_transformed_data(X_transformed, selected_names, y, output_path)

    return X_transformed, selected_names


def _save_transformed_data(
    X_transformed: np.ndarray,
    feature_names: list[str],
    y: pd.Series,
    output_path: str,
) -> None:
    """
    Save the transformed feature matrix (with target) to CSV.

    Parameters
    ----------
    X_transformed : np.ndarray
        Transformed feature matrix.
    feature_names : list[str]
        Column names for the transformed matrix.
    y : pd.Series
        Target series appended as the last column.
    output_path : str
        File path for the output CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_out = pd.DataFrame(X_transformed, columns=feature_names)
    df_out["Attrition"] = y.values

    df_out.to_csv(output_path, index=False)
    logger.info("Transformed dataset saved → %s", output_path)
    print(f"\n   ✅  Transformed dataset saved → {output_path}")
    print(f"       Rows: {df_out.shape[0]}  |  Columns: {df_out.shape[1]}")


# ===========================================================================
# STEP 7 – SAVE PIPELINE
# ===========================================================================


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """
    Persist the fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted Scikit-Learn pipeline.
    filepath : str
        Destination path (e.g., 'models/feature_pipeline.pkl').
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)
    logger.info("Pipeline saved → %s", filepath)

    print(f"\n{SEPARATOR}")
    print("  💾  PIPELINE SAVED")
    print(SEPARATOR)
    print(f"\n   ✅  Pipeline persisted → {filepath}")
    print(f"       File size: {os.path.getsize(filepath):,} bytes")


# ===========================================================================
# STEP 8 – LOAD PIPELINE
# ===========================================================================


def load_pipeline(filepath: str) -> Pipeline:
    """
    Load a previously saved pipeline from disk using joblib.

    Parameters
    ----------
    filepath : str
        Path to the .pkl file.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded, fitted pipeline.

    Raises
    ------
    FileNotFoundError
        If the .pkl file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Pipeline file not found at '{filepath}'. "
            "Run the pipeline at least once to generate it."
        )

    pipeline = joblib.load(filepath)
    logger.info("Pipeline loaded from → %s", filepath)

    print(f"\n{SEPARATOR}")
    print("  📂  PIPELINE LOADED")
    print(SEPARATOR)
    print(f"\n   ✅  Pipeline loaded from → {filepath}")
    print(f"   Pipeline steps: {[step[0] for step in pipeline.steps]}")

    return pipeline
