"""
preprocessing.py
================
Handles all data loading, exploration, and cleaning operations
for the Feature Engineering Pipeline project.

Author  : MLOps Assignment
Dataset : IBM HR Analytics Employee Attrition & Performance
"""

import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEPARATOR = "=" * 60
SUBSEP = "-" * 60


# ===========================================================================
# STEP 1 – DATA LOADING & EXPLORATION
# ===========================================================================


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given file path.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path.
    ValueError
        If the file cannot be parsed as a CSV.
    """
    logger.info("Loading dataset from: %s", filepath)
    try:
        df = pd.read_csv(filepath)
        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df
    except FileNotFoundError as exc:
        logger.error("File not found: %s", filepath)
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place the CSV in the data/ folder."
        ) from exc
    except Exception as exc:
        logger.error("Failed to parse CSV: %s", exc)
        raise ValueError(f"Could not read CSV file: {exc}") from exc


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Print comprehensive information about the DataFrame including
    shape, first 5 rows, data types, and basic statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.
    """
    print(f"\n{SEPARATOR}")
    print("  📊  DATASET OVERVIEW")
    print(SEPARATOR)

    # Shape
    print(f"\n🔷 Dataset Shape : {df.shape[0]} rows × {df.shape[1]} columns")

    # First 5 rows
    print(f"\n🔷 First 5 Rows:")
    print(SUBSEP)
    print(df.head().to_string())

    # Data types
    print(f"\n🔷 Data Types:")
    print(SUBSEP)
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null Count": df.count().values,
    })
    print(dtype_df.to_string(index=False))

    # Basic statistics (numerical only)
    print(f"\n🔷 Descriptive Statistics (Numerical):")
    print(SUBSEP)
    print(df.describe().round(2).to_string())

    logger.info("Dataset info displayed.")


def identify_features(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
) -> Tuple[list[str], list[str]]:
    """
    Identify numerical and categorical feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    drop_cols : list[str], optional
        Columns to exclude (e.g., ID cols or the target).

    Returns
    -------
    Tuple[list[str], list[str]]
        (numerical_features, categorical_features)
    """
    if drop_cols is None:
        drop_cols = []

    working_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numerical_features: list[str] = working_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features: list[str] = working_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"\n{SEPARATOR}")
    print("  🔍  FEATURE IDENTIFICATION")
    print(SEPARATOR)
    print(f"\n🔢 Numerical Features ({len(numerical_features)}):")
    for feat in numerical_features:
        print(f"   • {feat}")

    print(f"\n🔤 Categorical Features ({len(categorical_features)}):")
    for feat in categorical_features:
        print(f"   • {feat}")

    logger.info(
        "Features identified — Numerical: %d, Categorical: %d",
        len(numerical_features),
        len(categorical_features),
    )
    return numerical_features, categorical_features


# ===========================================================================
# STEP 2 – DATA CLEANING
# ===========================================================================


def inject_missing_values(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
    num_missing: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Programmatically inject a small number of NaN values into the
    dataset for demonstration purposes.

    This is required when the original dataset (IBM HR) has no missing
    values, so that the imputation steps can still be showcased.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    numerical_cols : list[str]
        Numerical column names eligible for injection.
    categorical_cols : list[str]
        Categorical column names eligible for injection.
    num_missing : int
        Total number of missing cells to inject per type.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with artificially injected NaN values.
    """
    random.seed(seed)
    np.random.seed(seed)
    df_dirty = df.copy()

    # Inject into numerical columns
    for _ in range(num_missing):
        row = random.randint(0, len(df_dirty) - 1)
        col = random.choice(numerical_cols[:10])   # limit to first 10 cols
        df_dirty.at[row, col] = np.nan

    # Inject into categorical columns
    for _ in range(num_missing):
        row = random.randint(0, len(df_dirty) - 1)
        col = random.choice(categorical_cols[:5])
        df_dirty.at[row, col] = np.nan

    logger.info(
        "Injected %d numerical + %d categorical missing values for demo.",
        num_missing,
        num_missing,
    )
    return df_dirty


def display_missing_values(df: pd.DataFrame, label: str = "") -> None:
    """
    Print a summary of missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    label : str
        Optional label printed above the table (e.g., 'BEFORE' / 'AFTER').
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct,
    })
    summary = summary[summary["Missing Count"] > 0]

    if label:
        print(f"\n📋 Missing Values — {label}")
        print(SUBSEP)

    if summary.empty:
        print("  ✅  No missing values found.")
    else:
        print(summary.to_string())


def clean_data(
    df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """
    Perform data cleaning:
      1. Display missing values BEFORE cleaning.
      2. Inject artificial missing values if none exist (demo requirement).
      3. Remove duplicate rows.
      4. Display missing values AFTER cleaning (imputation done in pipeline).

    Note: Actual imputation (mean/mode) is performed inside the
          Scikit-Learn pipeline (Step 3 / 5) to avoid data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame.
    numerical_cols : list[str]
        Numerical column names.
    categorical_cols : list[str]
        Categorical column names.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (duplicates removed, missing values present for
        the pipeline to handle).
    """
    print(f"\n{SEPARATOR}")
    print("  🧹  DATA CLEANING")
    print(SEPARATOR)

    # ── 2a. Show BEFORE state ────────────────────────────────────────────────
    total_missing_before = df.isnull().sum().sum()
    print(f"\n🔷 Missing Value Analysis — BEFORE Cleaning:")
    display_missing_values(df, label="BEFORE")

    # ── 2b. Inject artificial NaNs if dataset is clean ──────────────────────
    if total_missing_before == 0:
        print(
            "\n⚠️  No missing values found in the original dataset.\n"
            "   Injecting artificial NaN values for demonstration …"
        )
        df = inject_missing_values(df, numerical_cols, categorical_cols)
        print("\n📋 Missing Values — AFTER Injection (for demo):")
        display_missing_values(df, label="AFTER INJECTION")

    # ── 2c. Remove duplicate rows ────────────────────────────────────────────
    dup_count = df.duplicated().sum()
    print(f"\n🔷 Duplicate Rows Found : {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows.", dup_count)
        print(f"   ✅  {dup_count} duplicate(s) removed.")
    else:
        print("   ✅  No duplicates found.")

    # ── 2d. Show AFTER state ─────────────────────────────────────────────────
    print(f"\n🔷 Missing Value Analysis — AFTER Cleaning (pre-imputation):")
    display_missing_values(df, label="AFTER DEDUP (imputation in pipeline)")
    total_missing_after = df.isnull().sum().sum()
    print(f"\n   Total missing cells remaining : {total_missing_after}")
    print(
        "   (These will be handled by SimpleImputer inside the pipeline.)"
    )

    logger.info(
        "Data cleaning complete. Shape after cleaning: %s", df.shape
    )
    return df


# ===========================================================================
# VISUALIZATIONS
# ===========================================================================


def plot_missing_value_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """
    Generate and save a heatmap showing missing value locations
    across the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (with NaN values already injected).
    save_path : str
        File path to save the heatmap image (PNG).
    """
    plt.figure(figsize=(20, 6))
    sns.heatmap(
        df.isnull(),
        yticklabels=False,
        cbar=True,
        cmap="YlOrRd",
        linewidths=0.3,
    )
    plt.title(
        "Missing Value Heatmap — IBM HR Attrition Dataset",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Features", fontsize=11)
    plt.ylabel("Rows", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Missing value heatmap saved → %s", save_path)
    print(f"   📈 Missing value heatmap saved → {save_path}")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numerical_cols: list[str],
    save_path: str,
) -> None:
    """
    Generate and save a correlation heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    numerical_cols : list[str]
        Numerical columns to include in the correlation matrix.
    save_path : str
        File path to save the heatmap image (PNG).
    """
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    plt.title(
        "Feature Correlation Heatmap — Numerical Features",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Correlation heatmap saved → %s", save_path)
    print(f"   📈 Correlation heatmap saved → {save_path}")
