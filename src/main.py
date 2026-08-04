"""
main.py
=======
Orchestrates the complete Feature Engineering Pipeline for the
IBM HR Analytics Employee Attrition & Performance Dataset.

Run:
    python main.py

Workflow:
    Step 1  — Load & explore dataset
    Step 2  — Data cleaning
    Step 3  — Feature engineering (inside pipeline)
    Step 4  — Feature selection (SelectKBest)
    Step 5  — Build full pipeline
    Step 6  — Transform dataset & save
    Step 7  — Save pipeline
    Step 8  — Load pipeline & transform new data

Author  : MLOps Assignment
Dataset : IBM HR Analytics Employee Attrition & Performance
"""

import logging
import os
import sys

# ── Force UTF-8 output on Windows (handles emoji in console) ───────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python < 3.7 fallback (no reconfigure)

# ── Ensure src/ is on sys.path when run from project root ──────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd

# Project modules
import preprocessing as prep
import pipeline as pipe
import feature_selection as fs
import inference as infer
from pipeline import load_pipeline

# ---------------------------------------------------------------------------
# Logging (root logger configured here)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "employee_attrition.csv")
PIPELINE_PATH  = os.path.join(BASE_DIR, "models", "feature_pipeline.pkl")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
TRANSFORMED_CSV = os.path.join(OUTPUT_DIR, "transformed_data.csv")
HEATMAP_PATH   = os.path.join(OUTPUT_DIR, "missing_value_heatmap.png")
CORR_PATH      = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

# Number of top features to select
K_BEST = 20

# Columns to exclude from features (IDs, constants, or target)
DROP_COLS = [
    "Attrition",           # target
    "EmployeeNumber",      # ID
    "EmployeeCount",       # constant = 1
    "StandardHours",       # constant = 80
    "Over18",              # constant = 'Y'
]

SEPARATOR = "=" * 60


def screenshot_prompt(title: str, reason: str) -> None:
    """
    Print a clear, formatted screenshot reminder and pause execution
    until the user presses Enter.

    Parameters
    ----------
    title : str
        Screenshot title.
    reason : str
        Why this screenshot is important.
    """
    print(f"\n{'=' * 60}")
    print("  📸  TAKE SCREENSHOT NOW")
    print(f"{'=' * 60}")
    print(f"  Screenshot Title : {title}")
    print(f"  Reason           : {reason}")
    print(f"{'=' * 60}")
    input("\n  ▶  Press ENTER to continue … ")
    print()


# ===========================================================================
# STEP 1 — LOAD DATASET
# ===========================================================================

def main() -> None:
    """
    Main entry point — runs all 8 pipeline steps in sequence.
    """
    print(f"\n{'#' * 60}")
    print("  FEATURE ENGINEERING PIPELINE")
    print("  IBM HR Analytics — Employee Attrition & Performance")
    print(f"{'#' * 60}\n")
    logger.info("Starting Feature Engineering Pipeline …")

    # Ensure output / models directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — Load Dataset
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 1 — LOAD DATASET")
    print(SEPARATOR)

    df: pd.DataFrame = prep.load_data(DATA_PATH)
    prep.display_dataset_info(df)

    # Identify features (before cleaning so we know column types)
    target_col = "Attrition"
    numerical_features, categorical_features = prep.identify_features(
        df, drop_cols=DROP_COLS
    )

    screenshot_prompt(
        title="Dataset Loaded Successfully — Shape, First 5 Rows, Data Types",
        reason="Step 1 complete: Dataset loaded and explored.",
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — Data Cleaning
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 2 — DATA CLEANING")
    print(SEPARATOR)

    screenshot_prompt(
        title="Missing Value Analysis — Before Cleaning",
        reason="Capture missing value count and duplicate row count.",
    )

    df_clean: pd.DataFrame = prep.clean_data(
        df, numerical_features, categorical_features
    )

    screenshot_prompt(
        title="Data Cleaning Completed",
        reason=(
            "After cleaning: show missing values removed and "
            "duplicates removed."
        ),
    )

    # ── Visualisations ───────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  📊  GENERATING VISUALISATIONS")
    print(SEPARATOR)
    prep.plot_missing_value_heatmap(df_clean, HEATMAP_PATH)
    prep.plot_correlation_heatmap(
        df_clean, numerical_features, CORR_PATH
    )

    # ──────────────────────────────────────────────────────────────────────
    # Feature split — X and y
    # ──────────────────────────────────────────────────────────────────────
    X: pd.DataFrame = df_clean.drop(
        columns=[c for c in DROP_COLS if c in df_clean.columns]
    )
    y: pd.Series = (
        df_clean[target_col]
        .map({"Yes": 1, "No": 0})
        .fillna(0)
        .astype(int)
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — Feature Engineering preview
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 3 — FEATURE ENGINEERING")
    print(SEPARATOR)
    print("""
  Transformations to be applied inside the Pipeline:

  Numerical Features:
    • SimpleImputer  (strategy = 'mean')
    • StandardScaler (zero mean, unit variance)

  Categorical Features:
    • SimpleImputer  (strategy = 'most_frequent')
    • OneHotEncoder  (handle_unknown = 'ignore')
""")

    screenshot_prompt(
        title="Numerical & Categorical Features Identified",
        reason="Step 3 preparation: feature lists printed.",
    )

    screenshot_prompt(
        title="Feature Engineering — Scaling & Encoding Configuration",
        reason=(
            "Show configured transformers before pipeline is built."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — Build Pipeline
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 5 — BUILD PIPELINE")
    print(SEPARATOR)

    full_pipeline = pipe.create_preprocessing_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        k_best=K_BEST,
    )

    print(f"\n   Pipeline Steps:")
    for step_name, step_obj in full_pipeline.steps:
        print(f"   • {step_name} → {type(step_obj).__name__}")

    screenshot_prompt(
        title="Pipeline Created Successfully",
        reason="Step 5: Full preprocessing pipeline architecture displayed.",
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6 — Transform Dataset
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 6 — TRANSFORM DATASET")
    print(SEPARATOR)

    X_transformed, selected_names = pipe.transform_dataset(
        pipeline=full_pipeline,
        X=X,
        y=y,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        output_path=TRANSFORMED_CSV,
    )

    screenshot_prompt(
        title="Original Shape vs Transformed Shape",
        reason="Step 6: Show how the dataset dimensions changed.",
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — Feature Selection (standalone display + chart)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 4 — FEATURE SELECTION ANALYSIS")
    print(SEPARATOR)

    # Get names BEFORE SelectKBest from the fitted pipeline
    all_names_before_sel = pipe.get_feature_names_after_preprocessing(
        full_pipeline, numerical_features, categorical_features
    )

    # Get the already-transformed array (before SelectKBest)
    preprocessor_step = full_pipeline.named_steps["preprocessor"]
    X_preprocessed = preprocessor_step.transform(X)

    _, _, _ = fs.perform_feature_selection(
        X_transformed=X_preprocessed,
        y=y,
        feature_names_in=all_names_before_sel,
        k=K_BEST,
        save_path=IMPORTANCE_PATH,
    )

    screenshot_prompt(
        title="Feature Selection — Selected Features & Scores",
        reason=(
            "Step 4: SelectKBest results — feature names and F-scores."
        ),
    )

    screenshot_prompt(
        title="Transformed Dataset Saved",
        reason=(
            "Step 6: Confirm CSV saved to output/transformed_data.csv."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7 — Save Pipeline
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 7 — SAVE PIPELINE")
    print(SEPARATOR)

    pipe.save_pipeline(full_pipeline, PIPELINE_PATH)

    screenshot_prompt(
        title="Pipeline Saved Successfully",
        reason=(
            "Step 7: feature_pipeline.pkl saved to models/ folder."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8 — Reuse Pipeline
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  STEP 8 — REUSE PIPELINE (INFERENCE)")
    print(SEPARATOR)

    loaded_pipeline = load_pipeline(PIPELINE_PATH)

    screenshot_prompt(
        title="Pipeline Loaded Successfully",
        reason="Step 8: Pipeline loaded from disk — ready for inference.",
    )

    new_employees_df = infer.create_sample_employees()
    _ = infer.transform_new_data(loaded_pipeline, new_employees_df)

    screenshot_prompt(
        title="New Employee Data Transformed Successfully",
        reason=(
            "Step 8: New employee records transformed using "
            "loaded pipeline."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  📁  OUTPUT FOLDER CONTENTS")
    print(SEPARATOR)
    print(f"\n   Directory: {OUTPUT_DIR}")
    print("-" * 40)
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, filename)
        size = os.path.getsize(filepath)
        print(f"   📄 {filename:<35}  {size:>10,} bytes")

    screenshot_prompt(
        title="Output Folder Contents",
        reason="Show all generated files in the output/ directory.",
    )

    print(f"\n{SEPARATOR}")
    print("  🌳  FINAL PROJECT STRUCTURE")
    print(SEPARATOR)
    _print_project_tree(BASE_DIR)

    screenshot_prompt(
        title="Final Project Structure",
        reason=(
            "Show complete project folder tree with all generated files."
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'#' * 60}")
    print("  ✅  PIPELINE EXECUTION COMPLETE")
    print(f"{'#' * 60}\n")
    logger.info("Feature Engineering Pipeline completed successfully.")


def _print_project_tree(base_dir: str, indent: str = "") -> None:
    """
    Recursively print a tree-style directory listing.

    Parameters
    ----------
    base_dir : str
        Root directory to display.
    indent : str
        Current indentation string (used in recursion).
    """
    skip = {".venv", "__pycache__", ".git", ".idea"}

    entries = sorted(os.listdir(base_dir))
    for i, entry in enumerate(entries):
        full_path = os.path.join(base_dir, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(f"   {indent}{connector}{entry}")

        if os.path.isdir(full_path) and entry not in skip:
            ext = "    " if i == len(entries) - 1 else "│   "
            _print_project_tree(full_path, indent + ext)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
