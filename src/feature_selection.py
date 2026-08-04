"""
feature_selection.py
====================
Implements feature selection using Scikit-Learn's SelectKBest
with the chi2 / f_classif scoring functions.

Author  : MLOps Assignment
Dataset : IBM HR Analytics Employee Attrition & Performance
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60
SUBSEP = "-" * 60


# ===========================================================================
# STEP 4 – FEATURE SELECTION
# ===========================================================================


def perform_feature_selection(
    X_transformed: np.ndarray,
    y: pd.Series,
    feature_names_in: list[str],
    k: int = 20,
    save_path: str = "output/feature_importance.png",
) -> Tuple[np.ndarray, list[str], np.ndarray]:
    """
    Apply SelectKBest (f_classif) to an already-transformed feature matrix
    and return the top-k features, their names, and their scores.

    This function is used STANDALONE (outside the pipeline) for
    analysis and visualisation purposes.

    Parameters
    ----------
    X_transformed : np.ndarray
        The transformed feature matrix (output of ColumnTransformer).
    y : pd.Series
        Target variable (Attrition encoded as 0/1).
    feature_names_in : list[str]
        Column names corresponding to X_transformed.
    k : int
        Number of top features to select.
    save_path : str
        Where to save the feature importance bar chart.

    Returns
    -------
    Tuple[np.ndarray, list[str], np.ndarray]
        (X_selected, selected_feature_names, scores_array)
    """
    logger.info(
        "Running SelectKBest (f_classif) with k=%d on %d features …",
        k,
        X_transformed.shape[1],
    )

    # Guard: k must not exceed the number of features
    k_actual = min(k, X_transformed.shape[1])
    if k_actual < k:
        logger.warning(
            "Requested k=%d > available features (%d). Using k=%d.",
            k,
            X_transformed.shape[1],
            k_actual,
        )

    selector = SelectKBest(score_func=f_classif, k=k_actual)
    X_selected = selector.fit_transform(X_transformed, y)

    # Retrieve scores and selected names
    scores: np.ndarray = selector.scores_
    support_mask: np.ndarray = selector.get_support()

    selected_feature_names: list[str] = [
        feature_names_in[i]
        for i, selected in enumerate(support_mask)
        if selected
    ]

    # ── Display ──────────────────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  🎯  FEATURE SELECTION — SelectKBest (f_classif)")
    print(SEPARATOR)
    print(f"\n   Total features before selection : {X_transformed.shape[1]}")
    print(f"   Features selected (k)           : {k_actual}")

    # Score table for all features
    score_df = pd.DataFrame({
        "Feature": feature_names_in,
        "F-Score": scores,
        "Selected": support_mask,
    }).sort_values("F-Score", ascending=False)

    print(f"\n🔷 All Feature Scores (sorted by F-Score):")
    print(SUBSEP)
    print(score_df.to_string(index=False))

    # Selected features
    selected_score_df = score_df[score_df["Selected"]].reset_index(drop=True)
    print(f"\n🔷 Selected Features ({k_actual}):")
    print(SUBSEP)
    print(selected_score_df[["Feature", "F-Score"]].to_string(index=False))

    logger.info(
        "Feature selection complete. Selected %d features.", k_actual
    )

    # ── Visualisation ─────────────────────────────────────────────────────────
    _plot_feature_importance(
        feature_names=feature_names_in,
        scores=scores,
        selected_mask=support_mask,
        k=k_actual,
        save_path=save_path,
    )

    return X_selected, selected_feature_names, scores


def _plot_feature_importance(
    feature_names: list[str],
    scores: np.ndarray,
    selected_mask: np.ndarray,
    k: int,
    save_path: str,
) -> None:
    """
    Generate and save a horizontal bar chart of feature importance scores.

    Parameters
    ----------
    feature_names : list[str]
        All feature names.
    scores : np.ndarray
        F-scores for each feature.
    selected_mask : np.ndarray
        Boolean mask marking selected features.
    k : int
        Number of top-k features selected.
    save_path : str
        File path to save the chart.
    """
    # Sort by score descending
    order = np.argsort(scores)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_scores = scores[order]
    sorted_mask = selected_mask[order]

    # Color: selected = steelblue, others = lightgrey
    colors = ["steelblue" if m else "lightgrey" for m in sorted_mask]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_names) * 0.35)))
    bars = ax.barh(
        sorted_names[::-1],   # reverse so highest score is at top
        sorted_scores[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("F-Score (f_classif)", fontsize=11)
    ax.set_title(
        f"SelectKBest Feature Importance — Top {k} Selected Features",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # Annotate bars with score values
    for bar, score in zip(bars, sorted_scores[::-1]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            ha="left",
            fontsize=7,
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", label="Selected"),
        Patch(facecolor="lightgrey", label="Not Selected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance chart saved → %s", save_path)
    print(f"   📈 Feature importance chart saved → {save_path}")


def get_selector_for_pipeline(k: int = 20) -> SelectKBest:
    """
    Return a fresh SelectKBest instance ready to be embedded in a
    Scikit-Learn Pipeline.

    Parameters
    ----------
    k : int
        Number of top features to retain.

    Returns
    -------
    SelectKBest
        Configured SelectKBest transformer.
    """
    return SelectKBest(score_func=f_classif, k=k)
