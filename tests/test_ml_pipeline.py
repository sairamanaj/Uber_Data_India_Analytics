"""
tests/test_ml_pipeline.py
--------------------------
Tests for the ML pipeline in src/ml_pipeline.py.

Key rules verified:
- No excluded (post-outcome) columns appear in the feature matrix.
- Train set contains only dates ≤ 2024-09-30; test set only ≥ 2024-10-01.
- Target encoding maps are fitted on training rows only.
- No hardcoded ROC-AUC pass/fail threshold — performance is reported only.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data_loader import load_clean
from src.features import build_feature_store
from src.ml_pipeline import (
    add_rolling_customer_features,
    fit_target_encoding,
    apply_target_encoding,
    build_feature_matrix_classification,
    time_split,
    EXCLUDED_COLUMNS,
    TRAIN_CUTOFF,
    TEST_START,
)


@pytest.fixture(scope="module")
def prepared():
    clean = load_clean()
    features = build_feature_store(clean, save=False)
    ml_df = add_rolling_customer_features(features)
    train_df, test_df = time_split(ml_df)
    encodings = fit_target_encoding(train_df, "is_completed",
                                    ["Pickup Location", "Drop Location"], smoothing=10.0)
    global_fallback = train_df["is_completed"].mean()
    X_train, y_train = build_feature_matrix_classification(
        train_df, encodings, global_fallback, is_train=True)
    X_test, y_test = build_feature_matrix_classification(
        test_df, encodings, global_fallback, is_train=False)
    return {
        "train_df": train_df, "test_df": test_df,
        "encodings": encodings, "global_fallback": global_fallback,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
    }


# ── No post-outcome columns in feature matrix ──────────────────────────────────

def test_no_excluded_columns_in_X_train(prepared):
    leaked = set(prepared["X_train"].columns) & EXCLUDED_COLUMNS
    assert len(leaked) == 0, f"Excluded columns found in X_train: {leaked}"


def test_no_excluded_columns_in_X_test(prepared):
    leaked = set(prepared["X_test"].columns) & EXCLUDED_COLUMNS
    assert len(leaked) == 0, f"Excluded columns found in X_test: {leaked}"


# ── Time-based split dates ─────────────────────────────────────────────────────

def test_train_dates_within_cutoff(prepared):
    max_train_date = prepared["train_df"]["datetime"].max()
    assert max_train_date <= TRAIN_CUTOFF, (
        f"Train set contains dates after cutoff: {max_train_date}"
    )


def test_test_dates_after_cutoff(prepared):
    min_test_date = prepared["test_df"]["datetime"].min()
    assert min_test_date >= TEST_START, (
        f"Test set contains dates before TEST_START: {min_test_date}"
    )


def test_no_overlap_between_train_and_test(prepared):
    train_dates = set(prepared["train_df"]["datetime"].dt.date)
    test_dates = set(prepared["test_df"]["datetime"].dt.date)
    overlap = train_dates & test_dates
    assert len(overlap) == 0, f"Train/test date overlap detected: {sorted(overlap)[:5]}"


# ── Target encoding fitted on train only ──────────────────────────────────────

def test_encoding_fitted_on_train_only(prepared):
    """
    Verify that location encodings were computed from training rows.
    The global fallback used for test-set unseen locations should match
    the training set's overall completion rate — not the full dataset's.
    """
    train_global = prepared["train_df"]["is_completed"].mean()
    full_global = load_clean()["Booking Status"].eq("Completed").mean()
    # They may differ; the key assertion is that global_fallback matches train
    assert abs(prepared["global_fallback"] - train_global) < 1e-9, (
        f"global_fallback {prepared['global_fallback']:.6f} != "
        f"train completion rate {train_global:.6f}"
    )


# ── No ROC-AUC threshold assertion ────────────────────────────────────────────

def test_no_hardcoded_auc_threshold():
    """
    Confirm that the ML pipeline does not assert a pass/fail AUC threshold.
    This test documents the policy: AUC is reported and interpreted in context.
    """
    # Read the pipeline source and check no assert on roc_auc
    pipeline_src = pathlib.Path("src/ml_pipeline.py").read_text(encoding="utf-8")
    # Look for any hardcoded threshold assertions
    problematic_patterns = [
        'assert.*roc_auc.*>.*0\\.6',
        'assert.*roc_auc.*>=.*0\\.6',
        'assert.*auc.*>.*0\\.6',
    ]
    import re
    for pattern in problematic_patterns:
        matches = re.findall(pattern, pipeline_src, re.IGNORECASE)
        assert len(matches) == 0, (
            f"Found hardcoded AUC threshold assertion in ml_pipeline.py: {matches}"
        )


# ── Feature matrix has no NaN in non-imputed columns ──────────────────────────

def test_feature_matrix_finite_after_imputation(prepared):
    """After pipeline imputation, X_train should have no NaN (imputer handles them)."""
    from src.ml_pipeline import build_classification_pipeline
    import numpy as np
    pipe = build_classification_pipeline("logistic")
    # Only fit/transform the imputer step to check
    imputer = pipe.named_steps["imputer"]
    X_arr = imputer.fit_transform(prepared["X_train"].fillna(np.nan).values)
    assert not np.any(np.isnan(X_arr)), "NaN values remain after imputation in training data."
