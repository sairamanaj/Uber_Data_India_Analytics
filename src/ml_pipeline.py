"""
src/ml_pipeline.py
------------------
ML training and evaluation pipeline for:
  Task A: Completion prediction (binary classification)
  Task B: Fare estimation (regression, completed rides only)

Key constraints:
- Time-based train/test split ONLY (train ≤ Sep 2024, test ≥ Oct 2024).
- All post-outcome columns strictly excluded (see EXCLUDED_COLUMNS).
- Customer history features computed via time-sorted rolling to prevent leakage.
- Target encoding for locations fitted on training set only.
- Optional gradient boosting (Stage 2) only enabled if explicitly requested.
- No hardcoded ROC-AUC pass/fail threshold; performance is reported and interpreted.
"""

from __future__ import annotations

import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = pathlib.Path("outputs/models")

TRAIN_CUTOFF = pd.Timestamp("2024-09-30 23:59:59")
TEST_START = pd.Timestamp("2024-10-01 00:00:00")

# ── Columns that are NEVER allowed as features ─────────────────────────────────
EXCLUDED_COLUMNS = {
    "Booking Value",          # post-outcome
    "Ride Distance",          # post-outcome (for Task A; explicitly permitted for Task B)
    "Driver Ratings",         # post-outcome
    "Customer Rating",        # post-outcome
    "Payment Method",         # post-outcome
    "Cancelled Rides by Customer",
    "Reason for cancelling by Customer",
    "Cancelled Rides by Driver",
    "Driver Cancellation Reason",
    "Incomplete Rides",
    "Incomplete Rides Reason",
    "Booking Status",         # the target
    "Booking ID",             # identifier
    "Customer ID",            # identifier
    "Date",
    "Time",
    "datetime",
    "first_booking_date",
    "date_only",
    "cohort_month",
    "booking_month",
    # EDA aggregates (computed on full dataset — leaks future)
    "eda_customer_total_rides",
    "eda_customer_completed_rides",
    "eda_customer_ocv",
    "eda_customer_avg_fare",
    "fare_per_km",            # derived from Booking Value
    "is_cancelled",           # derived from Booking Status
    "cancellation_party",     # derived from Booking Status
}


# ── Rolling customer history (time-safe) ──────────────────────────────────────

def add_rolling_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-safe per-customer rolling features using only strictly prior bookings.
    Sorted by datetime; customer's first booking gets 0 / NaN.

    Columns added:
        customer_prior_rides            (int)
        customer_prior_completion_rate  (float, NaN for first booking)
        customer_prior_cancellation_rate(float, NaN for first booking)

    Implementation note: results are assigned back via index (not positional list
    extension) to handle the case where a customer's bookings are not contiguous
    in the datetime-sorted dataframe.
    """
    df = df.sort_values("datetime").copy()
    df["_is_completed_int"] = df["is_completed"].astype(int)
    df["_is_cancelled_int"] = df["is_cancelled"].astype(int)

    # Pre-allocate output columns with correct dtypes
    df["customer_prior_rides"] = 0
    df["customer_prior_completion_rate"] = np.nan
    df["customer_prior_cancellation_rate"] = np.nan

    # groupby with sort=True so groups are in alphabetical order — does not matter
    # because we assign back by index. Groups are iterated in sorted-datetime order
    # within each group because df is already sorted by datetime.
    for _cid, grp in df.groupby("Customer ID", sort=False):
        n = len(grp)
        cumrides = np.arange(n, dtype=float)              # 0, 1, 2, … (prior count)
        cumcomp = grp["_is_completed_int"].cumsum().shift(1, fill_value=0).values
        cumcanc = grp["_is_cancelled_int"].cumsum().shift(1, fill_value=0).values

        with np.errstate(invalid="ignore", divide="ignore"):
            comp_rate = np.where(cumrides > 0, cumcomp / cumrides, np.nan)
            canc_rate = np.where(cumrides > 0, cumcanc / cumrides, np.nan)

        # Assign by index to avoid positional misalignment
        df.loc[grp.index, "customer_prior_rides"] = cumrides.astype(int)
        df.loc[grp.index, "customer_prior_completion_rate"] = comp_rate
        df.loc[grp.index, "customer_prior_cancellation_rate"] = canc_rate

    df = df.drop(columns=["_is_completed_int", "_is_cancelled_int"])
    return df


# ── Target encoding (location) ────────────────────────────────────────────────

def fit_target_encoding(
    train_df: pd.DataFrame,
    target_col: str,
    loc_cols: list[str],
    smoothing: float = 10.0,
) -> dict[str, dict[str, float]]:
    """
    Fit smoothed target encoding for location columns on training data.
    global_mean * smoothing / (count + smoothing) + group_mean * count / (count + smoothing)

    Returns a dict {col_name: {location: encoded_value}}.
    """
    global_mean = train_df[target_col].mean()
    encodings: dict[str, dict[str, float]] = {}
    for col in loc_cols:
        stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (
            stats["mean"] * stats["count"] + global_mean * smoothing
        ) / (stats["count"] + smoothing)
        encodings[col] = smoothed.to_dict()
    return encodings


def apply_target_encoding(
    df: pd.DataFrame,
    encodings: dict[str, dict[str, float]],
    global_fallback: float,
) -> pd.DataFrame:
    """Apply pre-fitted target encoding; unseen locations get global_fallback."""
    df = df.copy()
    for col, mapping in encodings.items():
        new_col = col.lower().replace(" ", "_") + "_enc"
        df[new_col] = df[col].map(mapping).fillna(global_fallback)
    return df


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_feature_matrix_classification(
    df: pd.DataFrame,
    loc_encodings: dict[str, dict[str, float]] | None,
    global_fallback: float,
    is_train: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y for Task A (completion prediction).
    loc_encodings must be fitted on train only.
    """
    df = df.copy()

    # Apply target encoding
    if loc_encodings is not None:
        df = apply_target_encoding(df, loc_encodings, global_fallback)

    # One-hot encode Vehicle Type
    vehicle_dummies = pd.get_dummies(df["Vehicle Type"], prefix="vehicle", drop_first=False)

    # Time slot ordinal encoding
    slot_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"]
    df["time_slot_enc"] = df["time_slot"].astype(str).map(
        {s: i for i, s in enumerate(slot_order)}
    ).fillna(-1).astype(int)

    numeric_features = [
        "hour", "day_of_week", "month", "is_weekend",
        "Avg VTAT", "Avg CTAT",
        "customer_prior_rides",
        "customer_prior_completion_rate",
        "customer_prior_cancellation_rate",
        "pickup_location_enc", "drop_location_enc",
        "time_slot_enc",
        "months_since_first",
    ]

    X = pd.concat(
        [df[[c for c in numeric_features if c in df.columns]], vehicle_dummies],
        axis=1,
    )
    X["is_weekend"] = X["is_weekend"].astype(int)

    y = df["is_completed"].astype(int)

    # Sanity check: none of the excluded columns should be in X
    leaked = set(X.columns) & EXCLUDED_COLUMNS
    if leaked:
        raise ValueError(f"[LEAKAGE] Excluded columns found in feature matrix: {leaked}")

    return X, y


# ── Train / test split ────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (≤ Sep 2024) and test (≥ Oct 2024)."""
    train = df[df["datetime"] <= TRAIN_CUTOFF].copy()
    test = df[df["datetime"] >= TEST_START].copy()
    assert len(train) + len(test) <= len(df), "Overlapping split detected."
    print(f"[SPLIT] Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


# ── Model training ────────────────────────────────────────────────────────────

def build_classification_pipeline(model_name: str = "logistic") -> Pipeline:
    """
    Build a sklearn Pipeline for classification.
    model_name: 'logistic' or 'random_forest'
    """
    if model_name == "logistic":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose 'logistic' or 'random_forest'.")
    return Pipeline(steps)


def train_classify(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "logistic",
) -> Pipeline:
    pipe = build_classification_pipeline(model_name)
    pipe.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"clf_{model_name}.joblib")
    print(f"[OK] Trained {model_name} classifier → saved.")
    return pipe


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_classifier(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """
    Evaluate a trained classifier on the test set.
    Returns a dict of metrics. Does NOT assert any threshold.
    """
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "model": model_name,
        "roc_auc": round(auc, 4),
        "avg_precision": round(avg_precision, 4),
        "precision_1": round(report["1"]["precision"], 4),
        "recall_1": round(report["1"]["recall"], 4),
        "f1_1": round(report["1"]["f1-score"], 4),
        "precision_0": round(report["0"]["precision"], 4),
        "recall_0": round(report["0"]["recall"], 4),
        "f1_0": round(report["0"]["f1-score"], 4),
        "confusion_matrix": cm,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "pr_precision": prec_curve,
        "pr_recall": rec_curve,
    }
    print(f"\n[EVAL] {model_name}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Avg Prec: {avg_precision:.4f}")
    print(classification_report(y_test, y_pred))
    return metrics


# ── Fare estimation (Task B) ───────────────────────────────────────────────────

def build_fare_features(
    df: pd.DataFrame,
    loc_encodings: dict[str, dict[str, float]],
    global_fallback: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build X, y for Task B (fare regression on completed rides)."""
    df = df[df["is_completed"]].copy()
    df = apply_target_encoding(df, loc_encodings, global_fallback)

    vehicle_dummies = pd.get_dummies(df["Vehicle Type"], prefix="vehicle", drop_first=False)
    numeric = ["Ride Distance", "hour", "day_of_week", "month",
               "pickup_location_enc", "drop_location_enc"]

    X = pd.concat([df[[c for c in numeric if c in df.columns]], vehicle_dummies], axis=1)
    y = df["Booking Value"]
    return X, y


def train_fare_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "linear",
) -> Pipeline:
    if model_name == "linear":
        reg = LinearRegression()
        steps = [("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler()), ("reg", reg)]
    elif model_name == "random_forest":
        reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        steps = [("imputer", SimpleImputer(strategy="median")), ("reg", reg)]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"reg_{model_name}.joblib")
    print(f"[OK] Trained {model_name} fare regressor → saved.")
    return pipe


def evaluate_regressor(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "y_pred": y_pred,
    }
    print(f"\n[EVAL] {model_name} (fare regression)")
    print(f"  MAE : {mae:.2f} INR")
    print(f"  RMSE: {rmse:.2f} INR")
    print(f"  R²  : {r2:.4f}")
    return metrics
