"""
tests/test_features.py
----------------------
Tests for feature engineering in src/features.py.

Key leakage tests:
- customer_prior_rides = 0 for each customer's first booking.
- customer_prior_completion_rate = NaN for each customer's first booking.
- fare_per_km is null wherever Booking Value or Ride Distance is null.
- is_completed is True iff Booking Status == "Completed".
- cohort_month == min(booking_month) per customer.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data_loader import load_clean
from src.features import build_feature_store
from src.ml_pipeline import add_rolling_customer_features


@pytest.fixture(scope="module")
def feature_df():
    clean = load_clean()
    return build_feature_store(clean, save=False)


@pytest.fixture(scope="module")
def ml_df(feature_df):
    return add_rolling_customer_features(feature_df)


# ── fare_per_km ────────────────────────────────────────────────────────────────

def test_fare_per_km_null_when_fare_null(feature_df):
    mask = feature_df["Booking Value"].isna()
    assert feature_df.loc[mask, "fare_per_km"].isna().all(), (
        "fare_per_km is not null where Booking Value is null."
    )


def test_fare_per_km_null_when_distance_null(feature_df):
    mask = feature_df["Ride Distance"].isna()
    assert feature_df.loc[mask, "fare_per_km"].isna().all(), (
        "fare_per_km is not null where Ride Distance is null."
    )


# ── is_completed ───────────────────────────────────────────────────────────────

def test_is_completed_true_iff_completed_status(feature_df):
    expected = feature_df["Booking Status"].astype(str) == "Completed"
    pd.testing.assert_series_equal(
        feature_df["is_completed"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


# ── cohort_month ───────────────────────────────────────────────────────────────

def test_cohort_month_equals_min_booking_month(feature_df):
    """cohort_month must be the earliest booking_month per customer."""
    df = feature_df.copy()
    min_bm = df.groupby("Customer ID")["booking_month"].min()
    df["expected_cohort"] = df["Customer ID"].map(min_bm)
    mismatch = (df["cohort_month"] != df["expected_cohort"]).sum()
    assert mismatch == 0, f"{mismatch} rows have cohort_month != min(booking_month)."


# ── Rolling leakage tests ──────────────────────────────────────────────────────

def test_customer_prior_rides_zero_for_first_booking(ml_df):
    """Each customer's first booking must have customer_prior_rides == 0."""
    first_idx = ml_df.groupby("Customer ID")["datetime"].idxmin()
    first_bookings = ml_df.loc[first_idx]
    non_zero = (first_bookings["customer_prior_rides"] != 0).sum()
    assert non_zero == 0, (
        f"{non_zero} customers have customer_prior_rides != 0 on their first booking."
    )


def test_customer_prior_completion_rate_nan_for_first_booking(ml_df):
    """Each customer's first booking must have customer_prior_completion_rate == NaN."""
    # Use idxmin on datetime to get the true first booking per customer,
    # avoiding the pitfall of groupby().first() returning a non-first row when
    # the DataFrame index is not aligned with sort order.
    first_idx = ml_df.groupby("Customer ID")["datetime"].idxmin()
    first_bookings = ml_df.loc[first_idx]
    non_nan = first_bookings["customer_prior_completion_rate"].notna().sum()
    assert non_nan == 0, (
        f"{non_nan} customers have non-NaN customer_prior_completion_rate on their first booking."
    )


def test_prior_rides_never_uses_future_data(ml_df):
    """
    For any booking, customer_prior_rides must equal the number of
    prior bookings by that customer (strictly before this datetime).
    Spot-check on a sample of customers with 3+ bookings.
    """
    df = ml_df.sort_values("datetime").copy()
    repeat = df.groupby("Customer ID").filter(lambda g: len(g) >= 3)
    if len(repeat) == 0:
        pytest.skip("No customers with 3+ bookings in data.")

    sample_custs = repeat["Customer ID"].unique()[:50]
    for cid in sample_custs:
        cust = repeat[repeat["Customer ID"] == cid].sort_values("datetime").reset_index(drop=True)
        for i, row in cust.iterrows():
            expected = i  # 0-indexed: i prior bookings before the i-th one
            actual = int(row["customer_prior_rides"])
            assert actual == expected, (
                f"Customer {cid}: booking {i} has prior_rides={actual}, expected={expected}."
            )
