"""
tests/test_data_loader.py
-------------------------
Tests for the raw → cleaned pipeline.

Key rules verified:
- Raw CSV is never modified (SHA-256 hash checked before/after).
- "null" strings are fully replaced.
- Triple-quotes stripped from IDs.
- datetime column is parsed.
- Null alignment matches expected structural patterns (warns, does not fail).
- Row count deviation is reported but not an assertion failure.
"""

import hashlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data_loader import load_raw, clean, RAW_PATH, EXPECTED_ROW_COUNT


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def raw_df():
    return load_raw()


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    return clean(raw_df, verbose=False)


# ── Raw CSV integrity ──────────────────────────────────────────────────────────

def test_raw_csv_exists():
    assert RAW_PATH.exists(), f"Raw CSV not found at {RAW_PATH}"


def test_raw_csv_not_modified_by_clean(raw_df):
    """The clean() function must never write to the source file."""
    hash_before = sha256(RAW_PATH)
    clean(raw_df.copy(), verbose=False)
    hash_after = sha256(RAW_PATH)
    assert hash_before == hash_after, "Raw CSV was modified during cleaning!"


# ── Row count ─────────────────────────────────────────────────────────────────

def test_row_count_reported_not_asserted(raw_df):
    """Row count deviation should be reportable — we check it is non-zero and print it."""
    actual = len(raw_df)
    assert actual > 0, "Raw dataframe has no rows."
    if actual != EXPECTED_ROW_COUNT:
        warnings.warn(f"Row count is {actual:,}, expected {EXPECTED_ROW_COUNT:,}.")


# ── Triple-quote stripping ─────────────────────────────────────────────────────

def test_no_triple_quotes_in_booking_id(cleaned_df):
    has_quotes = cleaned_df["Booking ID"].str.contains('"', na=False).any()
    assert not has_quotes, "Triple-quotes remain in Booking ID after cleaning."


def test_no_triple_quotes_in_customer_id(cleaned_df):
    has_quotes = cleaned_df["Customer ID"].str.contains('"', na=False).any()
    assert not has_quotes, "Triple-quotes remain in Customer ID after cleaning."


# ── "null" string replacement ──────────────────────────────────────────────────

def test_no_null_strings_remain(cleaned_df):
    str_cols = cleaned_df.select_dtypes(include="object").columns
    for col in str_cols:
        count = (cleaned_df[col] == "null").sum()
        assert count == 0, f'Column "{col}" still has {count} "null" strings after cleaning.'


# ── datetime parsing ───────────────────────────────────────────────────────────

def test_datetime_column_exists_and_parsed(cleaned_df):
    assert "datetime" in cleaned_df.columns, "datetime column missing after cleaning."
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["datetime"]), (
        "datetime column is not datetime64 dtype."
    )


def test_no_nat_datetimes(cleaned_df):
    nat_count = cleaned_df["datetime"].isna().sum()
    assert nat_count == 0, f"{nat_count} NaT values found in datetime column."


# ── Structural null alignment ──────────────────────────────────────────────────

def test_booking_value_null_for_noncompleted(cleaned_df):
    """
    Booking Value is non-null for Completed AND Incomplete rides.
    (Incomplete rides started and incurred a fare before ending abnormally.)
    It must be null for Cancelled by Customer, Cancelled by Driver, and No Driver Found.
    This was verified against the actual dataset: 9,000 Incomplete rows carry a Booking Value.
    """
    non_revenue_statuses = {"Cancelled by Customer", "Cancelled by Driver", "No Driver Found"}
    violations = cleaned_df[
        cleaned_df["Booking Status"].isin(non_revenue_statuses) &
        cleaned_df["Booking Value"].notna()
    ]
    assert len(violations) == 0, (
        f"{len(violations)} cancelled/no-driver rows have a non-null Booking Value."
    )


def test_vtat_null_only_for_no_driver_found(cleaned_df):
    """Avg VTAT should only be null for 'No Driver Found' rows."""
    unexpected = cleaned_df[
        cleaned_df["Avg VTAT"].isna() &
        (cleaned_df["Booking Status"] != "No Driver Found")
    ]
    assert len(unexpected) == 0, (
        f"{len(unexpected)} rows have null Avg VTAT for status other than 'No Driver Found'."
    )


# ── Dtype checks ───────────────────────────────────────────────────────────────

def test_numeric_dtypes(cleaned_df):
    for col in ["Booking Value", "Ride Distance", "Avg VTAT", "Avg CTAT",
                "Driver Ratings", "Customer Rating"]:
        assert pd.api.types.is_float_dtype(cleaned_df[col]), (
            f"Column '{col}' is not float64."
        )


def test_categorical_dtypes(cleaned_df):
    for col in ["Booking Status", "Vehicle Type"]:
        assert hasattr(cleaned_df[col], "cat"), f"Column '{col}' is not categorical."
