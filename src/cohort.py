"""
src/cohort.py
-------------
Cohort and retention matrix builder.
All revenue figures are Gross Booking Value (GBV), not net revenue or LTV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cohort retention count matrix.

    Returns a DataFrame where:
        index   = cohort_month (Period[M])
        columns = months_since_first (0, 1, 2, …)
        values  = unique customers active in that relative month
    """
    cohort_data = (
        df.groupby(["cohort_month", "months_since_first"])["Customer ID"]
        .nunique()
        .reset_index()
        .rename(columns={"Customer ID": "active_customers"})
    )
    matrix = cohort_data.pivot(
        index="cohort_month",
        columns="months_since_first",
        values="active_customers",
    )
    matrix.index = matrix.index.astype(str)
    matrix.columns = [int(c) for c in matrix.columns]
    return matrix


def build_cohort_retention_rate(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Convert count matrix to retention rate (%).
    Divides each row by its cohort-0 value.
    """
    cohort_sizes = matrix[0]
    rate_matrix = matrix.div(cohort_sizes, axis=0) * 100
    return rate_matrix.round(1)


def build_cohort_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cohort GBV matrix.

    Returns a DataFrame where:
        index   = cohort_month
        columns = months_since_first
        values  = total Booking Value (GBV) from that cohort in that relative month
    """
    completed = df[df["is_completed"]].copy()
    cohort_rev = (
        completed.groupby(["cohort_month", "months_since_first"])["Booking Value"]
        .sum()
        .reset_index()
        .rename(columns={"Booking Value": "gbv"})
    )
    matrix = cohort_rev.pivot(
        index="cohort_month",
        columns="months_since_first",
        values="gbv",
    )
    matrix.index = matrix.index.astype(str)
    matrix.columns = [int(c) for c in matrix.columns]
    return matrix


def avg_retention_curve(rate_matrix: pd.DataFrame) -> pd.Series:
    """Return mean retention % per relative month across all cohorts."""
    return rate_matrix.mean(axis=0).rename("avg_retention_pct")
