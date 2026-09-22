"""
src/data_loader.py
------------------
Raw → cleaned pipeline for ncr_ride_bookings.csv.

Rules enforced:
- Raw CSV is NEVER modified or overwritten.
- "null" strings replaced with NaN.
- Triple-quotes stripped from Booking ID and Customer ID.
- Null alignment validated and discrepancies reported (not silently accepted).
- Row count deviation is reported, not asserted.
- Cleaned data written to outputs/cleaned_data.parquet.
"""

from __future__ import annotations

import hashlib
import pathlib
import warnings

import numpy as np
import pandas as pd

RAW_PATH = pathlib.Path("Data/ncr_ride_bookings.csv")
CLEAN_PATH = pathlib.Path("outputs/cleaned_data.parquet")
QUARANTINE_PATH = pathlib.Path("outputs/completed_nullfare_quarantine.csv")

EXPECTED_ROW_COUNT = 150_000

# Columns that must be null for non-Completed rows
_NULL_FOR_NON_COMPLETED = [
    "Booking Value",
    "Ride Distance",
    "Payment Method",
]

# Valid booking statuses
VALID_STATUSES = {
    "Completed",
    "Cancelled by Driver",
    "Cancelled by Customer",
    "No Driver Found",
    "Incomplete",
}


def _file_sha256(path: pathlib.Path) -> str:
    """Return SHA-256 hex digest of a file (used to verify raw CSV is unchanged)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> pd.DataFrame:
    """
    Read the raw CSV without any transformations.
    Returns a DataFrame with all columns as object dtype.
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)
    return df


def clean(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all cleaning transforms to a raw DataFrame.
    Does NOT touch the source file.

    Parameters
    ----------
    df : raw DataFrame from load_raw()
    verbose : print validation warnings to stdout

    Returns
    -------
    Cleaned DataFrame
    """
    df = df.copy()

    # ── 1. Row count validation ────────────────────────────────────────────────
    actual_rows = len(df)
    if actual_rows != EXPECTED_ROW_COUNT:
        warnings.warn(
            f"[DATA] Row count is {actual_rows:,}, expected {EXPECTED_ROW_COUNT:,}. "
            "Proceeding with actual count.",
            stacklevel=2,
        )
    elif verbose:
        print(f"[OK] Row count: {actual_rows:,}")

    # ── 2. Replace "null" strings with NaN ────────────────────────────────────
    df.replace("null", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)

    # ── 3. Strip triple-quotes from ID columns ────────────────────────────────
    for col in ("Booking ID", "Customer ID"):
        df[col] = df[col].str.strip('"')

    # ── 4. Parse datetime ─────────────────────────────────────────────────────
    df["datetime"] = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    unparsed = df["datetime"].isna().sum()
    if unparsed > 0:
        warnings.warn(
            f"[DATA] {unparsed} rows have unparseable datetime — set to NaT.",
            stacklevel=2,
        )

    # ── 5. Cast numeric columns ───────────────────────────────────────────────
    float_cols = ["Booking Value", "Ride Distance", "Avg VTAT", "Avg CTAT",
                  "Driver Ratings", "Customer Rating"]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    int8_cols = [
        "Cancelled Rides by Customer",
        "Cancelled Rides by Driver",
        "Incomplete Rides",
    ]
    for col in int8_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")

    # ── 6. Cast categoricals ──────────────────────────────────────────────────
    cat_cols = ["Booking Status", "Vehicle Type", "Payment Method"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    # ── 7. Validate Booking Status values ─────────────────────────────────────
    unexpected = set(df["Booking Status"].dropna().unique()) - VALID_STATUSES
    if unexpected:
        warnings.warn(
            f"[DATA] Unexpected Booking Status values found: {unexpected}",
            stacklevel=2,
        )

    # ── 8. Null alignment validation ──────────────────────────────────────────
    _validate_null_alignment(df, verbose=verbose)

    # ── 9. Duplicate Booking ID check ─────────────────────────────────────────
    dup_count = df["Booking ID"].duplicated().sum()
    if dup_count > 0:
        warnings.warn(
            f"[DATA] {dup_count} duplicate Booking IDs detected.",
            stacklevel=2,
        )
        if verbose:
            print(f"  Sample duplicates:\n{df[df['Booking ID'].duplicated(keep=False)][['Booking ID','datetime','Booking Status']].head()}")
    elif verbose:
        print(f"[OK] No duplicate Booking IDs.")

    # ── 10. Quarantine: Completed rows with null Booking Value ────────────────
    integrity_mask = (df["Booking Status"] == "Completed") & df["Booking Value"].isna()
    quarantine_count = integrity_mask.sum()
    if quarantine_count > 0:
        warnings.warn(
            f"[DATA] {quarantine_count} Completed rides have null Booking Value "
            "(data integrity violation). Quarantined to "
            f"{QUARANTINE_PATH}.",
            stacklevel=2,
        )
        QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df[integrity_mask].to_csv(QUARANTINE_PATH, index=False)
        # Remove from main dataset
        df = df[~integrity_mask].copy()

    if verbose:
        _print_null_summary(df)

    return df


def _validate_null_alignment(df: pd.DataFrame, *, verbose: bool) -> None:
    """
    Validate structural null patterns and warn on deviations.
    Does not raise; only warns.

    Actual data observation: Incomplete rides (9,000 rows) carry non-null
    Booking Value, Ride Distance, and Payment Method — the ride started and
    incurred a fare before ending abnormally.  Only the three cancellation /
    no-driver statuses are guaranteed to have null fare fields.
    """
    # Fare fields must be null for cancelled / no-driver statuses (verified from data).
    # Incomplete rides legitimately carry fare data.
    _NON_REVENUE_STATUSES = {"Cancelled by Customer", "Cancelled by Driver", "No Driver Found"}
    for col in _NULL_FOR_NON_COMPLETED:
        if col not in df.columns:
            continue
        violation = df[df["Booking Status"].isin(_NON_REVENUE_STATUSES)][col].notna().sum()
        if violation > 0:
            warnings.warn(
                f"[DATA] {violation} cancelled/no-driver rows have a non-null '{col}'. "
                "Expected null for these statuses.",
                stacklevel=3,
            )
        elif verbose:
            print(f"[OK] '{col}' null alignment correct for cancelled/no-driver statuses.")

    # Avg VTAT should only be null for "No Driver Found"
    vtat_null_statuses = (
        df[df["Avg VTAT"].isna()]["Booking Status"]
        .value_counts()
        .to_dict()
    )
    expected_vtat_null = {"No Driver Found"}
    # Filter to statuses with actual non-zero null VTAT counts
    # (value_counts on Categorical includes all categories, even with count=0)
    actual_vtat_null_statuses = {s for s, c in vtat_null_statuses.items() if c > 0}
    unexpected_vtat_null = actual_vtat_null_statuses - expected_vtat_null
    if unexpected_vtat_null:
        warnings.warn(
            f"[DATA] Avg VTAT is null for unexpected statuses: "
            f"{dict((s, vtat_null_statuses[s]) for s in unexpected_vtat_null)}",
            stacklevel=3,
        )
    elif verbose:
        print("[OK] 'Avg VTAT' null alignment: only 'No Driver Found' rows.")


def _print_null_summary(df: pd.DataFrame) -> None:
    """Print a concise null-count table."""
    null_counts = df.isna().sum()
    null_pct = (null_counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
    summary = summary[summary["null_count"] > 0].sort_values("null_count", ascending=False)
    print("\n[NULL SUMMARY]")
    print(summary.to_string())
    print()


def load_clean(*, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Load the cleaned dataset from parquet, rebuilding if necessary.

    Parameters
    ----------
    force_rebuild : re-run cleaning even if parquet exists
    """
    if CLEAN_PATH.exists() and not force_rebuild:
        return pd.read_parquet(CLEAN_PATH)

    raw_hash_before = _file_sha256(RAW_PATH)
    raw = load_raw()
    cleaned = clean(raw, verbose=True)
    raw_hash_after = _file_sha256(RAW_PATH)

    if raw_hash_before != raw_hash_after:
        raise RuntimeError(
            "[CRITICAL] Raw CSV was modified during cleaning — this must never happen. "
            "Aborting without writing output."
        )

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(CLEAN_PATH, index=False)
    print(f"[OK] Cleaned data written to {CLEAN_PATH}  shape={cleaned.shape}")
    return cleaned


if __name__ == "__main__":
    df = load_clean(force_rebuild=True)
    print(df.dtypes)
    print(df.head(3))
