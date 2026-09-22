"""
src/features.py
---------------
Feature engineering on the cleaned DataFrame.

Rules:
- Raw CSV is never touched here; all transforms operate on the cleaned parquet.
- EDA-level customer aggregates (customer_total_rides, customer_ocv, etc.)
  are computed on the FULL dataset and must NEVER be used as ML features.
- ML-safe per-customer rolling features are in src/ml_pipeline.py.
- Zone mapping is based on manual inspection of all 176 location names;
  methodology is documented in PROJECT_PLAN.md §2 Stage 2.
  Locations not found in the mapping are labelled "Unknown".
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

FEATURE_STORE_PATH = pathlib.Path("outputs/feature_store.parquet")

# ── Zone mapping (inspected from full 176-location list) ──────────────────────
# Methodology: zones follow NCR administrative boundaries.
# Ambiguous locations are assigned to the nearest administrative unit.
# Unmapped locations → "Unknown" (never force-assigned).

LOCATION_ZONE: dict[str, str] = {
    # Delhi Core — Central / South-Central Delhi
    "AIIMS": "Delhi Core",
    "Barakhamba Road": "Delhi Core",
    "Central Secretariat": "Delhi Core",
    "Chanakyapuri": "Delhi Core",
    "Chandni Chowk": "Delhi Core",
    "Connaught Place": "Delhi Core",
    "Delhi Gate": "Delhi Core",
    "Hauz Khas": "Delhi Core",
    "Hauz Rani": "Delhi Core",
    "IIT Delhi": "Delhi Core",
    "INA Market": "Delhi Core",
    "India Gate": "Delhi Core",
    "Indraprastha": "Delhi Core",
    "ITO": "Delhi Core",
    "Jama Masjid": "Delhi Core",
    "Jor Bagh": "Delhi Core",
    "Khan Market": "Delhi Core",
    "Lajpat Nagar": "Delhi Core",
    "Lal Quila": "Delhi Core",
    "Lok Kalyan Marg": "Delhi Core",
    "Mandi House": "Delhi Core",
    "Moolchand": "Delhi Core",
    "Nehru Place": "Delhi Core",
    "Paharganj": "Delhi Core",
    "Panchsheel Park": "Delhi Core",
    "Pragati Maidan": "Delhi Core",
    "Rajiv Chowk": "Delhi Core",
    "RK Puram": "Delhi Core",
    "Sarojini Nagar": "Delhi Core",
    "South Extension": "Delhi Core",
    "Udyog Bhawan": "Delhi Core",
    "Vinobapuri": "Delhi Core",
    "New Delhi Railway Station": "Delhi Core",
    "Ashram": "Delhi Core",
    "Bhikaji Cama Place": "Delhi Core",
    "Green Park": "Delhi Core",
    # Delhi North
    "Adarsh Nagar": "Delhi North",
    "Ashok Park Main": "Delhi North",
    "Ashok Vihar": "Delhi North",
    "Azadpur": "Delhi North",
    "Civil Lines Gurgaon": "Delhi North",   # Note: name refers to Civil Lines area, not Gurgaon
    "GTB Nagar": "Delhi North",
    "Jahangirpuri": "Delhi North",
    "Kanhaiya Nagar": "Delhi North",
    "Keshav Puram": "Delhi North",
    "Model Town": "Delhi North",
    "Netaji Subhash Place": "Delhi North",
    "Pitampura": "Delhi North",
    "Pulbangash": "Delhi North",
    "Rithala": "Delhi North",
    "Rohini": "Delhi North",
    "Rohini East": "Delhi North",
    "Rohini West": "Delhi North",
    "Samaypur Badli": "Delhi North",
    "Shastri Nagar": "Delhi North",
    "Vishwavidyalaya": "Delhi North",
    # Delhi East
    "Akshardham": "Delhi East",
    "Anand Vihar": "Delhi East",
    "Anand Vihar ISBT": "Delhi East",
    "Dilshad Garden": "Delhi East",
    "Jhilmil": "Delhi East",
    "Karkarduma": "Delhi East",
    "Kashmere Gate": "Delhi East",
    "Kashmere Gate ISBT": "Delhi East",
    "Laxmi Nagar": "Delhi East",
    "Mansarovar Park": "Delhi East",
    "Mayur Vihar": "Delhi East",
    "Nirman Vihar": "Delhi East",
    "Preet Vihar": "Delhi East",
    "Seelampur": "Delhi East",
    "Shahdara": "Delhi East",
    "Shastri Park": "Delhi East",
    "Welcome": "Delhi East",
    "Yamuna Bank": "Delhi East",
    # Delhi West / South-West
    "Dwarka Mor": "Delhi West",
    "Dwarka Sector 21": "Delhi West",
    "Janakpuri": "Delhi West",
    "Kirti Nagar": "Delhi West",
    "Madipur": "Delhi West",
    "Moti Nagar": "Delhi West",
    "Mundka": "Delhi West",
    "Nawada": "Delhi West",
    "Paschim Vihar": "Delhi West",
    "Peeragarhi": "Delhi West",
    "Punjabi Bagh": "Delhi West",
    "Rajouri Garden": "Delhi West",
    "Ramesh Nagar": "Delhi West",
    "Subhash Nagar": "Delhi West",
    "Tagore Garden": "Delhi West",
    "Tilak Nagar": "Delhi West",
    "Uttam Nagar": "Delhi West",
    # Delhi South
    "Aya Nagar": "Delhi South",
    "Badarpur": "Delhi South",
    "Chhatarpur": "Delhi South",
    "Chirag Delhi": "Delhi South",
    "Ghitorni": "Delhi South",
    "Ghitorni Village": "Delhi South",
    "Govindpuri": "Delhi South",
    "Greater Kailash": "Delhi South",
    "IGI Airport": "Delhi South",
    "IGNOU Road": "Delhi South",
    "Jasola": "Delhi South",
    "Kalkaji": "Delhi South",
    "Maidan Garhi": "Delhi South",
    "Malviya Nagar": "Delhi South",
    "Mehrauli": "Delhi South",
    "Munirka": "Delhi South",
    "Okhla": "Delhi South",
    "Qutub Minar": "Delhi South",
    "Saket": "Delhi South",
    "Saket A Block": "Delhi South",
    "Sarai Kale Khan": "Delhi South",
    "Saidulajab": "Delhi South",
    "Sultanpur": "Delhi South",
    "Tughlakabad": "Delhi South",
    "Vasant Kunj": "Delhi South",
    # Gurgaon Core
    "Ambience Mall": "Gurgaon",
    "Cyber Hub": "Gurgaon",
    "DLF City Court": "Gurgaon",
    "DLF Phase 3": "Gurgaon",
    "Golf Course Road": "Gurgaon",
    "Gurgaon Railway Station": "Gurgaon",
    "Gurgaon Sector 29": "Gurgaon",
    "Gurgaon Sector 56": "Gurgaon",
    "Hero Honda Chowk": "Gurgaon",
    "Huda City Centre": "Gurgaon",
    "IFFCO Chowk": "Gurgaon",
    "MG Road": "Gurgaon",
    "Old Gurgaon": "Gurgaon",
    "Palam Vihar": "Gurgaon",
    "Sadar Bazar Gurgaon": "Gurgaon",
    "Sikanderpur": "Gurgaon",
    "Udyog Vihar": "Gurgaon",
    "Udyog Vihar Phase 4": "Gurgaon",
    "Sushant Lok": "Gurgaon",
    "Ardee City": "Gurgaon",
    # Gurgaon Peripheral
    "Badshahpur": "Gurgaon Peripheral",
    "Basai Dhankot": "Gurgaon Peripheral",
    "Civil Lines Gurgaon": "Gurgaon Peripheral",
    "Gwal Pahari": "Gurgaon Peripheral",
    "IMT Manesar": "Gurgaon Peripheral",
    "Kadarpur": "Gurgaon Peripheral",
    "Khandsa": "Gurgaon Peripheral",
    "Kherki Daula Toll": "Gurgaon Peripheral",
    "Manesar": "Gurgaon Peripheral",
    "Narsinghpur": "Gurgaon Peripheral",
    "New Colony": "Gurgaon Peripheral",
    "Pataudi Chowk": "Gurgaon Peripheral",
    "Sohna Road": "Gurgaon Peripheral",
    "Subhash Chowk": "Gurgaon Peripheral",
    "Vatika Chowk": "Gurgaon Peripheral",
    # Noida
    "Botanical Garden": "Noida",
    "Greater Noida": "Noida",
    "Indirapuram": "Noida",
    "Kaushambi": "Noida",
    "Noida Extension": "Noida",
    "Noida Film City": "Noida",
    "Noida Sector 18": "Noida",
    "Noida Sector 62": "Noida",
    "Vaishali": "Noida",
    "Raj Nagar Extension": "Noida",
    # Delhi Core (additional)
    "Arjangarh": "Delhi South",       # South Delhi metro station
    "Inderlok": "Delhi North",         # North Delhi / Shastri Nagar area
    "Karol Bagh": "Delhi Core",        # Central Delhi
    "Patel Chowk": "Delhi Core",       # Central Secretariat area
    "Rajiv Nagar": "Delhi North",      # North/West Delhi area
    "Satguru Ram Singh Marg": "Delhi West",  # West Delhi arterial road
    "Shivaji Park": "Delhi West",      # West Delhi
    "Tis Hazari": "Delhi North",       # North Delhi courts area
    "Vidhan Sabha": "Delhi North",     # North Delhi / Civil Lines area
    # Faridabad
    "Faridabad Sector 15": "Faridabad",
    # Outer NCR
    "Bhiwadi": "Outer NCR",
    "Bahadurgarh": "Outer NCR",
    "Ghaziabad": "Outer NCR",
    "Meerut": "Outer NCR",
    "Panipat": "Outer NCR",
    "Sonipat": "Outer NCR",
}

# Civil Lines Gurgaon is ambiguous — could refer to either Delhi's Civil Lines area
# or Gurgaon. The raw name includes "Gurgaon", so assigned to Gurgaon Peripheral.
# Overrides the "Delhi North" entry above:
LOCATION_ZONE["Civil Lines Gurgaon"] = "Gurgaon Peripheral"

TIME_SLOT_LABELS = {
    range(0, 6): "Early Morning",
    range(6, 11): "Morning",
    range(11, 15): "Afternoon",
    range(15, 20): "Evening",
    range(20, 23): "Night",
    range(23, 24): "Late Night",
}

# Pre-build hour → label lookup
_HOUR_TO_SLOT: dict[int, str] = {}
for _rng, _label in TIME_SLOT_LABELS.items():
    for _h in _rng:
        _HOUR_TO_SLOT[_h] = _label


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime-derived columns. Requires 'datetime' column."""
    df = df.copy()
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # 0=Monday
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["month_name"] = dt.dt.month_name()
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = df["day_of_week"] >= 5
    df["time_slot"] = df["hour"].map(_HOUR_TO_SLOT).astype("category")
    df["date_only"] = dt.dt.normalize()          # midnight-truncated date
    return df


def add_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ride-level derived columns."""
    df = df.copy()
    df["fare_per_km"] = np.where(
        df["Booking Value"].notna() & df["Ride Distance"].notna() & (df["Ride Distance"] > 0),
        df["Booking Value"] / df["Ride Distance"],
        np.nan,
    )
    df["is_completed"] = df["Booking Status"] == "Completed"
    df["is_cancelled"] = df["Booking Status"].isin(
        ["Cancelled by Customer", "Cancelled by Driver"]
    )
    df["cancellation_party"] = df["Booking Status"].map({
        "Cancelled by Customer": "Customer",
        "Cancelled by Driver": "Driver",
        "No Driver Found": "System",
    })  # NaN for Completed / Incomplete
    df["route"] = df["Pickup Location"] + " → " + df["Drop Location"]
    return df


def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map pickup and drop locations to geographic zones."""
    df = df.copy()
    df["pickup_zone"] = df["Pickup Location"].map(LOCATION_ZONE).fillna("Unknown")
    df["drop_zone"] = df["Drop Location"].map(LOCATION_ZONE).fillna("Unknown")
    unmapped = df[df["pickup_zone"] == "Unknown"]["Pickup Location"].unique()
    if len(unmapped) > 0:
        import warnings
        warnings.warn(
            f"[ZONE] {len(unmapped)} pickup location(s) not in zone mapping — "
            f"labelled 'Unknown': {sorted(unmapped)[:10]}",
            stacklevel=2,
        )
    return df


def add_cohort_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cohort and repeat-customer columns."""
    df = df.copy()
    df = df.sort_values("datetime")

    # Per-customer first booking date
    first_booking = df.groupby("Customer ID")["datetime"].min().rename("first_booking_date")
    df = df.join(first_booking, on="Customer ID")

    df["cohort_month"] = df["first_booking_date"].dt.to_period("M")
    df["booking_month"] = df["datetime"].dt.to_period("M")
    df["months_since_first"] = (
        (df["booking_month"] - df["cohort_month"])
        .apply(lambda x: x.n if pd.notna(x) else np.nan)
    )
    return df


def add_customer_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EDA-level per-customer aggregate columns.

    WARNING: These aggregates use the FULL dataset.
    They must NOT be used as ML features (target leakage risk).
    Clearly labelled 'eda_' prefix to make this visible.
    """
    df = df.copy()

    rides = df.groupby("Customer ID").size().rename("eda_customer_total_rides")
    completed_rides = (
        df[df["is_completed"]]
        .groupby("Customer ID")
        .size()
        .rename("eda_customer_completed_rides")
    )
    ocv = (
        df[df["is_completed"]]
        .groupby("Customer ID")["Booking Value"]
        .sum()
        .rename("eda_customer_ocv")           # Observed Customer Value — NOT LTV
    )
    avg_fare = (
        df[df["is_completed"]]
        .groupby("Customer ID")["Booking Value"]
        .mean()
        .rename("eda_customer_avg_fare")
    )

    for series in (rides, completed_rides, ocv, avg_fare):
        df = df.join(series, on="Customer ID")

    df["eda_customer_completed_rides"] = df["eda_customer_completed_rides"].fillna(0).astype(int)
    df["eda_customer_ocv"] = df["eda_customer_ocv"].fillna(0.0)

    return df


def build_feature_store(df: pd.DataFrame, *, save: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering transforms and return the enriched DataFrame.
    Optionally save to outputs/feature_store.parquet.
    """
    df = add_time_features(df)
    df = add_trip_features(df)
    df = add_zone_features(df)
    df = add_cohort_features(df)
    df = add_customer_eda_features(df)

    if save:
        out = pathlib.Path("outputs/feature_store.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[OK] Feature store written to {out}  shape={df.shape}")

    return df


if __name__ == "__main__":
    from src.data_loader import load_clean
    df_clean = load_clean()
    df_features = build_feature_store(df_clean, save=True)
    print(df_features.dtypes)
