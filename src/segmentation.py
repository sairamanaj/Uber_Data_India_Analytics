"""
src/segmentation.py
-------------------
RFM scoring and K-Means customer segmentation.
Uses only EDA-level data — not for ML features.
OCV (Observed Customer Value) is used as the M dimension; it is NOT LTV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REFERENCE_DATE = pd.Timestamp("2024-12-30")

RFM_SEGMENT_MAP: dict[str, str] = {
    # (R score, F score) → label  — simplified rules applied post-scoring
    # Full scoring uses combined RFM string; see rfm_label()
}


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM table at customer level.

    Columns produced:
        customer_id, recency_days, frequency, monetary_ocv,
        R, F, M (1–4 scores), rfm_score, rfm_segment

    monetary_ocv = sum of Booking Value on completed rides.
    Labelled 'OCV' (Observed Customer Value); NOT lifetime value.
    """
    completed = df[df["is_completed"]].copy()

    recency = (
        df.groupby("Customer ID")["datetime"]
        .max()
        .sub(REFERENCE_DATE)
        .dt.days.abs()
        .rename("recency_days")
    )
    frequency = df.groupby("Customer ID").size().rename("frequency")
    monetary = (
        completed.groupby("Customer ID")["Booking Value"]
        .sum()
        .rename("monetary_ocv")
    )

    rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
    rfm["monetary_ocv"] = rfm["monetary_ocv"].fillna(0.0)

    # Score 1–4 using quantiles (higher = better for F and M; lower recency = better)
    rfm["R"] = pd.qcut(rfm["recency_days"], q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    # Monetary: customers with 0 OCV get score 1; rest scored on quantiles
    rfm["M"] = _score_monetary(rfm["monetary_ocv"])

    rfm["rfm_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["rfm_segment"] = rfm.apply(_rfm_label, axis=1)

    return rfm


def _score_monetary(series: pd.Series) -> pd.Series:
    """Score monetary column 1–4, treating 0-OCV customers as score 1."""
    scores = pd.Series(1, index=series.index, dtype=int)
    non_zero = series[series > 0]
    if len(non_zero) >= 4:
        cut = pd.qcut(non_zero.rank(method="first"), q=3, labels=[2, 3, 4])
        scores.loc[non_zero.index] = cut.astype(int)
    return scores


def _rfm_label(row: pd.Series) -> str:
    r, f, m = row["R"], row["F"], row["M"]
    score = r + f + m
    if r >= 4 and f >= 4 and m >= 4:
        return "Champion"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 3 and f <= 2:
        return "Potential Loyalist"
    if r >= 3 and f == 1:
        return "New Customer"
    if r == 2 and f >= 2:
        return "At Risk"
    if r == 2 and f == 1:
        return "Hibernating"
    return "Lost"


def build_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-customer feature matrix for K-Means clustering.
    Only includes customers with at least 1 completed ride.
    """
    completed = df[df["is_completed"]].copy()

    total_rides = df.groupby("Customer ID").size().rename("total_rides")
    completed_rides = completed.groupby("Customer ID").size().rename("completed_rides")
    total_bv = completed.groupby("Customer ID")["Booking Value"].sum().rename("total_bv")
    avg_bv = completed.groupby("Customer ID")["Booking Value"].mean().rename("avg_bv")
    avg_dr = completed.groupby("Customer ID")["Driver Ratings"].mean().rename("avg_driver_rating_given")

    # Preferred vehicle type (mode)
    pref_vehicle = (
        completed.groupby("Customer ID")["Vehicle Type"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
        .rename("pref_vehicle")
    )

    # Preferred time slot (mode)
    pref_slot = (
        df.groupby("Customer ID")["time_slot"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
        .rename("pref_time_slot")
    )

    features = pd.concat(
        [total_rides, completed_rides, total_bv, avg_bv, avg_driver_rating_given, pref_vehicle, pref_slot],
        axis=1,
    ).dropna(subset=["total_bv"])   # only customers with ≥1 completed ride

    features["completed_rate"] = features["completed_rides"] / features["total_rides"]

    # Frequency-encode categorical columns on the cluster feature set itself
    for col in ("pref_vehicle", "pref_time_slot"):
        freq = features[col].value_counts(normalize=True)
        features[col + "_enc"] = features[col].map(freq)
    features = features.drop(columns=["pref_vehicle", "pref_time_slot"])

    return features


def run_kmeans(
    features: pd.DataFrame,
    k_range: range = range(2, 8),
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Scale features, run K-Means for each k in k_range, return best model
    (selected by silhouette score).

    Returns (best_model, labels, scaler, elbow_df)
    where elbow_df has columns [k, inertia, silhouette].
    """
    from sklearn.metrics import silhouette_score

    numeric_cols = features.select_dtypes(include="number").columns.tolist()
    X = features[numeric_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    elbow_df = pd.DataFrame(results)
    best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])

    best_model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    best_labels = best_model.fit_predict(X_scaled)

    return best_model, best_labels, scaler, elbow_df


def pca_2d(X_scaled: np.ndarray) -> np.ndarray:
    """Reduce scaled feature matrix to 2 PCA components for visualisation."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled)
