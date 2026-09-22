"""
dashboard/app.py
----------------
Streamlit multi-page dashboard for Uber Data India Analytics.

Run:  streamlit run dashboard/app.py
      (from project root)

Pages:
  1. Executive Overview
  2. Demand & Time Analysis
  3. Vehicle & Fare Analysis
  4. Location Intelligence
  5. Cancellation Analytics
  6. Customer Analytics
  7. Cohort & Retention
  8. ML Model Results
  9. Prescriptive Insights

Data source: outputs/feature_store.parquet (read-only).
All revenue shown as Gross Booking Value (GBV) — not net revenue.
OCV = Observed Customer Value (NOT LTV).
Unavailable metrics: CAC, LTV, ROAS, driver payout, commission, operating cost.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uber India Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_STORE = pathlib.Path("outputs/feature_store.parquet")
SLOT_ORDER = ["Early Morning", "Morning", "Afternoon", "Evening", "Night", "Late Night"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading feature store…")
def load_data() -> pd.DataFrame:
    if not FEATURE_STORE.exists():
        st.error(
            "Feature store not found. Please run notebooks 01–03 first to generate "
            "`outputs/feature_store.parquet`."
        )
        st.stop()
    df = pd.read_parquet(FEATURE_STORE)
    # Ensure categorical columns are proper strings for Plotly
    for col in ["Booking Status", "Vehicle Type", "Payment Method", "time_slot"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_ml_results():
    cls_path = pathlib.Path("outputs/reports/ml_results_classification.csv")
    fare_path = pathlib.Path("outputs/reports/ml_results_fare.csv")
    cls_df = pd.read_csv(cls_path) if cls_path.exists() else None
    fare_df = pd.read_csv(fare_path) if fare_path.exists() else None
    return cls_df, fare_df


@st.cache_data(show_spinner=False)
def load_prescriptions():
    p = pathlib.Path("outputs/reports/prescriptions.csv")
    return pd.read_csv(p) if p.exists() else None


# ── Shared helpers ────────────────────────────────────────────────────────────
def kpi(label: str, value: str, delta: str | None = None):
    st.metric(label=label, value=value, delta=delta)


def unavailable_notice():
    st.caption(
        "⚠️ **Unavailable metrics:** CAC · LTV · ROAS · Driver Payout · Commission · "
        "Operating Cost · Contribution Margin — no cost fields in dataset."
    )


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🚖 Uber India Analytics")
st.sidebar.markdown("*NCR Ride Bookings — 2024*")
st.sidebar.markdown("---")

PAGES = [
    "1 · Executive Overview",
    "2 · Demand & Time Analysis",
    "3 · Vehicle & Fare Analysis",
    "4 · Location Intelligence",
    "5 · Cancellation Analytics",
    "6 · Customer Analytics",
    "7 · Cohort & Retention",
    "8 · ML Model Results",
    "9 · Prescriptive Insights",
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Source: `Data/ncr_ride_bookings.csv`")
st.sidebar.caption("Revenue shown as **Gross Booking Value (GBV)** only.")
st.sidebar.caption("OCV = Observed Customer Value — **not** LTV.")

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
completed = df[df["is_completed"]].copy()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("Executive Overview")
    unavailable_notice()

    total = len(df)
    n_comp = df["is_completed"].sum()
    comp_rate = n_comp / total * 100
    total_gbv = completed["Booking Value"].sum()
    avg_fare = completed["Booking Value"].mean()
    avg_dr = completed["Driver Ratings"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Total Bookings", f"{total:,}")
    with c2: kpi("Completion Rate", f"{comp_rate:.1f}%")
    with c3: kpi("Gross Booking Value", f"₹{total_gbv/1e6:.1f}M")
    with c4: kpi("Avg Fare / Ride", f"₹{avg_fare:.0f}")
    with c5: kpi("Avg Driver Rating", f"{avg_dr:.2f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        monthly_gbv = completed.groupby("month")["Booking Value"].sum().reset_index()
        monthly_gbv["month_label"] = monthly_gbv["month"].apply(lambda m: MONTH_LABELS[m - 1])
        fig = px.bar(monthly_gbv, x="month_label", y="Booking Value",
                     title="Monthly Gross Booking Value (GBV)",
                     labels={"Booking Value": "GBV (₹)", "month_label": "Month"},
                     color_discrete_sequence=["#1f77b4"])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        status_counts = df["Booking Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, names="Status", values="Count",
                     title="Booking Status Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    # Month-over-month volume change
    monthly_vol = df.groupby("month").size().reset_index(name="bookings")
    monthly_vol["mom_change_pct"] = monthly_vol["bookings"].pct_change() * 100
    monthly_vol["month_label"] = monthly_vol["month"].apply(lambda m: MONTH_LABELS[m - 1])
    fig = px.line(monthly_vol, x="month_label", y="bookings",
                  title="Monthly Booking Volume", markers=True,
                  labels={"bookings": "Bookings", "month_label": "Month"})
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Demand & Time Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("Demand & Time Analysis")

    tab1, tab2, tab3 = st.tabs(["Hourly Heatmap", "Time-Slot Breakdown", "Weekend vs Weekday"])

    with tab1:
        heat = df.pivot_table(index="hour", columns="day_name", values="Booking ID",
                              aggfunc="count", fill_value=0)
        heat = heat.reindex(columns=[d for d in DOW_ORDER if d in heat.columns])
        fig = px.imshow(heat, title="Booking Volume: Hour × Day of Week",
                        labels={"x": "Day", "y": "Hour", "color": "Bookings"},
                        color_continuous_scale="YlOrRd", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        slot_stats = df.groupby("time_slot").agg(
            Bookings=("Booking ID", "count"),
            Completed=("is_completed", "sum"),
            Avg_VTAT=("Avg VTAT", "mean"),
            Avg_Fare=("Booking Value", "mean"),
        ).round(2).reset_index()
        slot_stats["Completion_Rate"] = (slot_stats["Completed"] / slot_stats["Bookings"] * 100).round(1)
        slot_stats["time_slot"] = pd.Categorical(slot_stats["time_slot"], categories=SLOT_ORDER, ordered=True)
        slot_stats = slot_stats.sort_values("time_slot")

        metric = st.selectbox("Metric", ["Bookings", "Completion_Rate", "Avg_VTAT", "Avg_Fare"])
        fig = px.bar(slot_stats, x="time_slot", y=metric,
                     title=f"{metric} by Time Slot",
                     color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        we = df.groupby("is_weekend").agg(
            Bookings=("Booking ID", "count"),
            Completed=("is_completed", "sum"),
            GBV=("Booking Value", "sum"),
            Avg_Fare=("Booking Value", "mean"),
        ).round(2).reset_index()
        we["Label"] = we["is_weekend"].map({False: "Weekday", True: "Weekend"})
        we["Completion_Rate"] = (we["Completed"] / we["Bookings"] * 100).round(1)
        st.dataframe(we[["Label", "Bookings", "Completion_Rate", "GBV", "Avg_Fare"]].set_index("Label"))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Vehicle & Fare Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("Vehicle & Fare Analysis")
    st.caption("⚠️ Net revenue / driver payout / commission are unavailable — GBV only.")

    veh = df.groupby("Vehicle Type").agg(
        Total_Bookings=("Booking ID", "count"),
        Completed=("is_completed", "sum"),
        Avg_Fare=("Booking Value", "mean"),
        Avg_Distance=("Ride Distance", "mean"),
        Avg_VTAT=("Avg VTAT", "mean"),
        Avg_Driver_Rating=("Driver Ratings", "mean"),
        GBV=("Booking Value", "sum"),
    ).round(2).reset_index()
    veh["Completion_Rate"] = (veh["Completed"] / veh["Total_Bookings"] * 100).round(1)
    veh["Fare_per_km"] = (veh["Avg_Fare"] / veh["Avg_Distance"]).round(2)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(veh.sort_values("Total_Bookings", ascending=True),
                     x="Total_Bookings", y="Vehicle Type", orientation="h",
                     title="Total Bookings by Vehicle Type",
                     color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(veh.sort_values("Completion_Rate", ascending=True),
                     x="Completion_Rate", y="Vehicle Type", orientation="h",
                     title="Completion Rate by Vehicle Type (%)",
                     color_discrete_sequence=["#2ca02c"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Vehicle Summary Table")
    st.dataframe(veh[["Vehicle Type", "Total_Bookings", "Completion_Rate",
                       "Avg_Fare", "Avg_Distance", "Avg_VTAT",
                       "Avg_Driver_Rating", "Fare_per_km"]].set_index("Vehicle Type"))

    st.subheader("Fare Distribution by Vehicle Type")
    fig = px.box(completed, x="Vehicle Type", y="Booking Value",
                 title="Fare Distribution (₹) by Vehicle Type",
                 color="Vehicle Type",
                 labels={"Booking Value": "Fare (₹)"},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Location Intelligence
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("Location Intelligence")
    st.caption("Zone mapping: 176 locations → 10 NCR zones. Methodology in PROJECT_PLAN.md §2.")

    n_top = st.slider("Show top N locations", 10, 30, 15)

    col1, col2 = st.columns(2)
    with col1:
        top_p = df["Pickup Location"].value_counts().head(n_top).reset_index()
        top_p.columns = ["Location", "Rides"]
        fig = px.bar(top_p.sort_values("Rides"), x="Rides", y="Location", orientation="h",
                     title=f"Top {n_top} Pickup Locations", color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        top_d = df["Drop Location"].value_counts().head(n_top).reset_index()
        top_d.columns = ["Location", "Rides"]
        fig = px.bar(top_d.sort_values("Rides"), x="Rides", y="Location", orientation="h",
                     title=f"Top {n_top} Drop Locations", color_discrete_sequence=["#ff7f0e"])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Zone-Level Volume")
    zone_vol = df["pickup_zone"].value_counts().reset_index()
    zone_vol.columns = ["Zone", "Rides"]
    fig = px.pie(zone_vol, names="Zone", values="Rides",
                 title="Ride Volume by Pickup Zone",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Location Performance Table")
    loc_perf_path = pathlib.Path("outputs/reports/location_performance.csv")
    if loc_perf_path.exists():
        lp = pd.read_csv(loc_perf_path, index_col=0)
        st.dataframe(lp.head(30))
    else:
        st.info("Run Notebook 07 to generate the location performance table.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Cancellation Analytics
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
    st.title("Cancellation Analytics")

    n_total = len(df)
    n_cc = (df["Booking Status"] == "Cancelled by Customer").sum()
    n_cd = (df["Booking Status"] == "Cancelled by Driver").sum()
    n_ndf = (df["Booking Status"] == "No Driver Found").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Cancelled by Customer", f"{n_cc:,} ({n_cc/n_total*100:.1f}%)")
    with c2: kpi("Cancelled by Driver", f"{n_cd:,} ({n_cd/n_total*100:.1f}%)")
    with c3: kpi("No Driver Found", f"{n_ndf:,} ({n_ndf/n_total*100:.1f}%)")
    with c4: kpi("Total Failed", f"{n_cc+n_cd+n_ndf:,} ({(n_cc+n_cd+n_ndf)/n_total*100:.1f}%)")

    col1, col2 = st.columns(2)
    with col1:
        cr = (df[df["Booking Status"] == "Cancelled by Customer"]
              ["Reason for cancelling by Customer"].value_counts().reset_index())
        cr.columns = ["Reason", "Count"]
        fig = px.bar(cr, x="Count", y="Reason", orientation="h",
                     title="Customer Cancellation Reasons",
                     color_discrete_sequence=["#d62728"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dr = (df[df["Booking Status"] == "Cancelled by Driver"]
              ["Driver Cancellation Reason"].value_counts().reset_index())
        dr.columns = ["Reason", "Count"]
        fig = px.bar(dr, x="Count", y="Reason", orientation="h",
                     title="Driver Cancellation Reasons",
                     color_discrete_sequence=["#1f77b4"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # Monthly cancellation rate trend
    monthly_cr = df.groupby("month").apply(
        lambda g: g["is_cancelled"].sum() / len(g) * 100
    ).reset_index(name="canc_rate")
    monthly_cr["month_label"] = monthly_cr["month"].apply(lambda m: MONTH_LABELS[m - 1])
    fig = px.line(monthly_cr, x="month_label", y="canc_rate",
                  title="Monthly Cancellation Rate (%)", markers=True,
                  labels={"canc_rate": "Cancellation Rate (%)", "month_label": "Month"},
                  color_discrete_sequence=["#d62728"])
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Customer Analytics
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("Customer Analytics")
    st.caption("OCV = Observed Customer Value (sum of fares on completed rides). **Not** LTV.")
    unavailable_notice()

    rides_per_cust = df.groupby("Customer ID").size()
    repeat_rate = (rides_per_cust > 1).mean() * 100

    c1, c2, c3 = st.columns(3)
    with c1: kpi("Unique Customers", f"{len(rides_per_cust):,}")
    with c2: kpi("Repeat Customer Rate", f"{repeat_rate:.1f}%")
    with c3:
        avg_ocv = completed.groupby("Customer ID")["Booking Value"].sum().mean()
        kpi("Avg OCV per Customer", f"₹{avg_ocv:.0f}")

    col1, col2 = st.columns(2)
    with col1:
        dist = rides_per_cust.clip(upper=8).value_counts().sort_index().reset_index()
        dist.columns = ["Rides", "Customers"]
        fig = px.bar(dist, x="Rides", y="Customers",
                     title="Rides per Customer Distribution (capped at 8)",
                     color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rfm_path = pathlib.Path("outputs/reports/rfm_segments.csv")
        if rfm_path.exists():
            rfm = pd.read_csv(rfm_path)
            seg_counts = rfm["rfm_segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Customers"]
            fig = px.bar(seg_counts.sort_values("Customers", ascending=True),
                         x="Customers", y="Segment", orientation="h",
                         title="RFM Segment Sizes",
                         color_discrete_sequence=["#2ca02c"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run Notebook 05 to generate RFM segments.")

    cluster_path = pathlib.Path("outputs/reports/cluster_profiles.csv")
    if cluster_path.exists():
        st.subheader("Customer Cluster Profiles")
        cp = pd.read_csv(cluster_path, index_col=0)
        st.dataframe(cp.round(2))
    else:
        st.info("Run Notebook 05 to generate cluster profiles.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Cohort & Retention
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[6]:
    st.title("Cohort & Retention Analysis")
    st.caption("All revenue = Gross Booking Value (GBV). Not net revenue.")

    ret_path = pathlib.Path("outputs/reports/cohort_retention_rate.csv")
    gbv_path = pathlib.Path("outputs/reports/cohort_gbv.csv")

    if ret_path.exists():
        rate_mat = pd.read_csv(ret_path, index_col=0)
        rate_mat.columns = rate_mat.columns.astype(int)

        st.subheader("Cohort Retention Rate (%)")
        fig = px.imshow(rate_mat.fillna(0),
                        title="Cohort Retention Rate (%)",
                        labels={"x": "Months Since First Ride", "y": "Cohort Month", "color": "%"},
                        color_continuous_scale="YlOrRd_r",
                        text_auto=".1f", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

        # Retention curve
        curve = rate_mat.mean(axis=0).reset_index()
        curve.columns = ["Month", "Avg_Retention"]
        fig = px.line(curve, x="Month", y="Avg_Retention",
                      title="Average Retention Curve (all cohorts)", markers=True,
                      labels={"Avg_Retention": "Avg Retention (%)", "Month": "Months Since First Ride"},
                      color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Notebook 06 to generate cohort matrices.")

    if gbv_path.exists():
        gbv_mat = pd.read_csv(gbv_path, index_col=0)
        gbv_mat.columns = gbv_mat.columns.astype(int)
        st.subheader("Cohort Gross Booking Value (₹ thousands)")
        fig = px.imshow((gbv_mat.fillna(0) / 1000).round(0),
                        title="Cohort GBV (₹ thousands)",
                        labels={"x": "Months Since First Ride", "y": "Cohort Month", "color": "₹K"},
                        color_continuous_scale="Blues", text_auto=".0f", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — ML Model Results
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[7]:
    st.title("ML Model Results")
    st.caption("Time-based split: train ≤ Sep 2024 / test ≥ Oct 2024. No hardcoded pass/fail threshold.")

    cls_df, fare_df = load_ml_results()

    if cls_df is not None:
        st.subheader("Task A — Completion Prediction (Classification)")
        display_cols = [c for c in ["model","roc_auc","avg_precision","precision_1",
                                     "recall_1","f1_1"] if c in cls_df.columns]
        st.dataframe(cls_df[display_cols].set_index("model"))

        for _, row in cls_df.iterrows():
            auc = row.get("roc_auc", 0)
            if auc >= 0.75:
                interp = "Strong discrimination — suitable for dispatch prioritisation."
            elif auc >= 0.65:
                interp = "Moderate discrimination — useful for risk flagging."
            elif auc >= 0.55:
                interp = "Weak discrimination — limited operational use."
            else:
                interp = "Near-random — not useful operationally; review feature set."
            st.markdown(f"**{row['model']}** (AUC={auc:.3f}): {interp}")
    else:
        st.info("Run Notebook 11 to generate ML results.")

    if fare_df is not None:
        st.subheader("Task B — Fare Estimation (Regression)")
        fare_display = [c for c in ["model","mae","rmse","r2"] if c in fare_df.columns]
        st.dataframe(fare_df[fare_display].set_index("model"))
    else:
        st.info("Run Notebook 11 to generate fare estimation results.")

    # Feature importance (if saved)
    st.markdown("---")
    st.info("Feature importance charts and confusion matrices are saved to `outputs/plots/11_*.png`. "
            "Open those files or run Notebook 11 interactively to view them.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — Prescriptive Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == PAGES[8]:
    st.title("Prescriptive Insights")
    unavailable_notice()

    presc_df = load_prescriptions()
    if presc_df is not None:
        st.subheader("Prescription Summary")
        st.dataframe(presc_df.set_index("area"))
    else:
        st.info("Run Notebook 12 to generate prescriptions.")

    st.markdown("---")
    st.subheader("Supply Gap Scenario Explorer")
    st.caption("Scenario estimate: upper-bound only. Assumes recovered NDF rides complete at median fare.")

    median_fare = float(completed["Booking Value"].median()) if len(completed) > 0 else 414.0
    reduction_pct = st.slider("NDF rate reduction (%)", min_value=5, max_value=50, value=20, step=5)

    ndf_count = (df["Booking Status"] == "No Driver Found").sum()
    scenario_rides = int(ndf_count * reduction_pct / 100)
    scenario_gbv = scenario_rides * median_fare

    c1, c2, c3 = st.columns(3)
    with c1: kpi("Total NDF Bookings", f"{ndf_count:,}")
    with c2: kpi(f"Recovered Rides ({reduction_pct}% scenario)", f"{scenario_rides:,}")
    with c3: kpi("Estimated GBV Uplift", f"₹{scenario_gbv/1e6:.2f}M")

    st.warning(
        "⚠️ This is a **hypothetical upper-bound scenario**, not a forecast. "
        "It assumes every recovered NDF ride would complete at the median fare (₹"
        f"{median_fare:.0f}). Actual recoverable GBV would be lower. "
        "**Net revenue impact cannot be estimated** — no cost data is available."
    )

    st.markdown("---")
    st.subheader("Unavailable Metrics Documentation")
    unavailable = {
        "CAC (Customer Acquisition Cost)": "No acquisition channel or marketing spend field in dataset.",
        "LTV (Customer Lifetime Value)": "No cost-side data; cannot compute net margin.",
        "ROAS (Return on Ad Spend)": "No ad spend data in dataset.",
        "Driver Payout": "No payout rate or payout amount field.",
        "Platform Commission": "No commission rate field.",
        "Operating Cost": "No cost data of any kind.",
        "Contribution Margin": "Requires revenue minus variable cost; cost fields absent.",
    }
    for metric, reason in unavailable.items():
        st.markdown(f"**{metric}:** {reason}")
