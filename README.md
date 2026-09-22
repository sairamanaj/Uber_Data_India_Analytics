# Ride-Hailing Growth, Customer Retention & Operational Analytics — Uber Data India

A complete, four-tier analytics project built on the NCR ride-bookings dataset (150,000 records, Jan–Dec 2024).

---

## Dataset

link - https://www.kaggle.com/datasets/anilrohan/uber-data-india

| Property | Value |
|---|---|
| File | `Data/ncr_ride_bookings.csv` |
| Rows | 150,000 ride bookings |
| Period | 2024-01-01 → 2024-12-30 |
| Region | NCR (Delhi, Gurgaon, Noida, Faridabad, Ghaziabad) |
| Columns | 21 (Date, Time, Booking ID, Booking Status, Customer ID, Vehicle Type, Pickup/Drop Location, Avg VTAT, Avg CTAT, cancellation fields, Booking Value, Ride Distance, Driver/Customer Ratings, Payment Method) |

> **The raw CSV is read-only and is never modified.** All outputs are written to `outputs/`.

---

## Repository Structure

```
Uber_Data_India_Analytics/
├── Data/
│   └── ncr_ride_bookings.csv          # Raw source — READ ONLY
├── notebooks/
│   ├── 01_data_quality_audit.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda_descriptive.ipynb
│   ├── 04_diagnostic_analytics.ipynb
│   ├── 05_customer_segmentation.ipynb
│   ├── 06_cohort_retention.ipynb
│   ├── 07_location_analysis.ipynb
│   ├── 08_vehicle_fare_analysis.ipynb
│   ├── 09_time_demand_analysis.ipynb
│   ├── 10_cancellation_analysis.ipynb
│   ├── 11_predictive_ml.ipynb
│   └── 12_prescriptive_analytics.ipynb
├── src/
│   ├── data_loader.py        # Raw → cleaned parquet pipeline
│   ├── features.py           # Feature engineering + zone mapping
│   ├── segmentation.py       # RFM scoring + K-Means clustering
│   ├── cohort.py             # Cohort / retention matrix builder
│   ├── ml_pipeline.py        # ML training, evaluation, leakage prevention
│   └── utils.py              # Shared helpers
├── dashboard/
│   └── app.py                # Streamlit 9-page dashboard
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_ml_pipeline.py
├── outputs/                  # Generated (gitignore candidate)
│   ├── cleaned_data.parquet
│   ├── feature_store.parquet
│   ├── models/
│   ├── plots/
│   └── reports/
├── report/
│   └── Uber_India_Analytics_Report.docx
├── PROJECT_PLAN.md
├── requirements.txt
├── requirements-optional.txt
└── AGENTS.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Optional (only if Stage 2 ML is triggered after evaluating LR and Random Forest):

```bash
pip install -r requirements-optional.txt
```

---

## Running the Notebooks

Run in order — each notebook writes outputs consumed by the next:

```bash
jupyter notebook
```

| # | Notebook | Tier | Produces |
|---|---|---|---|
| 01 | Data Quality Audit | Pre | `outputs/reports/data_quality_summary.csv` |
| 02 | Data Cleaning | Pre | `outputs/cleaned_data.parquet` |
| 03 | EDA / Descriptive | 1 | `outputs/feature_store.parquet`, charts |
| 04 | Diagnostic Analytics | 2 | Diagnostic charts, vehicle/cancellation tables |
| 05 | Customer Segmentation | 2/3 | `outputs/reports/rfm_segments.csv`, cluster profiles |
| 06 | Cohort & Retention | 2 | Cohort heatmaps, retention curve |
| 07 | Location Analysis | 1/2 | Location performance table, zone flow |
| 08 | Vehicle & Fare Analysis | 1/2 | Vehicle diagnostics, fare distribution charts |
| 09 | Time-Based Demand | 1/2 | Hourly/monthly demand charts |
| 10 | Cancellation Analysis | 1/2 | Cancellation reason charts, monthly trend |
| 11 | Predictive ML | 3 | `outputs/models/`, `outputs/reports/ml_results_*.csv` |
| 12 | Prescriptive Analytics | 4 | `outputs/reports/prescriptions.csv` |

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard has 9 pages covering all analysis tiers. Notebooks 01–03 must be run first to generate the feature store.

---

## Running the Tests

```bash
pytest tests/ -v
```

> **Note:** Tests that exercise the full ML pipeline (`test_ml_pipeline.py`) require
> `outputs/cleaned_data.parquet` and `outputs/feature_store.parquet` to exist.
> Run notebooks 01–03 first, or run `python src/data_loader.py` and `python src/features.py`.

---

## Analytics Tier Summary

| Tier | Question | Notebooks |
|---|---|---|
| 1 — Descriptive | What happened? | 03, 07, 08, 09 |
| 2 — Diagnostic | Why did it happen? | 04, 05, 06, 10 |
| 3 — Predictive | What will happen? | 11 |
| 4 — Prescriptive | What should we do? | 12 |

---

## Key Findings

*(Populated after notebooks are executed)*

---

## Limitations & Unavailable Metrics

The dataset contains no driver ID, driver payout, commission rate, operating cost, acquisition channel, or marketing spend. The following metrics **cannot be computed**:

| Metric | Reason |
|---|---|
| CAC | No acquisition channel or marketing spend data |
| LTV | No cost-side data — net margin unknown |
| ROAS | No ad spend data |
| Driver payout | No payout rate or amount field |
| Platform commission | No commission rate field |
| Operating cost | No cost data of any kind |
| Contribution margin | Requires revenue minus variable cost |

All revenue figures in this project are **Gross Booking Value (GBV)** only.  
Per-customer cumulative revenue is labelled **OCV (Observed Customer Value)** — it is not LTV.
