# AGENTS.md

This file provides guidance to agents when working with code in this repository.

---

## Project Type

Pure Python data analytics project. No web framework, no package manager beyond pip.  
Single data source: `Data/ncr_ride_bookings.csv` (150,000 rows, 21 columns, Jan–Dec 2024 NCR rides).

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Optional ML libs (only if Stage 2 ML is triggered after evaluating LR + RF)
pip install -r requirements-optional.txt

# Run notebooks (must be in order 01→12)
jupyter notebook

# Run dashboard (from project root)
streamlit run dashboard/app.py

# Run tests (requires outputs/cleaned_data.parquet and feature_store.parquet to exist first)
pytest tests/ -v

# Rebuild cleaned data and feature store without Jupyter
python src/data_loader.py
python src/features.py
```

---

## Critical Non-Obvious Rules

### Data
- `Data/ncr_ride_bookings.csv` is **read-only**. No writes, no overwrite, no in-place modification ever.  
  `src/data_loader.py` verifies via SHA-256 hash before and after cleaning.
- `Booking ID` and `Customer ID` are triple-quoted (`"""CNR1234"""`) in the raw CSV — always `.str.strip('"')` before use.
- String `"null"` (not Python `None`) is used throughout the raw CSV — replace with `np.nan` before any numeric cast.
- `Incomplete` rides (9,000 rows) carry **non-null** `Booking Value`, `Ride Distance`, and `Payment Method` — the ride started and incurred a fare. Only `Cancelled by Customer`, `Cancelled by Driver`, and `No Driver Found` rows are guaranteed null on fare fields.
- `Driver Ratings` and `Customer Rating` are null for non-Completed and most Incomplete rows — never impute them.

### Notebooks
- All 12 notebooks require `nbformat=4.5` with a unique `"id"` (8-char) on every cell. Without `id`, VS Code's Jupyter extension refuses to open the file with a misleading "file not found" error.
- Notebooks must run top-to-bottom in order: 01→02 produces `cleaned_data.parquet`, 03 produces `feature_store.parquet`, which all later notebooks consume.
- If you create or rewrite a notebook, use the pattern in `make_notebooks.py` (deleted after use) which assigns `uuid.uuid4()[:8]` ids.

### ML / Leakage
- `EXCLUDED_COLUMNS` in `src/ml_pipeline.py` lists all post-outcome columns — never add them to any feature matrix for Task A (completion prediction).
- Customer history features (`customer_prior_*`) must use `add_rolling_customer_features()` which sorts by `datetime` and shifts — never use the EDA aggregates (`eda_customer_*`) as ML features.
- Train/test split is **always time-based**: train ≤ 2024-09-30, test ≥ 2024-10-01. No random splits anywhere.
- `add_rolling_customer_features()` assigns results via `.loc[grp.index]` (not positional list extension) — customer rows are not contiguous in the datetime-sorted dataframe, so positional assignment would misalign.
- Target encoding for `Pickup Location` / `Drop Location` is fitted **on the training set only** via `fit_target_encoding()`.

### Terminology
- Revenue is always **Gross Booking Value (GBV)** — the customer fare. Never "net revenue" or "revenue".
- Per-customer cumulative revenue is **OCV (Observed Customer Value)** — never "LTV".
- **CAC, LTV, ROAS, driver payout, commission, operating cost, contribution margin are unavailable** (no cost fields). Document as unavailable; never fabricate a proxy.

### Zone Mapping
- The 176-location → 10-zone mapping lives in `LOCATION_ZONE` dict in `src/features.py`. Unmapped locations get `"Unknown"` — never force-assign. If adding new locations, extend the dict there.

### Stage 2 ML
- `xgboost`, `lightgbm`, `shap` are in `requirements-optional.txt` and are **not installed by default**. Only install if Logistic Regression + Random Forest Stage 1 results justify gradient boosting. Decision must be documented in Notebook 11.

### Tests
- `test_ml_pipeline.py::test_no_hardcoded_auc_threshold` scans `src/ml_pipeline.py` with regex — do not add any `assert.*roc_auc.*>.*0.6` style assertions to that module.
- Tests that exercise the full pipeline require parquet files to exist — run notebooks 01–03 first.
