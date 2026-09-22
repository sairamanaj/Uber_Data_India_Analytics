# PROJECT_PLAN.md

## Ride-Hailing Growth, Customer Retention & Operational Analytics — Uber Data India

> **Status:** Plan — awaiting approval before implementation.
> Last updated with corrections: 2025-01 (11-point review incorporated — see §12).

---

## 0. Dataset Ground Truth

**Source file:** `Data/ncr_ride_bookings.csv` — **read-only; never modified or overwritten**  
**Rows:** 150,000 (validated at load time; discrepancies reported, not assumed)  
**Date range:** 2024-01-01 → 2024-12-30 (full calendar year, verified)  
**Region:** NCR — Delhi, Gurgaon, Noida, Faridabad, Ghaziabad, plus peripheral locations

### Confirmed Columns

| Column | Dtype (post-clean) | Notes |
|---|---|---|
| `Date` | string → `datetime` | Booking date |
| `Time` | string → `datetime` | Combined with Date |
| `Booking ID` | string | Triple-quoted `"""CNRxxxxxxx"""` — strip required |
| `Booking Status` | category | 5 values (see below) |
| `Customer ID` | string | Triple-quoted `"""CIDxxxxxxx"""` — strip required |
| `Vehicle Type` | category | 7 types |
| `Pickup Location` | string | 176 unique values (inspected — see §2 Stage 2) |
| `Drop Location` | string | 176 unique values |
| `Avg VTAT` | float / null | Null only for "No Driver Found" (10,500 rows); all other statuses have values |
| `Avg CTAT` | float / null | Null for "No Driver Found" and all cancellation statuses (48,000 rows) |
| `Cancelled Rides by Customer` | Int8 / null | 1 iff status = Cancelled by Customer |
| `Reason for cancelling by Customer` | string / null | 5 categories |
| `Cancelled Rides by Driver` | Int8 / null | 1 iff status = Cancelled by Driver |
| `Driver Cancellation Reason` | string / null | 4 categories |
| `Incomplete Rides` | Int8 / null | 1 iff status = Incomplete |
| `Incomplete Rides Reason` | string / null | 3 categories |
| `Booking Value` | float / null | INR; null for all non-Completed rows (57,000) |
| `Ride Distance` | float / null | km; null for all non-Completed rows |
| `Driver Ratings` | float / null | Null for non-Completed + some Incomplete (57,000 rows) |
| `Customer Rating` | float / null | Same null pattern as Driver Ratings |
| `Payment Method` | category / null | Null for all non-Completed rows |

### Booking Status Distribution (confirmed)

| Status | Count | % |
|---|---|---|
| Completed | 93,000 | 62.0% |
| Cancelled by Driver | 27,000 | 18.0% |
| Cancelled by Customer | 10,500 | 7.0% |
| No Driver Found | 10,500 | 7.0% |
| Incomplete | 9,000 | 6.0% |

### Vehicle Types (confirmed)
`Auto` (37,419) · `Go Mini` (29,806) · `Go Sedan` (27,141) · `Bike` (22,517) · `Premier Sedan` (18,111) · `eBike` (10,557) · `Uber XL` (4,449)

### Payment Methods (completed rides only, confirmed)
`UPI` (45,909) · `Cash` (25,367) · `Uber Wallet` (12,276) · `Credit Card` (10,209) · `Debit Card` (8,239)

### Measured Distributions (from actual data — not assumed)

| Metric | Count | Min | P25 | P50 | P75 | P95 | Max |
|---|---|---|---|---|---|---|---|
| `Avg VTAT` (min) | 139,500 | 2 | 5.3 | 8.3 | 11.3 | 14.6 | 20 |
| `Avg CTAT` (min) | 102,000 | 10 | 21.6 | 28.8 | 36.8 | 43.4 | 45 |
| `Booking Value` (INR) | 102,000 | 50 | 234 | 414 | 689 | 1,224 | 4,277 |
| `Ride Distance` (km) | 102,000 | 1 | 12.46 | 23.72 | 36.82 | 47.35 | 50 |

> These ranges are **data-derived facts**, not hardcoded assumptions. Any range checks
> in the data quality audit will reference these measured values and flag actual outliers
> (values outside the observed min/max), not assumed ranges.

### Null Alignment (validated)
- `Avg VTAT` is null for **exactly** "No Driver Found" (10,500); present for all other statuses.
- `Avg CTAT` is null for "No Driver Found" + all Cancelled statuses (48,000 total).
- `Booking Value`, `Ride Distance`, `Payment Method`: null for all non-Completed (57,000).
- `Driver Ratings`, `Customer Rating`: null for non-Completed + some Incomplete (57,000 total).

### Unsupported Metrics — Documented as Unavailable

The dataset contains no driver ID, driver payout, commission rate, operating cost, acquisition channel, or marketing spend. The following metrics **cannot be computed** and will be documented as unavailable in every notebook and the final report:

| Metric | Why unavailable |
|---|---|
| **CAC** (Customer Acquisition Cost) | No acquisition channel or marketing spend field |
| **LTV** (Customer Lifetime Value) | No cost-side data; cannot compute net margin |
| **ROAS** (Return on Ad Spend) | No ad spend data |
| **Driver payout** | No payout rate or payout amount field |
| **Platform commission** | No commission rate field |
| **Operating cost** | No cost data of any kind |
| **Contribution margin** | Requires revenue minus variable cost; cost absent |

> **No proxy will be fabricated for these metrics.** Where revenue is discussed,
> it is labelled "Gross Booking Value" (total fare paid by customers). Where
> per-customer cumulative revenue is discussed, it is labelled
> **"Observed Customer Value (OCV)"** — defined strictly as the sum of
> `Booking Value` for completed rides attributed to that customer within the
> observation window. OCV is **not** LTV; it makes no forward projection
> and no claim about profitability.

---

## 1. Repository Structure

```
Uber_Data_India_Analytics/
│
├── Data/
│   └── ncr_ride_bookings.csv          # Raw source — READ ONLY, never modified
│
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
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Raw → cleaned parquet pipeline
│   ├── features.py             # Feature engineering functions
│   ├── segmentation.py         # RFM + clustering logic
│   ├── cohort.py               # Cohort/retention matrix builder
│   ├── ml_pipeline.py          # Model training and evaluation
│   └── utils.py                # Shared helpers
│
├── dashboard/
│   └── app.py                  # Streamlit multi-page dashboard
│
├── outputs/
│   ├── cleaned_data.parquet    # Cleaned dataset (derived, not raw)
│   ├── feature_store.parquet   # ML-ready features
│   ├── models/                 # Saved model artifacts (.pkl / .joblib)
│   ├── plots/                  # Static chart exports (.png)
│   └── reports/                # Auto-generated summary tables (.csv)
│
├── report/
│   └── Uber_India_Analytics_Report.docx
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_ml_pipeline.py
│
├── PROJECT_PLAN.md
├── README.md
├── requirements.txt
└── AGENTS.md
```

---

## 2. Data Pipeline

### Stage 1 — Raw Ingestion (`src/data_loader.py`)

1. Read `Data/ncr_ride_bookings.csv` with `pandas.read_csv()`. File is never written to.
2. **Validate row count**: print actual row count and warn if it differs from 150,000 (do not assert-fail; report the discrepancy).
3. **Strip triple-quotes** from `Booking ID` and `Customer ID` using `.str.strip('"')`.
4. **Parse datetime**: combine `Date` + `Time` → single `datetime` column (`pd.to_datetime`).
5. **Replace "null" strings** with `np.nan` across all columns.
6. **Cast dtypes**:
   - `Booking Value`, `Ride Distance`, `Avg VTAT`, `Avg CTAT` → `float64`
   - `Driver Ratings`, `Customer Rating` → `float64`
   - `Cancelled Rides by Customer`, `Cancelled Rides by Driver`, `Incomplete Rides` → `Int8` (nullable)
   - `Booking Status`, `Vehicle Type`, `Payment Method` → `pd.Categorical`
7. **Validate null alignment** (report deviations, do not silently accept):
   - Warn if any `Booking Value` is non-null for a non-Completed row.
   - Warn if `Avg VTAT` is null for any status other than "No Driver Found".
8. **Check for duplicate Booking IDs**; report count (do not silently drop).
9. Save → `outputs/cleaned_data.parquet`. Raw CSV untouched.

### Stage 2 — Location Zone Mapping (`src/data_loader.py` or `src/features.py`)

The 176 unique location names have been manually inspected (full list verified in §0). A zone mapping is **defensible** based on well-known NCR geography:

| Zone | Basis | Representative locations |
|---|---|---|
| Delhi Core | Central / South Delhi landmarks | Connaught Place, India Gate, Khan Market, AIIMS, Hauz Khas, Saket, Lajpat Nagar, Karol Bagh, Paharganj, ITO, Barakhamba Road, Rajiv Chowk, Central Secretariat, Chandni Chowk, Jama Masjid, etc. |
| Delhi North | North Delhi areas | Rohini, Rohini East/West, Pitampura, Netaji Subhash Place, GTB Nagar, Azadpur, Model Town, Jahangirpuri, Rithala, Samaypur Badli, Adarsh Nagar, Civil Lines area, Ashok Vihar, etc. |
| Delhi East | East Delhi / Trans-Yamuna | Mayur Vihar, Laxmi Nagar, Shahdara, Preet Vihar, Dilshad Garden, Anand Vihar, Karkarduma, Nirman Vihar, Seelampur, Shastri Park, Yamuna Bank, Akshardham, Indirapuram, Vaishali, etc. |
| Delhi West / SW | West and South-West Delhi | Dwarka (Sector 21, Mor), Janakpuri, Uttam Nagar, Nawada, Paschim Vihar, Peeragarhi, Mundka, Tagore Garden, Rajouri Garden, Madipur, Kirti Nagar, Subhash Nagar, etc. |
| Delhi South | South Delhi / Airport belt | IGI Airport, Mahipalpur, Vasant Kunj, Aya Nagar, Ghitorni, Ghitorni Village, Mehrauli, Qutub Minar, Sultanpur, Chhatarpur, Maidan Garhi, IGNOU Road, Saidulajab, Mandi House area, etc. |
| Gurgaon | All Gurgaon / Gurugram | Cyber Hub, Golf Course Road, MG Road, Sikanderpur, IFFCO Chowk, Huda City Centre, DLF Phase 3, DLF City Court, Gurgaon Sector 29/56, Ambience Mall, Gurgaon Railway Station, Old Gurgaon, Sadar Bazar Gurgaon, Udyog Vihar, Hero Honda Chowk, Palam Vihar, etc. |
| Gurgaon Peripheral | Gurgaon outskirts / NH-48 | Khandsa, Narsinghpur, IMT Manesar, Manesar, Kherki Daula Toll, Pataudi Chowk, Gwal Pahari, Badshahpur, Vatika Chowk, Sohna Road, Subhash Chowk, Ardee City, Sushant Lok, etc. |
| Noida | Noida / Greater Noida | Noida Sector 18, Noida Sector 62, Noida Extension, Greater Noida, Noida Film City, Botanical Garden, Kaushambi, etc. |
| Faridabad | Faridabad area | Faridabad Sector 15, Badarpur, Tughlakabad area |
| Outer NCR | Beyond NCR core | Sonipat, Panipat, Meerut, Bhiwadi, Bahadurgarh |

> **Methodology note (documented in Notebook 07):** Zone assignment is based on
> well-established administrative boundaries and common knowledge of NCR geography.
> Any ambiguous location is assigned to the zone of nearest administrative unit.
> The full location→zone mapping dictionary is defined once in `src/features.py`
> and referenced consistently across all notebooks and the dashboard.
> Locations not matching any zone are flagged as "Unknown" rather than force-assigned.

### Stage 3 — Feature Engineering (`src/features.py`)

Derived columns added **only to the cleaned/output dataframe**, never to the raw CSV:

| Feature | Formula | Notes |
|---|---|---|
| `hour` | `datetime.dt.hour` | |
| `day_of_week` | `datetime.dt.dayofweek` | 0=Monday |
| `day_name` | `datetime.dt.day_name()` | |
| `month` | `datetime.dt.month` | |
| `month_name` | `datetime.dt.month_name()` | |
| `quarter` | `datetime.dt.quarter` | |
| `is_weekend` | `day_of_week >= 5` | |
| `time_slot` | hour-bucketed: Early Morning (0–5) / Morning (6–10) / Afternoon (11–14) / Evening (15–19) / Night (20–22) / Late Night (23) | |
| `fare_per_km` | `Booking Value / Ride Distance` | Null where either is null |
| `is_completed` | `Booking Status == "Completed"` | Boolean |
| `is_cancelled` | Status ∈ {Cancelled by Customer, Cancelled by Driver} | Boolean |
| `cancellation_party` | "Customer" / "Driver" / "System" (No Driver Found) / None | |
| `first_booking_date` | `min(datetime)` per `Customer ID` | |
| `cohort_month` | `first_booking_date` floored to month-start | |
| `booking_month` | `datetime` floored to month-start | |
| `months_since_first` | `(booking_month - cohort_month)` in calendar months | |
| `pickup_zone` | location→zone mapping (see Stage 2) | "Unknown" if unmapped |
| `drop_zone` | same mapping for Drop Location | |
| `route` | `Pickup Location + " → " + Drop Location` | |
| `customer_total_rides` | count per `Customer ID` (over full dataset — for EDA only) | |
| `customer_completed_rides` | count of Completed per `Customer ID` (EDA only) | |
| `customer_ocv` | sum of `Booking Value` per `Customer ID` (EDA only) | Labelled "Observed Customer Value (OCV)" — not LTV |
| `customer_avg_fare` | mean `Booking Value` per `Customer ID` (EDA only) | |

> **ML features** that require per-customer history are computed separately in
> `src/ml_pipeline.py` using a time-sorted rolling approach to prevent leakage.
> The EDA-level customer aggregates above use the full dataset and must **never**
> be used as ML features.

### Stage 4 — Feature Store

Save enriched dataframe → `outputs/feature_store.parquet` for use by all notebooks.

---

## 3. Notebook Specifications

### Notebook 01 — Data Quality Audit

**Tier:** Pre-analytics  
**Inputs:** `Data/ncr_ride_bookings.csv` (raw)  
**Outputs:** Written quality report (Markdown cells), `outputs/reports/data_quality_summary.csv`

**Contents:**
- Actual row count, column list, dtypes as read
- Identify triple-quote encoding on `Booking ID` / `Customer ID` (show examples)
- Null count and null % per column — as a table and a heatmap
- String "null" frequency per column (distinct from `NaN`)
- Duplicate `Booking ID` check (count and show examples if any)
- Date range and continuity (all 12 months present?)
- Booking Status alignment check:
  - `Booking Value` null ↔ non-Completed (report exceptions)
  - `Avg VTAT` null ↔ "No Driver Found" only (report exceptions)
  - `Cancelled Rides by Customer` = 1 ↔ status = "Cancelled by Customer" (report exceptions)
- **Observed numeric distributions** (do not assume ranges):
  - Histograms and box plots of `Avg VTAT`, `Avg CTAT`, `Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`
  - Report actual min/max/percentiles (as recorded in §0 of this plan)
  - Flag values outside observed bounds only if they appear in a later run
- Summary table: "Issue | Affected Rows | Planned Resolution"

---

### Notebook 02 — Data Cleaning

**Tier:** Pre-analytics  
**Inputs:** `Data/ncr_ride_bookings.csv` (raw, read-only)  
**Outputs:** `outputs/cleaned_data.parquet`

**Contents:**
- Strip triple-quotes from IDs
- Replace "null" strings → `NaN`
- Parse `datetime`
- Cast all dtypes
- Document explicitly: nulls in `Booking Value`, `Ride Distance`, `Payment Method` are
  **structurally expected** for non-Completed rides — they are not imputed
- Document: `Driver Ratings` / `Customer Rating` nulls are not imputed (missing = ride did not produce a rating)
- Flag and isolate any `Booking Status == "Completed"` rows where `Booking Value` is still null after cleaning (data integrity violations — count, display, quarantine to a separate output file)
- **No outlier removal** at this stage; outlier investigation happens in Notebook 03
- Print final shape, null summary per column, and cast dtypes
- Write `outputs/cleaned_data.parquet`; confirm write succeeded

---

### Notebook 03 — EDA / Descriptive Analytics (Tier 1)

**Tier:** 1 — Descriptive  
**Inputs:** `outputs/cleaned_data.parquet`  
**Outputs:** `outputs/feature_store.parquet`, charts to `outputs/plots/03_*`

**3.1 Ride Volume Overview**
- Total bookings, completed rides, each cancellation type count and rate
- Monthly booking volume trend (bar + line)
- Day-of-week volume distribution

**3.2 Booking Status Breakdown**
- Stacked bar by month (status composition over time)
- Pie / donut overall

**3.3 Vehicle Type Analysis**
- Rides, completion rate, avg `Booking Value`, avg `Ride Distance`, avg `Avg VTAT` per vehicle type

**3.4 Revenue Overview**
- Total **Gross Booking Value** (completed rides only; explicitly not called LTV or net revenue)
- Monthly GBV trend
- GBV by vehicle type and by payment method
- Avg fare per ride, avg fare per km (with actual observed distribution — violin/box)

**3.5 Service Quality Overview**
- `Driver Ratings` and `Customer Rating` distributions (histograms)
- Avg rating by vehicle type
- Avg `Avg VTAT` and `Avg CTAT` by vehicle type and by month
- Outlier inspection: are any VTAT or CTAT values anomalous given the measured percentile table?

**3.6 Location Overview**
- Top 25 pickup and drop locations by volume
- Top 25 routes
- Zone-level ride volume (based on defensible mapping from §2)

**3.7 Payment Method Overview**
- Payment mix on completed rides
- Payment method by vehicle type

---

### Notebook 04 — Diagnostic Analytics (Tier 2)

**Tier:** 2 — Diagnostic  
**Inputs:** `outputs/feature_store.parquet`

**4.1 Cancellation Deep-Dive**
- Customer cancellation reasons ranked
- Driver cancellation reasons ranked
- Cancellation rate by vehicle type
- Cancellation rate by `time_slot` and day of week
- Cancellation rate by `pickup_zone`
- VTAT quartile vs customer cancellation rate (is higher VTAT associated with higher customer cancellation rate?)

**4.2 "No Driver Found" Deep-Dive**
- Rate by vehicle type, by `time_slot`, by `pickup_zone`
- Monthly trend (supply gap improving or worsening over 2024?)

**4.3 Incomplete Rides**
- Incomplete reason distribution
- Incomplete rate by vehicle type and time slot

**4.4 VTAT / CTAT Drivers**
- VTAT distribution by vehicle type, by `time_slot`, by `pickup_zone`
- CTAT distribution by vehicle type, by `time_slot`

**4.5 Rating Drivers**
- Low-rated rides (Driver Rating < 4.0) by vehicle type and route
- Correlation: `Booking Value` vs `Driver Rating`
- Correlation: `Ride Distance` vs `Customer Rating`
- Does higher VTAT correlate with lower customer rating? (scatter + grouped means)

**4.6 Revenue Diagnostics**
- Fare-per-km by vehicle type
- Monthly avg fare trend
- Revenue concentration: top 10% routes by GBV

---

### Notebook 05 — Customer Segmentation (Tier 2 / Tier 3)

**Tier:** 2 / 3  
**Inputs:** `outputs/feature_store.parquet`

> **Honest caveat documented at top of notebook:** 148,788 unique Customer IDs across
> 150,000 bookings. Most customers appear only once. RFM and clustering will reflect
> this reality. Segment sizes will be small for multi-ride segments. This is treated as
> a meaningful finding (high one-time usage, low repeat), not a data deficiency to hide.

**5.1 Repeat Customer Analysis**
- Proportion of customers: 1 ride / 2 rides / 3+ rides
- Rides-per-customer distribution (histogram, log scale)
- Completed-ride rate: repeat vs one-time customers
- **Observed Customer Value (OCV)** comparison: repeat vs one-time
  *(OCV = sum of Booking Value on completed rides within observation window; not LTV)*

**5.2 RFM Scoring**
- **Recency:** days since last booking (reference: 2024-12-30)
- **Frequency:** total bookings per customer
- **Monetary:** sum of `Booking Value` on completed rides (0 for customers with no completed ride)
- Score each 1–4 using quantiles
- Segment labels applied after scoring: Champions, Loyal, Potential Loyalist, New Customer, At Risk, Hibernating, Lost
- Segment size table and visualisation

**5.3 K-Means Clustering**
- Feature set for customers with ≥ 1 completed ride:
  `[total_rides, completed_rides, completed_rate, total_booking_value, avg_booking_value, preferred_vehicle_type (frequency encoded), preferred_time_slot (encoded), avg_driver_rating_given]`
- Scale with `StandardScaler`
- Elbow plot + silhouette score to select k (do not pre-assume k)
- PCA 2D scatter of clusters
- Cluster profile table (mean features per cluster)
- Cluster labels assigned empirically after profiling

**5.4 Segment Characterisation**
- Preferred vehicle type, payment method, time slot per segment
- Cancellation rate per segment
- Avg rating given per segment

---

### Notebook 06 — Cohort & Retention Analysis (Tier 2)

**Tier:** 2  
**Inputs:** `outputs/feature_store.parquet`

> **Documented caveat:** Given ~148,788 unique customers over 150,000 rides, true
> month-over-month retention will be very low. This is real signal — high single-use
> / low-return behaviour — and is reported as such.

**6.1 Cohort Construction**
- Cohort = month of customer's first booking (Jan–Dec 2024 → 12 cohorts)
- Matrix: rows = cohort month, columns = months since first ride
- Cell = unique customers active in that relative month

**6.2 Retention Rate Matrix**
- Divide by cohort size → retention %
- Standard cohort heatmap
- Avg retention curve across cohorts at month 0, 1, 2, …

**6.3 Revenue Cohort Matrix**
- Same structure, cell = total `Booking Value` from that cohort in that relative month
- Revenue per retained customer over time (GBV, not net revenue)

**6.4 Cohort Findings**
- Best-retention cohort; trend across cohorts; month-0 → month-1 drop

---

### Notebook 07 — Location Analysis (Tier 1 / Tier 2)

**Tier:** 1 / 2  
**Inputs:** `outputs/feature_store.parquet`

**7.1 Volume**
- Top 25 pickup / drop locations
- Top 25 routes
- Zone-level aggregation with methodology note

**7.2 Performance by Location**
- Completion rate by pickup location
- Avg `Booking Value` by pickup location
- Avg `Avg VTAT` by pickup location (supply responsiveness per area)
- "No Driver Found" rate by zone

**7.3 Directional Flow**
- Zone-to-zone ride volume cross-tab
- Intra-zone vs inter-zone share

**7.4 Revenue by Location**
- GBV per pickup location (top 20)
- Fare per km by zone

---

### Notebook 08 — Vehicle & Fare Analysis (Tier 1 / Tier 2)

**Tier:** 1 / 2  
**Inputs:** `outputs/feature_store.parquet`

**8.1 Supply & Demand**
- Demand (bookings) vs supply (completed rides) by vehicle type
- "No Driver Found" rate by vehicle type
- VTAT by vehicle type

**8.2 Fare Analysis**
- Fare distribution per vehicle (violin / box)
- Fare per km per vehicle type
- Fare distribution by `time_slot`

**8.3 Service Quality per Vehicle**
- Completion rate, cancellation rate (by party), avg ratings

**8.4 Revenue by Vehicle**
- GBV per vehicle type, GBV per completed ride by vehicle type
- Prominently documented: *"Net revenue, driver payout, commission, and margin
  cannot be calculated from this dataset. Only Gross Booking Value is reported."*

---

### Notebook 09 — Time-Based Demand Analysis (Tier 1 / Tier 2)

**Tier:** 1 / 2  
**Inputs:** `outputs/feature_store.parquet`

- Hourly demand: bookings, completion rate, cancellation rate, avg VTAT by hour
- Day-of-week: volume, GBV, completion rate
- Monthly seasonality: volume, GBV, avg fare, completion rate
- Time-slot comparison: volume, GBV, cancellation rate, avg rating per slot
- Weekend vs weekday: volume, GBV, avg fare, cancellation rate

---

### Notebook 10 — Cancellation Analysis (Tier 1 / Tier 2)

**Tier:** 1 / 2  
**Inputs:** `outputs/feature_store.parquet`

- Overall cancellation landscape: total, rate, split by party
- Customer cancellations: reason distribution, by vehicle type, by `time_slot`, by month; VTAT at time of cancellation
- Driver cancellations: reason distribution, by vehicle type, by `time_slot`
- **Revenue impact estimate**: avg `Booking Value` × cancelled ride count as a rough upper-bound potential recovery — clearly labelled "scenario estimate, assumes cancelled rides would have completed at average fare, which is not guaranteed"
- Monthly cancellation rate trend

---

### Notebook 11 — Predictive ML (Tier 3)

**Tier:** 3 — Predictive  
**Inputs:** `outputs/feature_store.parquet`

#### 11.1 Primary Task: Completion Prediction (Binary Classification)

**Target:** `is_completed` (1 = Completed, 0 = all other outcomes)

#### 11.2 Leakage Prevention

The following columns are **post-outcome** and **completely excluded** from features:

| Column | Reason |
|---|---|
| `Booking Value` | Only exists after trip completes |
| `Ride Distance` | Only known after trip |
| `Driver Ratings` | Only after completion |
| `Customer Rating` | Only after completion |
| `Payment Method` | Only after completion |
| `Cancelled Rides by Customer` | Is an outcome |
| `Reason for cancelling by Customer` | Post-outcome |
| `Cancelled Rides by Driver` | Is an outcome |
| `Driver Cancellation Reason` | Post-outcome |
| `Incomplete Rides` | Is an outcome |
| `Incomplete Rides Reason` | Post-outcome |
| `Booking Status` | The target itself |
| `Booking ID` | Non-informative identifier |
| `fare_per_km` | Derived from post-outcome columns |
| `customer_total_rides` (EDA version) | Computed on full dataset — leaks future |
| `customer_completed_rides` (EDA version) | Same |
| `customer_ocv` (EDA version) | Same |
| `customer_avg_fare` (EDA version) | Same |

**Customer history features must use time-safe rolling computation** in `src/ml_pipeline.py`:
- Sort all bookings by `datetime`
- For each booking, compute customer history using only **strictly prior** rows
- This gives: `customer_prior_rides`, `customer_prior_completion_rate`, `customer_prior_cancellation_rate`
- A customer's first booking will have prior history of 0 / NaN → imputed to 0

#### 11.3 Permitted Features

| Feature | Notes |
|---|---|
| `Vehicle Type` | One-hot encoded |
| `Pickup Location` | Target encoded (mean `is_completed` per location, **computed on training set only**) |
| `Drop Location` | Same |
| `hour` | Numeric |
| `day_of_week` | Numeric |
| `month` | Numeric |
| `is_weekend` | Boolean |
| `time_slot` | Ordinal or one-hot |
| `Avg VTAT` | Available at dispatch time; null for No Driver Found — imputed with training median inside pipeline |
| `Avg CTAT` | Available at dispatch time; null for cancellations — imputed with training median inside pipeline |
| `customer_prior_rides` | Time-safe rolling count |
| `customer_prior_completion_rate` | Time-safe rolling rate |
| `customer_prior_cancellation_rate` | Time-safe rolling rate |

All preprocessing (imputation, encoding, scaling) is wrapped in a `sklearn.Pipeline` to ensure no leakage across the train/test boundary.

#### 11.4 Train / Test Split

- **Time-based split only** (no random splits): train on 2024-01-01 – 2024-09-30, test on 2024-10-01 – 2024-12-30
- Approximate split: ~75% train / ~25% test
- Target-encoding for locations computed on **training set only**, applied to test set

#### 11.5 Models — Staged Approach

**Stage 1 (mandatory):**
1. **Logistic Regression** — interpretable baseline; `class_weight='balanced'`
2. **Random Forest Classifier** — nonlinear comparison; `class_weight='balanced'`

**Stage 2 (conditional):** Only proceed to gradient boosting (XGBoost, LightGBM) if:
- LR and RF both underperform meaningfully (ROC-AUC gap suggests room for improvement), **and**
- The added complexity is justified for the business use case  
- Decision and reasoning documented in the notebook

Optional heavy libraries (`xgboost`, `lightgbm`, `shap`) are therefore **not in the base `requirements.txt`**.
They will be added to an `requirements-optional.txt` only if Stage 2 is triggered.

#### 11.6 Model Evaluation

Metrics reported on the **held-out test set** for each model:

| Metric | Purpose |
|---|---|
| ROC-AUC | Discrimination ability (class-imbalance robust) |
| Precision | Of predicted completions, how many actually completed |
| Recall | Of actual completions, how many were predicted |
| F1-score | Harmonic mean of precision/recall |
| Confusion Matrix | Visualise error types |
| Precision-Recall curve | Better diagnostic under mild imbalance |

> **No hardcoded pass/fail threshold.** Performance is reported as-is and
> interpreted in business context: "Does this ROC-AUC provide actionable
> discrimination for dispatch prioritisation?" An AUC of 0.62 may still
> be useful for supply allocation; an AUC of 0.52 is not. The notebook will
> state the business interpretation explicitly.

Feature importance plotted for Random Forest (built-in impurity importance).
If SHAP is installed (Stage 2 only), SHAP values are added; otherwise, skipped.

#### 11.7 Secondary Task: Fare Estimation (Regression — completed rides only)

**Target:** `Booking Value` (INR), completed rides subset (93,000 rows)  
**Permitted features:** `Vehicle Type`, `Pickup Location` (encoded), `Drop Location` (encoded), `Ride Distance`, `hour`, `day_of_week`, `month`  
> `Ride Distance` is permitted here: the task is framed as fare *estimation* at the route-planning stage, where route distance is known from map APIs. This framing is documented.

**Models (staged same approach):**  
Stage 1: Linear Regression (baseline), Random Forest Regressor  
Stage 2: XGBoost Regressor (only if RF shows meaningful room for improvement)

**Metrics:** MAE, RMSE, R², residual plots  
**Split:** same time-based split as classification

---

### Notebook 12 — Prescriptive Analytics (Tier 4)

**Tier:** 4 — Prescriptive  
**Inputs:** Outputs from notebooks 03–11

**12.1 Supply Optimisation**
- Identify top (location, time_slot) cells with highest "No Driver Found" rate → driver incentive zone recommendations
- Scenario: "If No Driver Found rate reduced by X% in identified cells → estimated additional rides × median fare" — clearly labelled *scenario estimate*

**12.2 Cancellation Reduction**
- High-VTAT vehicle types → VTAT threshold alert prescription
- "Driver not moving" cancellations → movement monitoring prescription
- Scenario: reduce customer cancellation rate by X% → estimated additional completions × median fare

**12.3 Customer Retention**
- Month-1 retention drop is the critical churn point → re-engagement notification at day 7 post first-ride prescription
- RFM segment-based actions (Champions → loyalty tier; At Risk → discount; Lost → win-back)
- Documented: *"Revenue impact of retention actions cannot be quantified without cost data (CAC, LTV). Prescriptions are directional."*

**12.4 Vehicle Mix**
- Under-supplied vehicle types in high-demand zones/times → targeted driver incentives
- Document: *"Vehicle retirement recommendations require driver payout data, which is unavailable."*

**12.5 Peak Demand Management**
- Demand-supply gap index (bookings / completions ratio by hour × day) → surge pricing window recommendations
- Driver scheduling recommendations for low-completion slots

**12.6 Payment Mix**
- UPI dominance → cashback prescription to increase digital wallet adoption
- Cash-heavy routes → digital migration incentive prescription

---

## 4. Streamlit Dashboard (`dashboard/app.py`)

Run: `streamlit run dashboard/app.py`

### Page 1 — Executive Overview
KPI cards: Total Bookings, Completion Rate, Gross Booking Value (INR), Avg Fare, Avg Driver Rating  
Monthly GBV trend (line), Booking status donut, MoM volume Δ

### Page 2 — Demand & Time Analysis
Heatmap: hour × day_of_week volume; monthly trend; time-slot bar; weekend/weekday toggle

### Page 3 — Vehicle & Fare Analysis
Vehicle comparison table (completion rate, avg fare, avg distance, avg VTAT);  
fare distribution box plots; fare per km table

### Page 4 — Location Intelligence
Top pickup/drop locations; zone-to-zone flow table;  
location performance table (volume, completion rate, avg fare, avg VTAT);  
zone mapping methodology note displayed in sidebar

### Page 5 — Cancellation Analytics
Cancellation funnel; by vehicle type; by time slot; monthly rate trend

### Page 6 — Customer Analytics
Repeat customer rate KPI; rides-per-customer histogram;  
RFM segment sizes; cluster profiles; **OCV label visible, LTV explicitly disclaimed**

### Page 7 — Cohort & Retention
Interactive cohort heatmap; retention curve; GBV cohort matrix

### Page 8 — ML Model Results
Model comparison table (LR vs RF, all metrics); feature importance bar;  
confusion matrix; ROC-AUC curve; business interpretation text  
Optional: single-ride completion probability estimator (input: vehicle, pickup, drop, hour, day)

### Page 9 — Prescriptive Insights
Scenario sliders → estimated revenue impact; segment action table;  
supply gap top-10 table; unsupported metrics disclaimer panel

---

## 5. Jupyter Notebook Standards

- Each notebook opens with a **Markdown header block**: Tier, Purpose, Inputs, Outputs, Caveats/Limitations
- All charts include: title, axis labels with units, data source note ("Source: ncr_ride_bookings.csv")
- Every analytical claim is directly backed by the aggregated number in the cell output above it
- **Limitations section at end of each notebook**: what cannot be measured and why
- Notebooks run top-to-bottom in order (01 → 12) without errors or manual intervention
- Charts saved to `outputs/plots/{notebook_number}_{descriptor}.png`

---

## 6. Word Report Structure (`report/Uber_India_Analytics_Report.docx`)

1. **Title Page** — Project title, author, date, dataset description, version
2. **Executive Summary** — 5–7 key findings, actionable prescriptions summary
3. **Dataset Overview** — Columns, row count, date range, data quality findings, null patterns
4. **Methodology** — Analytics ladder (Tier 1–4), pipeline description, leakage prevention strategy, model selection rationale
5. **Descriptive Analytics** — Volumes, GBV, vehicle mix, payment mix, ratings
6. **Diagnostic Analytics** — Cancellation root causes, rating drivers, VTAT analysis
7. **Customer Analytics** — Repeat usage, OCV (not LTV), RFM segments, clusters, cohort retention
8. **Location & Time Analytics** — Zone analysis, peak patterns, supply gaps
9. **Predictive Analytics** — Model results with business interpretation, fare estimation
10. **Prescriptive Analytics** — Actionable recommendations with scenario estimates
11. **Limitations & Unavailable Metrics** — CAC, LTV, ROAS, driver payout, operating cost, contribution margin: documented as unavailable, with note on what data would be needed to compute them
12. **Appendix** — Confusion matrices, full cohort matrix, cluster profiles, zone mapping table

---

## 7. README Structure

1. Project title and one-line description
2. Dataset overview (source, dimensions, date range)
3. Repository structure (file tree)
4. Setup: `pip install -r requirements.txt`; optional: `pip install -r requirements-optional.txt`
5. Running notebooks (order matters: 01 → 12)
6. Running the dashboard: `streamlit run dashboard/app.py`
7. Analytics tier summary
8. Key findings (populated after completion)
9. Limitations and unavailable metrics
10. Data access note: raw CSV is read-only; all outputs go to `outputs/`

---

## 8. Requirements

### `requirements.txt` (mandatory)

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
scikit-learn>=1.3
streamlit>=1.28
python-docx>=1.0
pyarrow>=13.0
openpyxl>=3.1
jupyter>=1.0
ipykernel>=6.0
statsmodels>=0.14
scipy>=1.11
```

### `requirements-optional.txt` (only if Stage 2 ML is triggered)

```
xgboost>=2.0
lightgbm>=4.0
shap>=0.43
```

---

## 9. Testing Strategy (`tests/`)

### `test_data_loader.py`
- Print actual row count; do not assert == 150,000 (report deviation, not fail)
- Assert no "null" strings remain after cleaning
- Assert `Booking ID` contains no triple-quotes post-clean
- Assert `datetime` column parsed as `datetime64`
- Warn (do not fail) if any `Booking Value` is non-null for a non-Completed row
- Assert `Data/ncr_ride_bookings.csv` is not modified after pipeline runs (compare file hash before/after)

### `test_features.py`
- Assert `fare_per_km` is null wherever `Booking Value` or `Ride Distance` is null
- Assert `is_completed` = True iff `Booking Status == "Completed"`
- Assert `customer_prior_rides` is 0 for each customer's first booking (leakage check)
- Assert `customer_prior_completion_rate` is NaN for each customer's first booking (leakage check)
- Assert `cohort_month` equals the minimum `booking_month` per customer

### `test_ml_pipeline.py`
- Assert none of the excluded post-outcome columns appear in the feature matrix
- Assert train set contains only dates ≤ 2024-09-30; test set only ≥ 2024-10-01
- Assert target-encoding maps are fitted on training rows only (check that no test-set location appears in the fit metadata with test-set target rates)
- Assert customer prior-history features are computed using time-sorted rolling (no future rows used)
- **Do not assert any ROC-AUC threshold.** Report the achieved value and document interpretation.

---

## 10. Implementation Order

| Step | Task | Output artifact |
|---|---|---|
| 1 | Write `requirements.txt` and `requirements-optional.txt` | Both files |
| 2 | Write `src/data_loader.py` | Module |
| 3 | Write `src/features.py` (incl. zone mapping dict) | Module |
| 4 | Run Notebook 01 (Quality Audit) | Quality report |
| 5 | Run Notebook 02 (Cleaning) | `outputs/cleaned_data.parquet` |
| 6 | Run Notebook 03 (EDA) | `outputs/feature_store.parquet`, charts |
| 7 | Run Notebook 04 (Diagnostic) | Charts + findings |
| 8 | Run Notebook 05 (Segmentation) | Segment assignments |
| 9 | Run Notebook 06 (Cohort) | Cohort matrices |
| 10 | Run Notebooks 07–10 (Location, Vehicle, Time, Cancellation) | Charts |
| 11 | Write `src/ml_pipeline.py` | Module |
| 12 | Run Notebook 11 (ML) | Trained models, evaluation report |
| 13 | Run Notebook 12 (Prescriptive) | Scenario tables |
| 14 | Build `dashboard/app.py` | Streamlit app |
| 15 | Write `tests/` | Passing test suite |
| 16 | Write `README.md` | README |
| 17 | Write `report/Uber_India_Analytics_Report.docx` | Final Word report |
| 18 | Write `AGENTS.md` | AGENTS.md |

---

## 11. Hard Constraints & Guardrails

1. **`Data/ncr_ride_bookings.csv` is read-only.** No write, overwrite, or in-place modification. All transforms produce new files under `outputs/`.
2. **No synthetic data, fake columns, or imputed values for structurally-null fields.** `Booking Value` is null for non-completed rides by design; it is not imputed.
3. **Leakage prevention is explicit and documented** in Notebook 11 with a named exclusion table (reproduced above in §3.11).
4. **All customer-level historical ML features** are computed in time-sorted rolling order using strictly prior bookings only.
5. **Time-based train/test split is mandatory.** No random splits for any ML task.
6. **Revenue is always labelled "Gross Booking Value (GBV)"**; per-customer cumulative revenue is labelled **"Observed Customer Value (OCV)"**. Neither is called LTV.
7. **Unavailable metrics** (CAC, LTV, ROAS, driver payout, commission, operating cost, contribution margin) are documented as unavailable in every notebook that touches revenue, in the dashboard, and in the report. No proxy is fabricated.
8. **No hardcoded numeric assertions** on distributions that were assumed rather than observed. All range bounds used in validation are derived from the measured percentile table in §0.
9. **No hardcoded ML pass/fail threshold.** Achieved ROC-AUC is reported and interpreted in business terms.
10. **Zone mapping is documented** with its methodology and a full location→zone table; unmapped locations are labelled "Unknown", not force-assigned.
11. **Optional ML libraries** (XGBoost, LightGBM, SHAP) are not installed by default; they are only added if Stage 2 ML is triggered after evaluating LR and RF performance.

---

## 12. Corrections Applied (Review Log)

This section records the 11-point correction set applied before implementation:

| # | Correction | Resolution in plan |
|---|---|---|
| 1 | Raw CSV is read-only | §0, §2 Stage 1, §11 constraint #1 — explicit at every touch point |
| 2 | No "LTV" proxy | §0 unsupported metrics, §3.5, §4, §6, §11 constraint #6 — all instances changed to OCV with definition |
| 3 | Do not assume VTAT 2–20 / CTAT 10–45 ranges | §0 table shows these are **measured, not assumed**; §3.1 quality audit uses observed values only; §11 constraint #8 |
| 4 | XGBoost / LightGBM / SHAP not auto-installed | §3.11 staged approach; `requirements-optional.txt`; §11 constraint #11 |
| 5 | LR baseline first, RF second | §3.11 Stage 1 explicitly ordered |
| 6 | Time-based split mandatory | §3.11, §11 constraint #5 |
| 7 | No arbitrary ROC-AUC pass/fail | §3.11 evaluation, `test_ml_pipeline.py`, §11 constraint #9 |
| 8 | No hardcoded row count or distribution assertions | §2 Stage 1 (warn, don't fail); §9 tests; §11 constraint #8 |
| 9 | No manual zone mapping without inspecting location names | §2 Stage 2 — full 176-location list inspected; defensible mapping with methodology note; "Unknown" for unmapped |
| 10 | All unsupported metrics documented as unavailable | §0 table, every relevant notebook, §6 report §11, dashboard Page 9, §11 constraint #7 |
| 11 | Update plan before implementation | This document is the corrected plan; awaiting approval |
