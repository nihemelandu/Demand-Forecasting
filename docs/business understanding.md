# Business Understanding Document
## M5 Demand Forecasting Project
### Point and Probabilistic Forecasts — Walmart Unit Sales

**Version:** 1.0

**Date:** 2025-02-17

**Author:** Ngozi

**Repository:** https://github.com/nihemelandu/demand-forecasting

---

## 1. Business Context

Walmart operates 10 stores across three US states — California (CA), Texas (TX), and Wisconsin (WI) — selling 3,049 products across 3 product categories (Foods, Hobbies, Household) and 7 product departments.

At this scale, inventory decisions are made across 42,840 distinct time series organized in a 12-level hierarchy. A forecasting error at any level of this hierarchy has a direct financial consequence:

- **Over-forecasting** → excess inventory → holding costs, markdowns, waste
- **Under-forecasting** → stockouts → lost sales, customer dissatisfaction, emergency replenishment costs

The business needs forecasts that are not just accurate on average but **calibrated** — meaning the uncertainty around each forecast is as reliable as the forecast itself. A buyer managing safety stock needs to trust that the 95th percentile forecast actually contains true demand 95% of the time.

---

## 2. Problem Statement

### Track 1 — Point Forecasting
**Question:** What will unit sales be for each of the 30,490 store-item combinations over the next 28 days?

**Business use:** Inventory replenishment orders, promotional planning, financial revenue projections.

### Track 2 — Probabilistic Forecasting
**Question:** What is the full distribution of likely unit sales for each of the 42,840 series over the next 28 days?

**Business use:** Safety stock optimisation, stockout risk quantification, cost-benefit analysis under demand uncertainty.

These are not independent problems. The probabilistic forecasts must be consistent with the point forecasts — a buyer cannot operate with a median forecast from one system and a prediction interval from another.

---

## 3. Stakeholder Map

The 12-level hierarchy directly maps to organisational stakeholders, each consuming forecasts at their relevant level of aggregation:

| Level | Aggregation | Series | Primary Stakeholder | Decision Driven |
|-------|-------------|--------|---------------------|-----------------|
| 1 | Total | 1 | CEO / Executives | Annual revenue planning |
| 2 | State | 3 | Regional VPs | Regional inventory allocation |
| 3 | Store | 10 | Store Managers | Store-level replenishment |
| 4 | Category | 3 | Category Buyers | Category purchasing strategy |
| 5 | Department | 7 | Department Managers | Department-level planning |
| 6 | State × Category | 9 | Regional Category Managers | Regional assortment decisions |
| 7 | State × Dept | 21 | Regional Dept Managers | Regional dept planning |
| 8 | Store × Category | 30 | Store Category Leads | In-store category management |
| 9 | Store × Dept | 70 | Store Dept Leads | In-store dept management |
| 10 | Item | 3,049 | Product Managers | SKU-level planning |
| 11 | State × Item | 9,147 | Regional SKU Managers | Regional SKU allocation |
| 12 | Store × Item | 30,490 | Operations | Daily replenishment execution |

**Key insight:** Forecasting errors at level 12 propagate upward through the hierarchy. A model that performs well at aggregate levels but poorly at level 12 is operationally useless — the people making daily replenishment decisions are working at level 12.

---

## 4. Data Assets

Three data files provided by Walmart via the M5 competition:

| File | Description | Format | Key Fields |
|------|-------------|--------|------------|
| sales_train_validation.csv | Daily unit sales per item-store | Wide form — 1 row per series, 1 column per day | item_id, store_id, d_1 to d_1941 |
| calendar.csv | Date metadata and events | Long form — 1 row per day | date, weekday, month, snap_CA/TX/WI, event_name, event_type |
| sell_prices.csv | Weekly item-store prices | Long form — 1 row per item-store-week | item_id, store_id, wm_yr_wk, sell_price |

**Historical range:** 2011-01-29 to 2016-06-19 — 1,941 days of training history.

**Forecast target:** The 28 days immediately following the training period.

**Critical data characteristic:** The dataset contains both smooth high-volume series (aggregate levels) and highly intermittent series (level 12 store-item combinations). These require fundamentally different modelling approaches and must be treated differently throughout the pipeline.

---

## 5. Success Criteria

Success criteria are defined before any modelling begins. Results are evaluated against these criteria on the holdout set — the final 28 days of training data, untouched throughout development.

### Primary Criteria (Competition Metrics)

| Track | Metric | Definition | Target |
|-------|--------|------------|--------|
| Track 1 | WRMSSE | Weighted Root Mean Squared Scaled Error | Beat all 6 competition baselines |
| Track 2 | WSPL | Weighted Scaled Pinball Loss | Beat all 6 competition baselines |

### Secondary Criteria (Statistical Soundness)

- Point forecast bias < 5% at every hierarchy level
- 95% prediction interval empirical coverage between 93% and 97%
- Zero quantile crossings in probabilistic forecasts
- Hierarchical coherence — bottom-up aggregation of level 12 point forecasts must equal level 1 forecast within rounding error

### Operational Criteria (Production Readiness)

- Full pipeline runs end-to-end in under 4 hours on 8 CPU cores
- All code passes unit tests before any commit
- Results are exactly reproducible from committed code and downloaded data files
- Every modelling decision is documented in CHANGELOG.md and DEVELOPMENT_LOG.md

---

## 6. Constraints and Assumptions

### Constraints

- **Data:** Only the three provided M5 files are used. No external data sources.
- **Compute:** 8 CPU cores, no GPU. Pipeline must be feasible within this constraint.
- **Reproducibility:** Random seeds are fixed for all stochastic operations. Results must reproduce exactly across runs.
- **Submission format:** Track 1 requires forecasts for 30,490 level-12 series only. Track 2 requires forecasts for all 42,840 series across all 12 levels.

### Assumptions

The following assumptions are explicitly stated here and will be verified empirically during Iteration 2:

1. **Structural zeros:** Zeros before the first non-zero observation represent periods when the item was not yet active — not genuine zero demand. These are excluded from model training and scale computations.
2. **Price as availability proxy:** If no price record exists in sell_prices.csv for a given item-store-week, the item was not stocked that week. This distinguishes missing data from true zero sales.
3. **Hierarchical consistency:** The sum of level-12 series equals the level-1 series for every day. Any inconsistency indicates a data quality issue.
4. **Stationarity of seasonality:** Weekly seasonal patterns are assumed stable across the full 1,941-day history. This will be tested during EDA.

---

## 7. Modelling Strategy

### Approach

Two-model strategy per track:

**Classical statistical models (baseline):** Naive, Seasonal Naive, SES, ETS, ARIMA, Kernel — implemented in `src/baseline_models.py`. These establish the performance floor.

**Global ML model (primary):** LightGBM trained on all series simultaneously. This is the approach validated by M5 competition winners.

The global model must measurably beat the classical baselines on the holdout set to justify its complexity. If it does not, the classical baselines are the production recommendation.

### Validation Framework

Walk-forward validation with expanding window — time-based splits only:

### Validation Framework

Walk-forward validation with expanding window — time-based splits only:

| Split   | Train        | Validate      | Notes                  |
|---------|-------------|---------------|------------------------|
| Fold 1  | d1 – d1857  | d1858 – d1885 |                        |
| Fold 2  | d1 – d1885  | d1886 – d1913 |                        |
| Holdout | d1 – d1913  | d1914 – d1941 | Touched exactly once   |

---

## 8. Project Workflow

### Methodology: Light Agile + CRISP-DM

| Iteration | CRISP-DM Phase | Core Question | Status |
|-----------|---------------|---------------|--------|
| 1 | Business Understanding | What are we building and why? | ✅ Complete |
| 2 | Data Understanding | What data do we have and is it trustworthy? | Next |
| 3 | Data Preparation | Can we run an end-to-end pipeline? | Pending |
| 4 | Modelling | Does LightGBM beat the baselines? | Pending |
| 5 | Evaluation | Do results meet our success criteria? | Pending |
| 6 | Deployment | Are results reproducible and submission-ready? | Pending |

### Non-Negotiable Standards

- Raw data files are never modified — all transformations produce new files
- Every assumption is stated explicitly and verified with code
- Every result is accompanied by its validation fold variance
- Failures and negative results are documented
- No result is reported without the code that produced it being committed first

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Intermittent series dominate WSPL score | High | High | Separate evaluation by intermittency class. Use Kernel and Croston as specific baselines |
| LightGBM overfits on validation folds | Medium | High | Walk-forward validation with 3 folds. Regularise with min_data_in_leaf |
| Pipeline too slow for 42,840 series | Medium | Medium | Parallelise with n_workers from config.yaml. Profile before optimising |
| Hierarchical incoherence in submissions | Low | High | Verify coherence check in submission generation script before any file is written |
| Data quality issues undiscovered until modelling | Medium | High | Dedicate full Iteration 2 to data validation before any feature engineering begins |

---

## 10. Deliverables by Iteration

| Iteration | Committed Artifacts |
|-----------|-------------------|
| 1 | docs/BUSINESS_UNDERSTANDING.md, config/config.yaml, README.md |
| 2 | src/data_validator.py, notebooks/01_eda.ipynb, docs/data_quality_report.md |
| 3 | src/data_loader.py, src/feature_engineer.py, src/pipeline.py, notebooks/02_pipeline_validation.ipynb |
| 4 | src/lgbm_point_model.py, src/lgbm_quantile_model.py, notebooks/03_model_development.ipynb |
| 5 | src/model_evaluator.py, notebooks/04_model_evaluation.ipynb |
| 6 | scripts/generate_submission.py, results/track1_submission.csv, results/track2_submission.csv |

---

**This document is the authoritative reference for all decisions made in this project. Any deviation from the success criteria, constraints, or assumptions stated here must be documented in CHANGELOG.md with an explicit justification.**

---

*Next action: Begin Iteration 2 — Data Understanding. First task: `src/data_validator.py`.*
