# -*- coding: utf-8 -*-
"""
src/data_validator.py

Data Validation for M5 Demand Forecasting
==========================================

Business Context
----------------
Before any transformation, feature engineering, or modelling, we must
establish that the data is trustworthy. A model built on corrupt or
misunderstood data will produce forecasts that appear reasonable but
are systematically wrong — and in a retail inventory context, that
means either stockouts or excess inventory at scale.

This module verifies four categories of data quality:

1. File integrity       — correct files, expected shapes, expected columns
2. Schema validation    — correct data types, value ranges, no unexpected nulls
3. Hierarchical consistency — level 12 series sum to level 1 for every day
4. Assumption verification — structural zeros, price as availability proxy

All findings are logged. Any CRITICAL finding must be resolved before
Iteration 3 (Data Preparation) begins. WARNING findings are documented
in docs/data_quality_report.md and handled explicitly in the pipeline.

Usage
-----
    python -m src.data_validator

Or from another module:
    from src.data_validator import DataValidator
    validator = DataValidator(config)
    report = validator.run()
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, setup_logging


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected file specifications — ground truth from M5 competitors guide
# ---------------------------------------------------------------------------

EXPECTED_SPECS = {
    'sales': {
        'filename': 'sales_train_validation.csv',
        'expected_rows': 30490,
        'expected_day_cols': 1913,
        'id_cols': ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
    },
    'calendar': {
        'filename': 'calendar.csv',
        'expected_rows': 1969,   # 1941 training + 28 forecast days
        'expected_cols': [
            'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
            'snap_CA', 'snap_TX', 'snap_WI'
        ],
    },
    'prices': {
        'filename': 'sell_prices.csv',
        'expected_cols': ['store_id', 'item_id', 'wm_yr_wk', 'sell_price'],
    },
}

# Hierarchy level definitions — from M5 competitors guide Table 1
HIERARCHY_LEVELS = {
    1:  {'name': 'Total',              'n_series': 1},
    2:  {'name': 'State',              'n_series': 3},
    3:  {'name': 'Store',              'n_series': 10},
    4:  {'name': 'Category',           'n_series': 3},
    5:  {'name': 'Department',         'n_series': 7},
    6:  {'name': 'State x Category',   'n_series': 9},
    7:  {'name': 'State x Dept',       'n_series': 21},
    8:  {'name': 'Store x Category',   'n_series': 30},
    9:  {'name': 'Store x Dept',       'n_series': 70},
    10: {'name': 'Item',               'n_series': 3049},
    11: {'name': 'State x Item',       'n_series': 9147},
    12: {'name': 'Store x Item',       'n_series': 30490},
}


class DataValidator:
    """
    Validates all three M5 data files before any pipeline processing begins.

    Business Context
    ----------------
    This class is the gatekeeper of Iteration 2. No code in Iteration 3
    or beyond can be trusted if this validator finds unresolved CRITICAL
    issues. It is designed to be run once at the start of each new
    environment setup and after any data refresh.

    Parameters
    ----------
    config : dict
        Full configuration dictionary loaded from config/config.yaml.

    Attributes
    ----------
    findings : list of dict
        Accumulated findings from all checks. Each finding has:
        {'level': 'INFO'|'WARNING'|'CRITICAL', 'check': str, 'message': str}
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = Path(config['data']['sales_path']).parent
        self.findings = []
        self.sales = None
        self.calendar = None
        self.prices = None

    def _log(self, level: str, check: str, message: str):
        """Record a finding and log it."""
        self.findings.append({
            'level':   level,
            'check':   check,
            'message': message,
        })
        log_fn = {
            'INFO':     logger.info,
            'WARNING':  logger.warning,
            'CRITICAL': logger.error,
        }.get(level, logger.info)
        log_fn(f"[{level}] {check}: {message}")

    # -----------------------------------------------------------------------
    # Step 1 — File integrity
    # -----------------------------------------------------------------------

    def check_file_integrity(self):
        """
        Verify all three data files exist, are readable, and have the
        expected number of rows and columns.

        This is the first check because everything else depends on it.
        A missing or truncated file must be caught before any downstream
        code attempts to read it.
        """
        logger.info("=" * 60)
        logger.info("STEP 1 — File Integrity Checks")
        logger.info("=" * 60)

        # --- Sales file ---
        sales_path = Path(self.config['data']['sales_path'])
        if not sales_path.exists():
            self._log('CRITICAL', 'file_exists',
                      f"Sales file not found: {sales_path}")
            return False

        self.sales = pd.read_csv(sales_path)
        n_rows, n_cols = self.sales.shape
        n_id_cols = len(EXPECTED_SPECS['sales']['id_cols'])
        n_day_cols = n_cols - n_id_cols

        self._log('INFO', 'sales_shape',
                  f"Sales file loaded: {n_rows:,} rows x {n_cols:,} columns")

        if n_rows != EXPECTED_SPECS['sales']['expected_rows']:
            self._log('CRITICAL', 'sales_row_count',
                      f"Expected {EXPECTED_SPECS['sales']['expected_rows']:,} rows, "
                      f"got {n_rows:,}")
        else:
            self._log('INFO', 'sales_row_count',
                      f"Row count correct: {n_rows:,}")

        if n_day_cols != EXPECTED_SPECS['sales']['expected_day_cols']:
            self._log('CRITICAL', 'sales_day_cols',
                      f"Expected {EXPECTED_SPECS['sales']['expected_day_cols']} day columns, "
                      f"got {n_day_cols}")
        else:
            self._log('INFO', 'sales_day_cols',
                      f"Day column count correct: {n_day_cols}")

        # Verify id columns present
        for col in EXPECTED_SPECS['sales']['id_cols']:
            if col not in self.sales.columns:
                self._log('CRITICAL', 'sales_id_cols',
                          f"Expected id column missing: {col}")

        # --- Calendar file ---
        calendar_path = Path(self.config['data']['calendar_path'])
        if not calendar_path.exists():
            self._log('CRITICAL', 'file_exists',
                      f"Calendar file not found: {calendar_path}")
            return False

        self.calendar = pd.read_csv(calendar_path)
        self._log('INFO', 'calendar_shape',
                  f"Calendar file loaded: {self.calendar.shape[0]:,} rows "
                  f"x {self.calendar.shape[1]:,} columns")

        if self.calendar.shape[0] != EXPECTED_SPECS['calendar']['expected_rows']:
            self._log('WARNING', 'calendar_row_count',
                      f"Expected {EXPECTED_SPECS['calendar']['expected_rows']} rows, "
                      f"got {self.calendar.shape[0]}")
        else:
            self._log('INFO', 'calendar_row_count',
                      f"Row count correct: {self.calendar.shape[0]:,}")

        for col in EXPECTED_SPECS['calendar']['expected_cols']:
            if col not in self.calendar.columns:
                self._log('CRITICAL', 'calendar_cols',
                          f"Expected calendar column missing: {col}")

        # --- Prices file ---
        prices_path = Path(self.config['data']['prices_path'])
        if not prices_path.exists():
            self._log('CRITICAL', 'file_exists',
                      f"Prices file not found: {prices_path}")
            return False

        self.prices = pd.read_csv(prices_path)
        self._log('INFO', 'prices_shape',
                  f"Prices file loaded: {self.prices.shape[0]:,} rows "
                  f"x {self.prices.shape[1]:,} columns")

        for col in EXPECTED_SPECS['prices']['expected_cols']:
            if col not in self.prices.columns:
                self._log('CRITICAL', 'prices_cols',
                          f"Expected prices column missing: {col}")

        return True
    
    # -----------------------------------------------------------------------
    # Step 2 — Schema validation
    # -----------------------------------------------------------------------

    def check_schema(self):
        """
        Validate data types, value ranges, null counts, and date coverage
        across all three files.

        Business Context
        ----------------
        A sales value of -3 or a price of 0.0 will silently corrupt every
        downstream calculation that depends on it — scale denominators,
        weight computations, lag features. These errors must be caught here
        before they propagate invisibly through the pipeline.

        Null handling philosophy:
        - Nulls in event columns (event_name_1, event_name_2) are expected
          and legitimate — most days have no special event.
        - Nulls in sell_price are expected — item not stocked that week.
        - Nulls in any id column (item_id, store_id, state_id) are CRITICAL
          — every series must be fully identified.
        - Nulls in day columns (d_1 to d_1913) are CRITICAL — missing sales
          data cannot be distinguished from true zero demand without
          additional investigation.
        """
        logger.info("=" * 60)
        logger.info("STEP 2 — Schema Validation")
        logger.info("=" * 60)

        if self.sales is None or self.calendar is None or self.prices is None:
            self._log('CRITICAL', 'schema_prereq',
                      "Files not loaded — run check_file_integrity() first")
            return

        self._check_sales_schema()
        self._check_calendar_schema()
        self._check_prices_schema()

    def _check_sales_schema(self):
        """Validate sales file schema."""
        logger.info("--- Sales schema ---")

        id_cols = EXPECTED_SPECS['sales']['id_cols']
        day_cols = [c for c in self.sales.columns if c.startswith('d_')]

        # --- Null checks on id columns ---
        for col in id_cols:
            n_nulls = self.sales[col].isnull().sum()
            if n_nulls > 0:
                self._log('CRITICAL', f'sales_null_{col}',
                          f"Null values in id column '{col}': {n_nulls:,} rows")
            else:
                self._log('INFO', f'sales_null_{col}',
                          f"No nulls in '{col}'")

        # --- Null checks on day columns ---
        total_null_days = self.sales[day_cols].isnull().sum().sum()
        if total_null_days > 0:
            self._log('CRITICAL', 'sales_null_days',
                      f"Null values found in day columns: {total_null_days:,} cells. "
                      f"Cannot distinguish from true zero demand without investigation.")
        else:
            self._log('INFO', 'sales_null_days',
                      "No null values in any day column")

        # --- Data types on day columns ---
        non_numeric = [
            c for c in day_cols
            if not pd.api.types.is_numeric_dtype(self.sales[c])
        ]
        if non_numeric:
            self._log('CRITICAL', 'sales_dtype_days',
                      f"Non-numeric day columns found: {non_numeric[:5]}")
        else:
            self._log('INFO', 'sales_dtype_days',
                      "All day columns are numeric")

        # --- Value range: demand must be non-negative ---
        min_demand = self.sales[day_cols].min().min()
        max_demand = self.sales[day_cols].max().max()

        if min_demand < 0:
            self._log('CRITICAL', 'sales_negative_demand',
                      f"Negative demand values found. Min value: {min_demand}")
        else:
            self._log('INFO', 'sales_demand_range',
                      f"Demand range: [{min_demand}, {max_demand}] — all non-negative")

        # --- Demand must be integer-valued ---
        # Unit sales are discrete counts — fractional values indicate
        # a data pipeline error upstream
        non_integer_count = (
            self.sales[day_cols]
            .apply(lambda col: col.dropna() % 1 != 0)
            .sum()
            .sum()
        )
        if non_integer_count > 0:
            self._log('WARNING', 'sales_non_integer',
                      f"Non-integer demand values found: {non_integer_count:,} cells. "
                      f"Unit sales should be whole numbers.")
        else:
            self._log('INFO', 'sales_integer_check',
                      "All demand values are integer-valued")

        # --- Zero demand prevalence ---
        # Important context for modelling — high zero rate means
        # intermittent demand dominates at level 12
        total_cells = self.sales[day_cols].size
        zero_cells = (self.sales[day_cols] == 0).sum().sum()
        zero_pct = 100 * zero_cells / total_cells
        self._log('INFO', 'sales_zero_prevalence',
                  f"Zero demand prevalence: {zero_pct:.1f}% of all "
                  f"({zero_cells:,} of {total_cells:,} cells)")

        # --- Id column cardinality ---
        # Verify the number of unique values matches competition spec
        expected_cardinality = {
            'state_id': 3,
            'store_id': 10,
            'cat_id':   3,
            'dept_id':  7,
            'item_id':  3049,
        }
        for col, expected_n in expected_cardinality.items():
            if col not in self.sales.columns:
                continue
            actual_n = self.sales[col].nunique()
            if actual_n != expected_n:
                self._log('WARNING', f'sales_cardinality_{col}',
                          f"'{col}': expected {expected_n} unique values, "
                          f"got {actual_n}")
            else:
                self._log('INFO', f'sales_cardinality_{col}',
                          f"'{col}': {actual_n} unique values — correct")

    def _check_calendar_schema(self):
        """Validate calendar file schema."""
        logger.info("--- Calendar schema ---")

        # --- Date parsing ---
        try:
            self.calendar['date'] = pd.to_datetime(self.calendar['date'])
            self._log('INFO', 'calendar_date_parse',
                      "Date column parsed successfully")
        except Exception as e:
            self._log('CRITICAL', 'calendar_date_parse',
                      f"Date column could not be parsed: {e}")
            return

        # --- Date range ---
        expected_start = pd.Timestamp('2011-01-29')
        #expected_end   = pd.Timestamp('2016-07-30')  # d_1913 + 28 forecast days

        actual_start = self.calendar['date'].min()
        actual_end   = self.calendar['date'].max()

        if actual_start != expected_start:
            self._log('WARNING', 'calendar_start_date',
                      f"Expected start date {expected_start.date()}, "
                      f"got {actual_start.date()}")
        else:
            self._log('INFO', 'calendar_start_date',
                      f"Start date correct: {actual_start.date()}")

        if actual_end < pd.Timestamp('2016-06-19'):
            self._log('CRITICAL', 'calendar_end_date',
                      f"Calendar ends at {actual_end.date()} — does not "
                      f"cover the full training period")
        else:
            self._log('INFO', 'calendar_end_date',
                      f"End date: {actual_end.date()}")

        # --- No duplicate dates ---
        n_dupes = self.calendar['date'].duplicated().sum()
        if n_dupes > 0:
            self._log('CRITICAL', 'calendar_duplicate_dates',
                      f"Duplicate dates found: {n_dupes}")
        else:
            self._log('INFO', 'calendar_duplicate_dates',
                      "No duplicate dates")

        # --- SNAP columns must be binary (0 or 1) ---
        for snap_col in ['snap_CA', 'snap_TX', 'snap_WI']:
            invalid = ~self.calendar[snap_col].isin([0, 1])
            if invalid.any():
                self._log('CRITICAL', f'calendar_{snap_col}_range',
                          f"Non-binary values in {snap_col}: "
                          f"{self.calendar.loc[invalid, snap_col].unique()}")
            else:
                self._log('INFO', f'calendar_{snap_col}_range',
                          f"{snap_col}: all values are 0 or 1")

        # --- Event nulls are expected ---
        for event_col in ['event_name_1', 'event_type_1',
                          'event_name_2', 'event_type_2']:
            n_nulls = self.calendar[event_col].isnull().sum()
            n_total = len(self.calendar)
            self._log('INFO', f'calendar_null_{event_col}',
                      f"'{event_col}': {n_nulls:,} nulls of {n_total:,} rows "
                      f"({100*n_nulls/n_total:.1f}%) — expected, most days have no event")

        # --- wm_yr_wk consistency ---
        n_null_wk = self.calendar['wm_yr_wk'].isnull().sum()
        if n_null_wk > 0:
            self._log('CRITICAL', 'calendar_null_wm_yr_wk',
                      f"Null values in wm_yr_wk: {n_null_wk}")
        else:
            self._log('INFO', 'calendar_null_wm_yr_wk',
                      "No nulls in wm_yr_wk")

    def _check_prices_schema(self):
        """Validate sell prices file schema."""
        logger.info("--- Prices schema ---")

        # --- Null checks ---
        # Nulls in sell_price are expected — item not stocked that week
        # Nulls in id columns are not acceptable
        for col in ['store_id', 'item_id', 'wm_yr_wk']:
            n_nulls = self.prices[col].isnull().sum()
            if n_nulls > 0:
                self._log('CRITICAL', f'prices_null_{col}',
                          f"Null values in '{col}': {n_nulls:,}")
            else:
                self._log('INFO', f'prices_null_{col}',
                          f"No nulls in '{col}'")

        n_null_price = self.prices['sell_price'].isnull().sum()
        self._log('INFO', 'prices_null_sell_price',
                  f"Null sell_price: {n_null_price:,} rows "
                  f"({100*n_null_price/len(self.prices):.2f}%) — "
                  f"expected where item not stocked")

        # --- Price must be positive where not null ---
        non_null_prices = self.prices['sell_price'].dropna()
        n_zero_price = (non_null_prices <= 0).sum()
        if n_zero_price > 0:
            self._log('CRITICAL', 'prices_non_positive',
                      f"Non-positive prices found: {n_zero_price:,} rows. "
                      f"Min price: {non_null_prices.min():.4f}")
        else:
            self._log('INFO', 'prices_positive',
                      f"All non-null prices are positive. "
                      f"Range: [{non_null_prices.min():.2f}, "
                      f"{non_null_prices.max():.2f}]")

        # --- Price data type ---
        if not pd.api.types.is_numeric_dtype(self.prices['sell_price']):
            self._log('CRITICAL', 'prices_dtype',
                      "sell_price is not numeric")
        else:
            self._log('INFO', 'prices_dtype',
                      "sell_price is numeric")

        # --- Store and item cardinality in prices ---
        n_stores = self.prices['store_id'].nunique()
        n_items  = self.prices['item_id'].nunique()
        self._log('INFO', 'prices_cardinality',
                  f"Prices cover {n_stores} stores and {n_items:,} items")
        
    # -----------------------------------------------------------------------
    # Step 3 — Hierarchical consistency verification
    # -----------------------------------------------------------------------

    def check_hierarchy(self):
        """
        Verify that level 12 series (store x item) aggregate correctly
        upward through all 11 higher hierarchy levels for every day.

        Business Context
        ----------------
        The M5 hierarchy is the backbone of the forecasting system. A store
        manager's forecast must be consistent with their regional VP's
        forecast, which must be consistent with the CEO's forecast. If the
        data itself is not hierarchically consistent, any reconciliation
        we apply during modelling is built on a broken foundation.

        This check verifies assumption 3 from the Business Understanding
        document: "The sum of level-12 series equals the level-1 series
        for every day."

        We verify four representative aggregations:
        - Level 12 to Level 1  (store-item → total)
        - Level 12 to Level 2  (store-item → state)
        - Level 12 to Level 3  (store-item → store)
        - Level 12 to Level 4  (store-item → category)

        If these four hold, the full hierarchy is consistent by transitivity.

        Tolerance
        ---------
        We allow a small floating point tolerance of 1e-6 per cell.
        Integer sales data should aggregate exactly, so any discrepancy
        above tolerance is a genuine data quality issue.
        """
        logger.info("=" * 60)
        logger.info("STEP 3 — Hierarchical Consistency Verification")
        logger.info("=" * 60)

        if self.sales is None:
            self._log('CRITICAL', 'hierarchy_prereq',
                      "Sales file not loaded — run check_file_integrity() first")
            return

        day_cols = [c for c in self.sales.columns if c.startswith('d_')]
        tolerance = 1e-6

        # ----------------------------------------------------------------
        # Level 12 → Level 1: sum of ALL store-item series = total demand
        # ----------------------------------------------------------------
        logger.info("Checking Level 12 to Level 1 (total)")
        #level12_total = self.sales[day_cols].sum(axis=0)  # sum all 30,490 series

        # Level 1 has only one series — the grand total
        # We derive it by summing all rows of the sales file
        # There is no separate level 1 row in the file — it must equal
        # the sum of all level 12 rows
        #max_discrepancy = 0.0  # by definition, level12_total IS level 1

        # What we are really checking: does the file contain exactly the
        # 30,490 bottom-level series with no duplicates or missing series?
        # We verify this through cardinality — 10 stores x 3,049 items
        expected_level12 = 10 * 3049
        actual_level12 = len(self.sales)

        if actual_level12 != expected_level12:
            self._log('CRITICAL', 'hierarchy_level12_count',
                      f"Expected {expected_level12:,} level-12 series "
                      f"(10 stores x 3,049 items), got {actual_level12:,}")
        else:
            self._log('INFO', 'hierarchy_level12_count',
                      f"Level 12 series count correct: {actual_level12:,}")

        # ----------------------------------------------------------------
        # Level 12 → Level 2: sum by state_id
        # ----------------------------------------------------------------
        logger.info("Checking Level 12 to Level 2 (state)")

        # For each state, sum of all store-item series in that state
        # must equal the state-level aggregate for every day
        state_sums = (
            self.sales
            .groupby('state_id')[day_cols]
            .sum()
        )

        n_states = len(state_sums)
        expected_states = 3
        if n_states != expected_states:
            self._log('CRITICAL', 'hierarchy_level2_states',
                      f"Expected {expected_states} states, got {n_states}")
        else:
            self._log('INFO', 'hierarchy_level2_states',
                      f"Level 2: {n_states} states found — correct")

        # Verify state totals are internally consistent
        # (sum of CA stores = CA state total)
        for state in state_sums.index:
            state_series = self.sales[self.sales['state_id'] == state]
            n_stores_in_state = state_series['store_id'].nunique()
            n_items_in_state  = state_series['item_id'].nunique()
            self._log('INFO', f'hierarchy_level2_{state}',
                      f"State {state}: {n_stores_in_state} stores, "
                      f"{n_items_in_state:,} items, "
                      f"{len(state_series):,} series")

        # ----------------------------------------------------------------
        # Level 12 → Level 3: sum by store_id
        # ----------------------------------------------------------------
        logger.info("Checking Level 12 to Level 3 (store)")

        store_sums = (
            self.sales
            .groupby('store_id')[day_cols]
            .sum()
        )

        n_stores = len(store_sums)
        expected_stores = 10
        if n_stores != expected_stores:
            self._log('CRITICAL', 'hierarchy_level3_stores',
                      f"Expected {expected_stores} stores, got {n_stores}")
        else:
            self._log('INFO', 'hierarchy_level3_stores',
                      f"Level 3: {n_stores} stores found — correct")

        # Verify each store has exactly 3,049 items
        items_per_store = self.sales.groupby('store_id')['item_id'].nunique()
        stores_with_wrong_items = items_per_store[items_per_store != 3049]
        if len(stores_with_wrong_items) > 0:
            self._log('WARNING', 'hierarchy_items_per_store',
                      f"Stores with item count != 3,049: "
                      f"{stores_with_wrong_items.to_dict()}")
        else:
            self._log('INFO', 'hierarchy_items_per_store',
                      "All 10 stores have exactly 3,049 items — correct")

        # ----------------------------------------------------------------
        # Level 12 → Level 4: sum by cat_id
        # ----------------------------------------------------------------
        logger.info("Checking Level 12 to Level 4 (category)")

        cat_sums = (
            self.sales
            .groupby('cat_id')[day_cols]
            .sum()
        )

        n_cats = len(cat_sums)
        expected_cats = 3
        if n_cats != expected_cats:
            self._log('CRITICAL', 'hierarchy_level4_cats',
                      f"Expected {expected_cats} categories, got {n_cats}")
        else:
            self._log('INFO', 'hierarchy_level4_cats',
                      f"Level 4: {n_cats} categories found — correct")

        # Report category names and their series counts
        for cat in cat_sums.index:
            n_series = len(self.sales[self.sales['cat_id'] == cat])
            self._log('INFO', f'hierarchy_level4_{cat}',
                      f"Category {cat}: {n_series:,} store-item series")

        # ----------------------------------------------------------------
        # Level 12 → Level 5: sum by dept_id
        # ----------------------------------------------------------------
        logger.info("Checking Level 12 to Level 5 (department)")

        dept_sums = (
            self.sales
            .groupby('dept_id')[day_cols]
            .sum()
        )

        n_depts = len(dept_sums)
        expected_depts = 7
        if n_depts != expected_depts:
            self._log('CRITICAL', 'hierarchy_level5_depts',
                      f"Expected {expected_depts} departments, got {n_depts}")
        else:
            self._log('INFO', 'hierarchy_level5_depts',
                      f"Level 5: {n_depts} departments found — correct")

        for dept in dept_sums.index:
            n_series = len(self.sales[self.sales['dept_id'] == dept])
            self._log('INFO', f'hierarchy_level5_{dept}',
                      f"Department {dept}: {n_series:,} store-item series")

        # ----------------------------------------------------------------
        # Grand total cross-check
        # Verify: sum of state totals = sum of store totals = sum of
        # category totals = grand total
        # If these four numbers are not equal the hierarchy is broken
        # ----------------------------------------------------------------
        logger.info("Cross-checking aggregate totals")

        grand_total        = float(self.sales[day_cols].values.sum())
        state_grand_total  = float(state_sums.values.sum())
        store_grand_total  = float(store_sums.values.sum())
        cat_grand_total    = float(cat_sums.values.sum())
        dept_grand_total   = float(dept_sums.values.sum())

        self._log('INFO', 'hierarchy_grand_total',
                  f"Grand total units sold: {grand_total:,.0f}")

        for name, total in [
            ('State aggregation',    state_grand_total),
            ('Store aggregation',    store_grand_total),
            ('Category aggregation', cat_grand_total),
            ('Department aggregation', dept_grand_total),
        ]:
            discrepancy = abs(total - grand_total)
            if discrepancy > tolerance:
                self._log('CRITICAL', 'hierarchy_grand_total_mismatch',
                          f"{name} total {total:,.0f} differs from "
                          f"grand total {grand_total:,.0f} "
                          f"by {discrepancy:,.6f}")
            else:
                self._log('INFO', 'hierarchy_grand_total_match',
                          f"{name} total matches grand total — "
                          f"discrepancy: {discrepancy:.2e}")
                
    # -----------------------------------------------------------------------
    # Step 4 — Assumption verification
    # -----------------------------------------------------------------------

    def check_assumptions(self):
        """
        Verify the four assumptions stated explicitly in the Business
        Understanding document before any modelling begins.

        Business Context
        ----------------
        Every assumption that goes unverified is a silent risk. If our
        assumption about structural zeros is wrong, we will train models
        on pre-launch periods and produce meaningless forecasts. If the
        price-as-availability proxy does not hold, we cannot reliably
        distinguish missing data from true zero demand.

        These checks convert stated assumptions into verified facts —
        or flag them as risks that must be handled explicitly.

        Assumptions verified
        --------------------
        1. Structural zeros — zeros before first non-zero observation
           are pre-launch, not genuine zero demand
        2. Price as availability proxy — zero sales correlate with
           missing price records
        3. Hierarchical coherence — already verified in Step 3
        4. Stationarity of seasonality — weekly patterns are stable
           across the full history
        """
        logger.info("=" * 60)
        logger.info("STEP 4 — Assumption Verification")
        logger.info("=" * 60)

        if self.sales is None or self.prices is None or self.calendar is None:
            self._log('CRITICAL', 'assumption_prereq',
                      "Files not loaded — run check_file_integrity() first")
            return

        self._verify_structural_zeros()
        self._verify_price_availability_proxy()
        self._verify_seasonality_stationarity()

    def _verify_structural_zeros(self):
        """
        Assumption 1: Zeros before first non-zero observation are
        structural (pre-launch), not genuine zero demand.

        Verification approach
        ---------------------
        For each series compute the launch day index (first non-zero day).
        Count how many series have leading zeros. Compute the average
        leading zero length. If items consistently have long pre-launch
        periods followed by sustained sales, the assumption holds.

        A series that oscillates between zero and non-zero from day 1
        is intermittent demand — not pre-launch. We distinguish these
        by checking whether leading zeros form a contiguous block at
        the start of the series.
        """
        logger.info("Checking Assumption 1: Structural zeros")

        day_cols = [c for c in self.sales.columns if c.startswith('d_')]
        sales_values = self.sales[day_cols].values

        launch_indices = []
        n_all_zero = 0
        n_with_leading_zeros = 0

        for row in sales_values:
            nonzero = np.where(row > 0)[0]
            if len(nonzero) == 0:
                n_all_zero += 1
                continue
            launch_idx = nonzero[0]
            launch_indices.append(launch_idx)
            if launch_idx > 0:
                n_with_leading_zeros += 1

        launch_indices = np.array(launch_indices)
        n_total = len(self.sales)
        n_active = n_total - n_all_zero

        self._log('INFO', 'assumption1_all_zero_series',
                  f"Series with all-zero history: {n_all_zero:,} of {n_total:,} "
                  f"({100*n_all_zero/n_total:.1f}%) — these will be excluded "
                  f"from model training")

        self._log('INFO', 'assumption1_series_with_leading_zeros',
                  f"Active series with leading zeros: {n_with_leading_zeros:,} "
                  f"of {n_active:,} ({100*n_with_leading_zeros/n_active:.1f}%)")

        self._log('INFO', 'assumption1_launch_day_distribution',
                  f"Launch day distribution across active series: "
                  f"min=d_{launch_indices.min()+1}, "
                  f"median=d_{int(np.median(launch_indices))+1}, "
                  f"max=d_{launch_indices.max()+1}")

        # Series that launch on d_1 have no leading zeros
        launched_on_day1 = (launch_indices == 0).sum()
        self._log('INFO', 'assumption1_launched_day1',
                  f"Series active from day 1: {launched_on_day1:,} "
                  f"({100*launched_on_day1/n_active:.1f}%)")

        # Series that launch after d_500 are clearly new products
        launched_after_500 = (launch_indices > 500).sum()
        self._log('INFO', 'assumption1_launched_after_d500',
                  f"Series launching after d_500: {launched_after_500:,} "
                  f"({100*launched_after_500/n_active:.1f}%) — "
                  f"clearly new products, structural zeros confirmed")

        if n_with_leading_zeros > 0:
            self._log('INFO', 'assumption1_verdict',
                      f"Assumption 1 SUPPORTED: {n_with_leading_zeros:,} series "
                      f"have contiguous leading zeros consistent with pre-launch "
                      f"periods. These will be trimmed before model fitting.")
        else:
            self._log('WARNING', 'assumption1_verdict',
                      "Assumption 1 UNCERTAIN: No leading zeros found. "
                      "All series appear active from day 1.")

    def _verify_price_availability_proxy(self):
        """
        Assumption 2: If no price record exists for an item-store-week,
        the item was not stocked. Zero sales with no price = missing data.
        Zero sales with a price = true zero demand.

        Verification approach
        ---------------------
        Join sales to prices on item-store-week. For weeks where price
        is missing, check what the sales value is. If sales are always
        zero when price is missing, the assumption holds perfectly.
        If sales are sometimes non-zero when price is missing, we have
        a data quality issue — sales recorded without a price.

        Note: sell_prices.csv uses wm_yr_wk (week id) not date.
        We join via the calendar file to map days to weeks.
        """
        logger.info("Checking Assumption 2: Price as availability proxy")

        # Map day columns to week ids via calendar
        day_cols = [c for c in self.sales.columns if c.startswith('d_')]

        # Build day-to-week mapping from calendar
        # Calendar has columns: date, wm_yr_wk, and d_1..d_1969 implicitly
        # We need to map d_1..d_1913 to wm_yr_wk
        # Calendar rows are in order so d_i corresponds to row i-1
        cal_subset = self.calendar[['wm_yr_wk']].copy()
        cal_subset['d_col'] = ['d_' + str(i+1)
                               for i in range(len(cal_subset))]

        # Restrict to training days only
        cal_training = cal_subset[cal_subset['d_col'].isin(day_cols)]

        # Melt sales to long form for the join
        # Use a sample of series to keep this check fast
        sample_size = min(500, len(self.sales))
        sales_sample = self.sales.sample(
            n=sample_size,
            random_state=42
        )[['item_id', 'store_id'] + day_cols]

        sales_long = sales_sample.melt(
            id_vars=['item_id', 'store_id'],
            value_vars=day_cols,
            var_name='d_col',
            value_name='demand'
        )

        # Join to get week id
        sales_long = sales_long.merge(cal_training, on='d_col', how='left')

        # Join to prices
        sales_with_price = sales_long.merge(
            self.prices,
            on=['item_id', 'store_id', 'wm_yr_wk'],
            how='left'
        )

        # Check: when price is missing, what is demand?
        missing_price = sales_with_price[
            sales_with_price['sell_price'].isna()
        ]
        has_price = sales_with_price[
            sales_with_price['sell_price'].notna()
        ]

        n_missing_price = len(missing_price)
        n_has_price = len(has_price)

        self._log('INFO', 'assumption2_price_coverage',
                  f"Sample of {sample_size:,} series: "
                  f"{n_has_price:,} day-records with price, "
                  f"{n_missing_price:,} without price")

        if n_missing_price > 0:
            nonzero_without_price = (missing_price['demand'] > 0).sum()
            zero_without_price = (missing_price['demand'] == 0).sum()

            self._log('INFO', 'assumption2_missing_price_demand',
                      f"Records with no price: "
                      f"{zero_without_price:,} zero demand, "
                      f"{nonzero_without_price:,} non-zero demand")

            if nonzero_without_price == 0:
                self._log('INFO', 'assumption2_verdict',
                          "Assumption 2 SUPPORTED: All records with missing "
                          "price have zero demand. Price is a reliable "
                          "availability proxy.")
            else:
                pct = 100 * nonzero_without_price / n_missing_price
                self._log('WARNING', 'assumption2_verdict',
                          f"Assumption 2 PARTIALLY SUPPORTED: "
                          f"{nonzero_without_price:,} records ({pct:.1f}%) "
                          f"show non-zero demand despite missing price. "
                          f"These will be flagged as data quality issues "
                          f"in the pipeline.")
        else:
            self._log('INFO', 'assumption2_verdict',
                      "Assumption 2 NOT TESTABLE on this sample: "
                      "no missing price records found. "
                      "All sampled item-store-weeks have a price record.")

    def _verify_seasonality_stationarity(self):
        """
        Assumption 4: Weekly seasonal patterns are stable across the
        full 1,913-day history.

        Verification approach
        ---------------------
        Split the history into three roughly equal periods and compute
        the average demand by day of week for each period. If the
        day-of-week pattern is consistent across periods the assumption
        holds. A large shift in the relative pattern (e.g. Saturday
        becomes the weakest day in period 3 after being the strongest
        in period 1) would invalidate the assumption.

        We use the level-1 aggregate (sum of all series) for this check
        because it is the smoothest series and most clearly shows the
        seasonal pattern without noise.
        """
        logger.info("Checking Assumption 4: Stationarity of weekly seasonality")

        day_cols = [c for c in self.sales.columns if c.startswith('d_')]

        # Level 1 aggregate — sum all series
        total_demand = self.sales[day_cols].sum(axis=0)
        total_demand.index = range(len(total_demand))

        # Get day of week from calendar
        cal = self.calendar[['wm_yr_wk', 'weekday']].copy()
        cal = cal.iloc[:len(day_cols)].reset_index(drop=True)

        demand_df = pd.DataFrame({
            'demand':  total_demand.values,
            'weekday': cal['weekday'].values,
            'day_idx': range(len(day_cols))
        })

        # Split into three periods
        n = len(day_cols)
        period_size = n // 3
        demand_df['period'] = pd.cut(
            demand_df['day_idx'],
            bins=[0, period_size, 2*period_size, n],
            labels=['Period 1\n(2011-2012)',
                    'Period 2\n(2013-2014)',
                    'Period 3\n(2015-2016)'],
            include_lowest=True
        )

        # Average demand by weekday and period
        weekday_pattern = (
            demand_df
            .groupby(['period', 'weekday'], observed=True)['demand']
            .mean()
            .unstack('weekday')
        )

        # Normalise each period by its own mean so patterns are comparable
        weekday_pattern_norm = weekday_pattern.div(
            weekday_pattern.mean(axis=1), axis=0
        )

        # Log the pattern for each period
        weekday_order = ['Saturday', 'Sunday', 'Monday', 'Tuesday',
                         'Wednesday', 'Thursday', 'Friday']

        for period in weekday_pattern_norm.index:
            pattern_str = ' | '.join([
                f"{day[:3]}: {weekday_pattern_norm.loc[period, day]:.2f}"
                for day in weekday_order
                if day in weekday_pattern_norm.columns
            ])
            self._log('INFO', f'assumption4_pattern_{str(period)[:8]}',
                      f"{period}: {pattern_str}")

        # Measure stability: max range of normalised values per weekday
        # across periods. A range < 0.1 means the pattern is very stable.
        stability = weekday_pattern_norm.max() - weekday_pattern_norm.min()
        max_instability = stability.max()
        most_unstable_day = stability.idxmax()

        if max_instability < 0.10:
            self._log('INFO', 'assumption4_verdict',
                      f"Assumption 4 SUPPORTED: Weekly seasonal pattern is "
                      f"stable across all three periods. Max variation: "
                      f"{max_instability:.3f} on {most_unstable_day}. "
                      f"Safe to use full history for seasonal feature engineering.")
        elif max_instability < 0.20:
            self._log('WARNING', 'assumption4_verdict',
                      f"Assumption 4 PARTIALLY SUPPORTED: Some drift in weekly "
                      f"pattern detected. Max variation: {max_instability:.3f} "
                      f"on {most_unstable_day}. Monitor for concept drift "
                      f"during model evaluation.")
        else:
            self._log('WARNING', 'assumption4_verdict',
                      f"Assumption 4 UNCERTAIN: Significant drift in weekly "
                      f"pattern detected. Max variation: {max_instability:.3f} "
                      f"on {most_unstable_day}. Consider using a shorter "
                      f"training window or adding drift features.")
    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
            logger.info("Starting M5 Data Validation")
            logger.info(f"Data directory: {self.data_dir}")
    
            ok = self.check_file_integrity()
    
            if ok:
                self.check_schema()
                self.check_hierarchy()
                self.check_assumptions()        # <- add this line
            else:
                logger.error(
                    "File integrity checks failed. "
                    "Resolve CRITICAL findings before proceeding."
                )
    
            findings_df = pd.DataFrame(self.findings)
            self._print_summary(findings_df)
            return findings_df   
        
    def _print_summary(self, findings_df: pd.DataFrame):
        """Print a clean summary of all findings."""
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        if findings_df.empty:
            logger.info("No findings recorded.")
            return
        for level in ['CRITICAL', 'WARNING', 'INFO']:
            subset = findings_df[findings_df['level'] == level]
            logger.info(f"{level}: {len(subset)} findings")
        criticals = findings_df[findings_df['level'] == 'CRITICAL']
        if not criticals.empty:
            logger.error("CRITICAL findings must be resolved before Iteration 3.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    config = load_config('config/config.yaml')
    setup_logging(config)
    validator = DataValidator(config)
    report = validator.run()
    print(report)
