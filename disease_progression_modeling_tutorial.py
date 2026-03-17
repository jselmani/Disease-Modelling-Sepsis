#!/usr/bin/env python3
"""
================================================================================
SELF-LEARNING TUTORIAL:
Disease Progression Modeling for Sepsis Using MIMIC-III
with Multi-Series Time-Aware Subsequence Clustering (MT-TICC)
================================================================================

OVERVIEW:
    This tutorial teaches how to model disease progression for sepsis patients
    using Electronic Health Records (EHR) from the MIMIC-III database. We apply
    a clustering-based approach inspired by the MT-TICC (Multi-series Time-aware
    Toeplitz Inverse Covariance-based Clustering) method proposed by Yang et al.

    The key idea: Rather than relying on expensive hand-labeled data, we use
    unsupervised subsequence clustering to discover interpretable "disease states"
    from multivariate clinical time series, then leverage these states as features
    for early prediction of septic shock.

IMPORTANT NOTE ON DATA:
    This tutorial uses SYNTHETIC clinical data by default (USE_SYNTHETIC = True).
    The synthetic data generator creates 200 ICU patients with 6 discrete clinical
    states that mirror the paper's discovered clusters (Table II). This design
    choice is intentional: our simplified MT-TICC implementation omits three key
    algorithmic components from the full paper (graphical lasso, Viterbi dynamic
    programming, and Toeplitz constraints), which means it cannot reliably
    discover meaningful clusters on real MIMIC-III data. Synthetic data with
    planted cluster structure allows us to demonstrate all pipeline concepts
    (clustering, transition analysis, LSTM prediction) effectively. See Section 11
    for a detailed discussion of limitations and what the full algorithm requires.

LEARNING OBJECTIVES:
    1. Extract and preprocess clinical time-series data from MIMIC-III
    2. Understand the TICC framework for subsequence clustering
    3. Implement a simplified MT-TICC pipeline with multi-series and time-awareness
    4. Use discovered clusters as features for septic shock early prediction
    5. Visualize and interpret disease progression patterns

PREREQUISITES:
    - Python 3.9+
    - Access to MIMIC-III database (PostgreSQL)
    - Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy,
      torch (PyTorch), psycopg2-binary

REFERENCE PAPER:
    Yang, X., Zhang, Y., & Chi, M. (2021). "Multi-series Time-aware Sequence
    Partitioning for Disease Progression Modeling." Department of Computer Science,
    North Carolina State University.

AUTHOR: Jiel Selmani
DATE:   March 2026
================================================================================
"""

# ==============================================================================
# SECTION 1: ENVIRONMENT SETUP AND IMPORTS
# ==============================================================================
# We begin by importing all required libraries. Each library serves a specific
# purpose in our pipeline:
#   - numpy/pandas: Core data manipulation
#   - matplotlib/seaborn: Visualization
#   - sklearn: Machine learning utilities (GMM, metrics, preprocessing)
#   - torch (PyTorch): LSTM model for prediction
#   - psycopg2: PostgreSQL connector for MIMIC-III queries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score)
# Deep Learning (LSTM for septic shock prediction)
# We use PyTorch for its native Apple Silicon (MPS) support and portability.
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Database connector for MIMIC-III (PostgreSQL)
import psycopg2
import psycopg2.extras

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("Environment setup complete. All libraries loaded successfully.")
print("=" * 70)

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
# Update these settings to match YOUR local MIMIC-III PostgreSQL setup.
# If you followed the official MIMIC-III build scripts, the defaults below
# should work. The only field you MUST change is 'user'.
#
# HOW TO SET UP MIMIC-III LOCALLY:
#   1. Install PostgreSQL (brew install postgresql on Mac, apt on Linux)
#   2. Create a database:  createdb mimic
#   3. Run the official MIMIC-III build scripts from:
#      https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres
#   4. Update MIMIC_DB_CONFIG below with your PostgreSQL username
#
# NOTE: If you are using local socket authentication (default on Mac/Linux),
# you can leave 'password' as an empty string.

MIMIC_DB_CONFIG = {
    'dbname':   'mimic',                    # Database name
    'user':     'your-user-name',           # <-- CHANGE THIS to your system username
    'password': '',                          # Leave empty for local socket auth
    'host':     'localhost',                 # Use 'localhost' for local setup
    'port':     '5432',                      # Default PostgreSQL port
}


# ==============================================================================
# SECTION 2: MIMIC-III DATA EXTRACTION
# ==============================================================================
# MIMIC-III (Medical Information Mart for Intensive Care III) is a freely
# available critical care database. It contains de-identified health records
# for ~40,000 ICU patients at Beth Israel Deaconess Medical Center (2001-2012).
#
# IMPORTANT: You must be a credentialed user of MIMIC-III and have completed
# the required CITI training to access this data. Always follow the data use
# agreement terms.
#
# Our data extraction pipeline follows these steps:
#   1. Identify sepsis-related admissions using ICD-9 codes and clinical rules
#   2. Extract vital signs from the CHARTEVENTS table
#   3. Extract lab results from the LABEVENTS table
#   4. Combine into a unified clinical time-series dataset

def connect_to_mimic(config=None):
    """
    Establish connection to the local MIMIC-III PostgreSQL database.

    Parameters:
    -----------
    config : dict - Database configuration. If None, uses MIMIC_DB_CONFIG
                    defined at the top of this file.

    Returns:
    --------
    conn : psycopg2 connection object

    Raises:
    -------
    Exception if connection fails. Common fixes:
      - Ensure PostgreSQL is running:  pg_isready
      - Verify your username in MIMIC_DB_CONFIG
      - Check that the 'mimic' database exists:  psql -l
    """
    if config is None:
        config = MIMIC_DB_CONFIG

    if config['user'] == 'your_postgres_username':
        raise ValueError(
            "You must set your PostgreSQL username in MIMIC_DB_CONFIG "
            "at the top of this file. Replace 'your_postgres_username' "
            "with your actual system username (e.g., run 'whoami' in terminal)."
        )

    # Build connection kwargs, omitting password if empty (socket auth)
    connect_kwargs = {
        'dbname': config['dbname'],
        'user':   config['user'],
        'host':   config['host'],
        'port':   config['port'],
    }
    if config.get('password'):
        connect_kwargs['password'] = config['password']

    conn = psycopg2.connect(**connect_kwargs)
    print(f"Connected to MIMIC-III database: {config['dbname']} "
          f"(user: {config['user']})")
    return conn


# ---------------------------------------------------------------------------
# 2a. SQL Query: Identify Sepsis Cohort
# ---------------------------------------------------------------------------
# We identify sepsis patients using two complementary data sources:
#   (1) DRG (Diagnosis-Related Group) codes from the drgcodes table
#       - these are assigned at discharge and reliably capture the primary
#       reason for hospitalization
#   (2) Free-text admission diagnosis from the admissions table
#       - captures sepsis mentions at the time of admission
#
# NOTE: Some MIMIC-III builds may not fully populate the diagnoses_icd table.
# DRG codes + admission diagnoses provide a robust alternative that does not
# depend on ICD-9 code availability.
#
# For septic shock specifically, we combine:
#   - Explicit "septic shock" in the admission diagnosis
#   - Vasopressor administration (norepinephrine, epinephrine, dopamine,
#     vasopressin) from inputevents_mv, following the Sepsis-3 definition
#     (Singer et al., 2016): persistent hypotension requiring vasopressors
#     to maintain MAP >= 65 mmHg

SEPSIS_COHORT_QUERY = """
-- ==========================================================================
-- STEP 1: Identify sepsis admissions using TWO complementary sources
-- ==========================================================================
-- NOTE: We use DRG codes and the admissions diagnosis field rather than
-- diagnoses_icd (ICD-9 codes). Depending on your MIMIC-III build, the
-- diagnoses_icd table may not be fully populated. DRG codes and admission
-- diagnoses provide robust alternative identification of sepsis patients.
--
-- DRG (Diagnosis-Related Group) codes are assigned at discharge and capture
-- the primary reason for hospitalization, making them reliable for cohort
-- identification.
WITH sepsis_from_drg AS (
    SELECT DISTINCT dg.hadm_id
    FROM mimiciii.drgcodes dg
    WHERE LOWER(dg.description) LIKE '%sepsis%'
       OR LOWER(dg.description) LIKE '%septicemia%'
),

sepsis_from_admission AS (
    SELECT DISTINCT a.hadm_id
    FROM mimiciii.admissions a
    WHERE LOWER(a.diagnosis) LIKE '%sepsis%'
       OR LOWER(a.diagnosis) LIKE '%septic%'
),

-- Combine both sources (UNION removes duplicates)
sepsis_admissions AS (
    SELECT hadm_id FROM sepsis_from_drg
    UNION
    SELECT hadm_id FROM sepsis_from_admission
),

-- ==========================================================================
-- STEP 2: Tag septic shock visits using clinical criteria
-- ==========================================================================
-- We identify septic shock using TWO complementary approaches:
--   (a) Explicit "septic shock" in the admission diagnosis text
--   (b) Vasopressor administration (norepinephrine, epinephrine, dopamine,
--       vasopressin) from the inputevents_mv table - this aligns with the
--       Sepsis-3 clinical definition: vasopressors needed to maintain
--       MAP >= 65 mmHg after adequate fluid resuscitation.
shock_visits AS (
    SELECT
        sa.hadm_id,
        a.subject_id,
        CASE
            WHEN LOWER(a.diagnosis) LIKE '%septic shock%'
              OR EXISTS (
                    SELECT 1 FROM mimiciii.inputevents_mv ie
                    WHERE ie.hadm_id = sa.hadm_id
                      AND ie.itemid IN (
                          221906,  -- Norepinephrine
                          221289,  -- Epinephrine
                          221662,  -- Dopamine
                          222315   -- Vasopressin
                      )
                 )
            THEN 1
            ELSE 0
        END AS septic_shock_flag
    FROM sepsis_admissions sa
    JOIN mimiciii.admissions a ON sa.hadm_id = a.hadm_id
),

-- ==========================================================================
-- STEP 3: Get patient demographics for stratified sampling
-- ==========================================================================
-- NOTE on age: MIMIC-III shifts dates of birth for patients >89 years old
-- to obscure their identity, resulting in ages of ~300. We cap age at 90.
demographics AS (
    SELECT
        p.subject_id,
        LEAST(
            EXTRACT(YEAR FROM AGE(a.admittime, p.dob))::numeric,
            90
        ) AS age,
        p.gender,
        a.ethnicity,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 3600.0 AS stay_hours
    FROM mimiciii.patients p
    JOIN mimiciii.admissions a ON p.subject_id = a.subject_id
)

-- ==========================================================================
-- STEP 4: Final cohort with demographics and shock labels
-- ==========================================================================
SELECT
    sv.subject_id,
    sv.hadm_id,
    sv.septic_shock_flag,
    d.age,
    d.gender,
    d.ethnicity,
    d.stay_hours,
    d.admittime
FROM shock_visits sv
JOIN demographics d ON sv.hadm_id = d.hadm_id
WHERE d.age >= 18            -- Adults only
  AND d.stay_hours >= 12     -- Minimum 12-hour stay for meaningful time series
  AND d.stay_hours <= 720    -- Exclude stays > 30 days (likely data errors)
ORDER BY sv.subject_id, d.admittime;
"""


# ---------------------------------------------------------------------------
# 2b. SQL Query: Extract Vital Signs
# ---------------------------------------------------------------------------
# We extract 6 vital sign measurements from CHARTEVENTS, which are recorded
# at the bedside by nurses and automated monitoring equipment.
# These correspond to the features used in the MT-TICC paper.

VITALS_QUERY = """
SELECT
    ce.subject_id,
    ce.hadm_id,
    ce.charttime,
    -- Vital signs with their MIMIC-III item IDs
    -- SBP: Systolic Blood Pressure (mmHg)
    MAX(CASE WHEN ce.itemid IN (51, 442, 455, 6701, 220179, 220050) THEN ce.valuenum END) AS sbp,
    -- MAP: Mean Arterial Pressure (mmHg)
    MAX(CASE WHEN ce.itemid IN (52, 456, 6702, 220052, 220181, 225312) THEN ce.valuenum END) AS map,
    -- Heart Rate (beats/min)
    MAX(CASE WHEN ce.itemid IN (211, 220045) THEN ce.valuenum END) AS hr,
    -- Respiratory Rate (breaths/min)
    MAX(CASE WHEN ce.itemid IN (618, 615, 220210, 224690) THEN ce.valuenum END) AS rr,
    -- Oxygen Saturation - SpO2 (%)
    MAX(CASE WHEN ce.itemid IN (646, 220277) THEN ce.valuenum END) AS spo2,
    -- Temperature (Celsius)
    MAX(CASE WHEN ce.itemid IN (223761, 678) THEN (ce.valuenum - 32) * 5.0/9.0
         WHEN ce.itemid IN (223762, 676) THEN ce.valuenum END) AS temp
FROM mimiciii.chartevents ce
WHERE ce.hadm_id IN (SELECT hadm_id FROM sepsis_cohort)
  AND ce.itemid IN (
      -- Pre-filter to only our target item IDs for efficiency.
      -- CHARTEVENTS has 330M+ rows; this reduces the scan dramatically.
      51, 442, 455, 6701, 220179, 220050,       -- SBP
      52, 456, 6702, 220052, 220181, 225312,     -- MAP
      211, 220045,                                -- HR
      618, 615, 220210, 224690,                   -- RR
      646, 220277,                                -- SpO2
      223761, 678, 223762, 676                    -- Temperature
  )
  AND ce.error IS DISTINCT FROM 1   -- Exclude erroneous entries
  AND ce.valuenum IS NOT NULL
  AND ce.valuenum > 0               -- Basic sanity check
GROUP BY ce.subject_id, ce.hadm_id, ce.charttime
ORDER BY ce.subject_id, ce.hadm_id, ce.charttime;
"""


# ---------------------------------------------------------------------------
# 2c. SQL Query: Extract Lab Results
# ---------------------------------------------------------------------------
# We extract 8 lab measurements from LABEVENTS. Lab values are typically
# recorded less frequently than vitals (every 4-12 hours vs every 1-4 hours).

LABS_QUERY = """
SELECT
    le.subject_id,
    le.hadm_id,
    le.charttime,
    -- Lab values with their MIMIC-III item IDs
    -- WBC: White Blood Cell count (K/uL)
    MAX(CASE WHEN le.itemid IN (51300, 51301) THEN le.valuenum END) AS wbc,
    -- Bilirubin (mg/dL)
    MAX(CASE WHEN le.itemid = 50885 THEN le.valuenum END) AS bilirubin,
    -- BUN: Blood Urea Nitrogen (mg/dL)
    MAX(CASE WHEN le.itemid = 51006 THEN le.valuenum END) AS bun,
    -- Lactate (mmol/L) - critical for sepsis severity assessment
    MAX(CASE WHEN le.itemid = 50813 THEN le.valuenum END) AS lactate,
    -- Creatinine (mg/dL)
    MAX(CASE WHEN le.itemid = 50912 THEN le.valuenum END) AS creatinine,
    -- Platelet count (K/uL)
    MAX(CASE WHEN le.itemid = 51265 THEN le.valuenum END) AS platelet,
    -- Neutrophils/Bands (%)
    MAX(CASE WHEN le.itemid IN (51256, 51144) THEN le.valuenum END) AS neutrophils,
    -- FiO2: Fraction of Inspired Oxygen (%)
    MAX(CASE WHEN le.itemid = 50816 THEN le.valuenum END) AS fio2
FROM mimiciii.labevents le
WHERE le.hadm_id IN (SELECT hadm_id FROM sepsis_cohort)
  AND le.valuenum IS NOT NULL
GROUP BY le.subject_id, le.hadm_id, le.charttime
ORDER BY le.subject_id, le.hadm_id, le.charttime;
"""


def extract_mimic_data(conn):
    """
    Execute SQL queries to extract the sepsis cohort, vital signs,
    and lab results from MIMIC-III.

    Strategy:
    ---------
    1. Run the cohort query to identify sepsis patients
    2. Store their hadm_ids in a temporary table ('sepsis_cohort')
       so the vitals and labs queries can efficiently join against it
       (CHARTEVENTS alone has 330M+ rows - we need to filter early)
    3. Extract vitals and labs for the cohort only
    4. Clean up the temporary table

    Parameters:
    -----------
    conn : psycopg2 connection object

    Returns:
    --------
    cohort_df : DataFrame with patient demographics and shock labels
    vitals_df : DataFrame with vital sign time series
    labs_df : DataFrame with lab result time series
    """
    cur = conn.cursor()

    # Step 1: Extract the sepsis cohort
    print("Extracting sepsis cohort...")
    cohort_df = pd.read_sql(SEPSIS_COHORT_QUERY, conn)
    n_shock = int(cohort_df['septic_shock_flag'].sum())
    n_total = len(cohort_df)
    print(f"  Found {n_total} admissions "
          f"({n_shock} shock, {n_total - n_shock} non-shock)")

    if n_total == 0:
        raise RuntimeError(
            "No sepsis admissions found. Check that your MIMIC-III database "
            "has data in the drgcodes and admissions tables."
        )

    # Step 2: Create a temporary table of cohort hadm_ids.
    # Temporary tables are session-scoped (only visible on this connection).
    # This allows VITALS_QUERY and LABS_QUERY to reference 'sepsis_cohort'
    # with efficient index-based lookups instead of repeated subqueries.
    print("  Creating temporary cohort table for efficient joins...")
    cur.execute("DROP TABLE IF EXISTS sepsis_cohort")
    cur.execute(
        "CREATE TEMP TABLE sepsis_cohort (hadm_id INTEGER PRIMARY KEY)"
    )
    hadm_ids = cohort_df['hadm_id'].unique().tolist()
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO sepsis_cohort (hadm_id) VALUES %s",
        [(h,) for h in hadm_ids]
    )
    conn.commit()
    print(f"  Temp table 'sepsis_cohort' created with {len(hadm_ids)} hadm_ids")

    # Step 3: Extract vital signs
    # This may take several minutes depending on your hardware - we are
    # scanning the CHARTEVENTS table (330M+ rows) filtered to our cohort
    # and target item IDs.
    print("Extracting vital signs (this may take several minutes)...")
    vitals_df = pd.read_sql(VITALS_QUERY, conn)
    print(f"  {len(vitals_df):,} vital sign records extracted")

    # Step 4: Extract lab results
    print("Extracting lab results...")
    labs_df = pd.read_sql(LABS_QUERY, conn)
    print(f"  {len(labs_df):,} lab result records extracted")

    # Step 5: Clean up temporary table
    cur.execute("DROP TABLE IF EXISTS sepsis_cohort")
    conn.commit()
    cur.close()

    return cohort_df, vitals_df, labs_df


# ==============================================================================
# SECTION 3: DATA PREPROCESSING
# ==============================================================================
# Raw EHR data is messy. Clinical events are recorded at irregular intervals,
# values can be physiologically implausible, and missingness rates can exceed
# 80% for some features. This section handles:
#   (1) Outlier removal using physiological bounds
#   (2) Time-series alignment and resampling
#   (3) Missing data imputation via forward-fill + mean backfill
#   (4) Feature normalization
#   (5) Cohort balancing via stratified sampling

# ---------------------------------------------------------------------------
# 3a. Physiological Bounds for Outlier Removal
# ---------------------------------------------------------------------------
# We define clinically reasonable ranges for each feature. Values outside
# these ranges are likely measurement errors or data entry mistakes.

PHYSIOLOGICAL_BOUNDS = {
    # Vital Signs
    'sbp':          (40, 300),     # Systolic BP: 40-300 mmHg
    'map':          (20, 200),     # Mean Arterial Pressure: 20-200 mmHg
    'hr':           (20, 300),     # Heart Rate: 20-300 bpm
    'rr':           (4, 70),       # Respiratory Rate: 4-70 breaths/min
    'spo2':         (50, 100),     # SpO2: 50-100%
    'temp':         (30, 43),      # Temperature: 30-43°C
    # Lab Results
    'wbc':          (0.1, 500),    # WBC: 0.1-500 K/uL
    'bilirubin':    (0, 70),       # Bilirubin: 0-70 mg/dL
    'bun':          (1, 300),      # BUN: 1-300 mg/dL
    'lactate':      (0.1, 30),     # Lactate: 0.1-30 mmol/L
    'creatinine':   (0.1, 30),     # Creatinine: 0.1-30 mg/dL
    'platelet':     (1, 1500),     # Platelet: 1-1500 K/uL
    'neutrophils':  (0, 100),      # Neutrophils: 0-100%
    'fio2':         (21, 100),     # FiO2: 21-100%
}

# The 14 features we will use, matching the MT-TICC paper
FEATURE_COLUMNS = [
    'sbp', 'map', 'rr', 'hr', 'spo2', 'temp',        # 6 vital signs
    'wbc', 'bilirubin', 'bun', 'lactate',              # 4 lab values
    'creatinine', 'platelet', 'neutrophils', 'fio2'    # 4 more lab values
]


def remove_outliers(df, bounds_dict):
    """
    Replace values outside physiological bounds with NaN.

    Rationale: A heart rate of 5000 or a temperature of -10°C is clearly
    an instrument error. Rather than dropping the entire row (which would
    lose other valid measurements), we set only the implausible value to NaN
    and handle it during imputation.

    Parameters:
    -----------
    df : DataFrame with clinical measurements
    bounds_dict : dict mapping column names to (min, max) tuples

    Returns:
    --------
    df_clean : DataFrame with outliers replaced by NaN
    """
    df_clean = df.copy()
    for col, (low, high) in bounds_dict.items():
        if col in df_clean.columns:
            mask = (df_clean[col] < low) | (df_clean[col] > high)
            n_outliers = mask.sum()
            if n_outliers > 0:
                print(f"  Removed {n_outliers} outliers from '{col}' "
                      f"(outside [{low}, {high}])")
                df_clean.loc[mask, col] = np.nan
    return df_clean


def merge_vitals_and_labs(vitals_df, labs_df):
    """
    Merge vital signs and lab results into a single time-series DataFrame.

    Strategy: Since vitals and labs are recorded at different times, we use
    an outer merge on (subject_id, hadm_id, charttime). This preserves all
    measurement timestamps. Missing values in the non-matching series will
    be filled during imputation.

    Parameters:
    -----------
    vitals_df : DataFrame with vital sign records
    labs_df : DataFrame with lab result records

    Returns:
    --------
    merged_df : DataFrame with all 14 features aligned by time
    """
    merged = pd.merge(
        vitals_df,
        labs_df,
        on=['subject_id', 'hadm_id', 'charttime'],
        how='outer',
        suffixes=('', '_lab')
    )
    merged = merged.sort_values(['subject_id', 'hadm_id', 'charttime'])
    merged = merged.reset_index(drop=True)
    print(f"  Merged dataset: {len(merged)} records, {len(FEATURE_COLUMNS)} features")
    return merged


def impute_missing_values(df, feature_cols):
    """
    Handle missing data using forward-fill then mean-backfill strategy.

    Clinical Rationale: In ICU settings, if a measurement isn't taken,
    the patient's true value is most likely close to their last recorded
    value (forward-fill). For the beginning of a stay where no prior
    values exist, we use the population mean as a conservative estimate.

    The paper reports ~80% missing rate, which is common in EHR data because
    different features are measured at different frequencies (e.g., HR every
    minute via monitors vs. lactate every 4-12 hours via blood drawSecs).

    Parameters:
    -----------
    df : DataFrame with clinical measurements
    feature_cols : list of feature column names

    Returns:
    --------
    df_imputed : DataFrame with no missing values in feature columns
    """
    df_imputed = df.copy()

    # Step 1: Forward-fill within each admission
    # (Carry the last observation forward until a new measurement appears)
    df_imputed[feature_cols] = df_imputed.groupby(
        ['subject_id', 'hadm_id']
    )[feature_cols].transform(lambda x: x.ffill())

    # Step 2: Backward-fill any remaining NaNs at the start of admissions
    # with the mean of that admission
    df_imputed[feature_cols] = df_imputed.groupby(
        ['subject_id', 'hadm_id']
    )[feature_cols].transform(lambda x: x.fillna(x.mean()))

    # Step 3: Fill any still-remaining NaNs with global mean
    # (handles admissions with 100% missing for a feature)
    global_means = df_imputed[feature_cols].mean()
    df_imputed[feature_cols] = df_imputed[feature_cols].fillna(global_means)

    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    print(f"  Remaining missing values after imputation: {missing_after}")
    return df_imputed


def normalize_features(df, feature_cols, method='standard'):
    """
    Normalize features to have comparable scales.

    Why normalize? Our 14 features have very different scales:
    - Heart rate: ~60-120 bpm
    - Platelet count: ~150,000-400,000 K/uL
    Without normalization, features with larger magnitudes would dominate
    the distance calculations in clustering.

    Parameters:
    -----------
    df : DataFrame with clinical measurements
    feature_cols : list of feature column names
    method : 'standard' (z-score) or 'minmax' (0-1 scaling)

    Returns:
    --------
    df_normalized : DataFrame with normalized features
    scaler : fitted scaler object (needed to transform test data)
    """
    df_norm = df.copy()
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    df_norm[feature_cols] = scaler.fit_transform(df_norm[feature_cols])
    print(f"  Features normalized using {method} scaling")
    return df_norm, scaler


def create_balanced_cohort(cohort_df, shock_col='septic_shock_flag',
                           random_state=42):
    """
    Create a balanced dataset via stratified random sampling.

    Problem: Septic shock is relatively rare (~5-8% of sepsis patients),
    creating a severe class imbalance. The paper addresses this by stratified
    sampling on age, sex, ethnicity, and stay duration to create a 1:1
    shock/non-shock ratio while maintaining similar demographic distributions.

    Parameters:
    -----------
    cohort_df : DataFrame with patient demographics and shock labels
    shock_col : name of the binary outcome column
    random_state : random seed for reproducibility

    Returns:
    --------
    balanced_df : DataFrame with equal shock/non-shock patients
    """
    shock = cohort_df[cohort_df[shock_col] == 1]
    non_shock = cohort_df[cohort_df[shock_col] == 0]

    n_shock = len(shock)
    print(f"  Shock visits: {n_shock}")
    print(f"  Non-shock visits: {len(non_shock)}")

    # Stratified sample of non-shock patients to match shock count
    non_shock_sampled = non_shock.sample(
        n=n_shock,
        random_state=random_state
    )

    balanced = pd.concat([shock, non_shock_sampled], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"  Balanced cohort: {len(balanced)} total "
          f"({n_shock} shock + {n_shock} non-shock)")
    return balanced


def preprocess_pipeline(vitals_df, labs_df, cohort_df):
    """
    Full preprocessing pipeline combining all steps above.

    This is the master function that orchestrates the entire data
    preprocessing workflow from raw MIMIC-III extracts to clean,
    normalized, balanced data ready for clustering.

    Parameters:
    -----------
    vitals_df : Raw vital signs DataFrame
    labs_df : Raw lab results DataFrame
    cohort_df : Patient cohort DataFrame

    Returns:
    --------
    data_dict : Dictionary containing processed data and metadata
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Remove outliers
    print("\nStep 1: Removing physiological outliers...")
    vitals_clean = remove_outliers(vitals_df, PHYSIOLOGICAL_BOUNDS)
    labs_clean = remove_outliers(labs_df, PHYSIOLOGICAL_BOUNDS)

    # Step 2: Merge vitals and labs
    print("\nStep 2: Merging vitals and labs...")
    merged = merge_vitals_and_labs(vitals_clean, labs_clean)

    # Step 3: Impute missing values
    print("\nStep 3: Imputing missing values...")
    imputed = impute_missing_values(merged, FEATURE_COLUMNS)

    # Step 4: Balance the cohort
    print("\nStep 4: Balancing the cohort...")
    balanced_cohort = create_balanced_cohort(cohort_df)

    # Filter to balanced cohort
    balanced_hadm_ids = set(balanced_cohort['hadm_id'].values)
    data = imputed[imputed['hadm_id'].isin(balanced_hadm_ids)].copy()

    # Step 5: Normalize features
    print("\nStep 5: Normalizing features...")
    data_norm, scaler = normalize_features(data, FEATURE_COLUMNS)

    # Step 6: Compute time intervals between consecutive events
    print("\nStep 6: Computing time intervals...")
    data_norm['charttime'] = pd.to_datetime(data_norm['charttime'])
    data_norm['delta_t'] = data_norm.groupby(
        ['subject_id', 'hadm_id']
    )['charttime'].diff().dt.total_seconds() / 3600.0  # Convert to hours
    data_norm['delta_t'] = data_norm['delta_t'].fillna(0)

    print(f"\n  Final dataset: {len(data_norm)} events across "
          f"{data_norm['hadm_id'].nunique()} admissions")

    return {
        'data': data_norm,
        'cohort': balanced_cohort,
        'scaler': scaler,
        'feature_columns': FEATURE_COLUMNS
    }


# ==============================================================================
# SECTION 4: UNDERSTANDING TICC
# ==============================================================================
# TICC (Toeplitz Inverse Covariance-based Clustering) is an unsupervised
# method for simultaneously segmenting and clustering multivariate time series.
#
# KEY CONCEPTS:
#
# 1. SUBSEQUENCE REPRESENTATION:
#    Instead of clustering individual time points, TICC uses a sliding window
#    of size ω (omega) to create subsequences. For a window of size 3 and
#    m=14 features, each subsequence is a (3 × 14 = 42)-dimensional vector.
#
#    Time:  t-2  t-1   t
#           [x1] [x1] [x1]
#           [x2] [x2] [x2]  → Flattened to 42-dim vector
#           [..] [..] [..]
#           [x14][x14][x14]
#
# 2. INVERSE COVARIANCE (PRECISION) MATRIX:
#    For each cluster k, TICC learns an inverse covariance matrix Θ_k that
#    captures the structural dependencies between features across time steps.
#    - Non-zero entries in Θ_k indicate conditional dependencies
#    - The INVERSE covariance (not covariance) is used because:
#      a) It directly represents conditional dependencies (partial correlations)
#      b) Zero entries mean features are conditionally independent
#      c) It naturally handles multicollinearity
#
# 3. BLOCK-WISE TOEPLITZ CONSTRAINT:
#    The key innovation: Θ_k is constrained to be block-wise Toeplitz.
#    This means the relationship between features at time lag i is the SAME
#    regardless of absolute position in the window. This captures
#    TIME-INVARIANT structural patterns.
#
#    Θ_k = | A(0)      A(1)^T    A(2)^T  |
#           | A(1)      A(0)      A(1)^T  |
#           | A(2)      A(1)      A(0)    |
#
#    Where A(i) ∈ R^(m×m) captures cross-feature dependencies at lag i.
#
# 4. OPTIMIZATION:
#    TICC minimizes: Σ_k [ Σ_n ( -log_likelihood + β·consistency ) + λ·||Θ_k||_1 ]
#    - Log-likelihood: How well each event fits its assigned cluster
#    - Consistency: Penalty for adjacent events being in different clusters
#    - Sparsity: L1 penalty on Θ_k for interpretability
#    Solved via EM: E-step (Viterbi for assignments), M-step (graphical lasso for Θ_k)


# ==============================================================================
# SECTION 5: MT-TICC IMPLEMENTATION
# ==============================================================================
# We extend the basic TICC in two critical ways:
#
# 1. MULTI-SERIES (M-TICC): EHRs consist of multiple patient visits (series).
#    Treating them independently leads to inconsistent patterns, while
#    concatenating them introduces artificial joint effects. M-TICC jointly
#    partitions and clusters across all series to learn shared patterns.
#
# 2. TIME-AWARENESS (MT-TICC): In EHRs, the interval between consecutive
#    records varies from seconds to days. Two events closer in time should
#    be more strongly encouraged to belong to the same cluster. We incorporate
#    a decay function: 1/log(e + Δt) into the consistency term.

class SimplifiedMTTICC:
    """
    Simplified implementation of Multi-series Time-aware TICC for
    educational purposes.

    This implementation captures the core ideas of MT-TICC while being
    more accessible than the full optimization-based approach. We use
    Gaussian Mixture Models (GMM) as the clustering backbone and add
    the multi-series consistency and time-awareness components.

    For a production implementation, refer to the original TICC code:
    https://github.com/davidhallac/TICC

    Parameters:
    -----------
    n_clusters : int - Number of disease state clusters (K)
    window_size : int - Sliding window size (ω)
    beta : float - Consistency penalty weight
    lambda_reg : float - Sparsity regularization coefficient
    max_iter : int - Maximum EM iterations
    tol : float - Convergence tolerance
    """

    def __init__(self, n_clusters=6, window_size=3, beta=10.0,
                 lambda_reg=1e-5, max_iter=50, tol=1e-4):
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_means_ = None
        self.cluster_covs_ = None
        self.cluster_inv_covs_ = None
        self.labels_ = None
        self.gmm_ = None

    def _create_subsequences(self, data, feature_cols):
        """
        Create sliding window subsequences from the time series.

        For each time point t, we extract the window [t-ω+1, ..., t] and
        flatten it into a single vector of dimension (ω × m).

        This is the fundamental representation that allows TICC to capture
        temporal dependencies: the flattened subsequence encodes not just
        the current feature values but also their recent trajectory.

        Parameters:
        -----------
        data : DataFrame with clinical measurements
        feature_cols : list of feature column names

        Returns:
        --------
        subsequences : np.array of shape (n_windows, ω * m)
        indices : corresponding row indices in original data
        delta_t_values : time intervals for each window position
        """
        values = data[feature_cols].values
        delta_t = data['delta_t'].values
        n_samples, n_features = values.shape
        omega = self.window_size

        subsequences = []
        indices = []
        delta_t_values = []

        for i in range(omega - 1, n_samples):
            # Extract window of ω consecutive events
            window = values[i - omega + 1:i + 1]  # Shape: (ω, m)
            subseq = window.flatten()              # Shape: (ω * m,)
            subsequences.append(subseq)
            indices.append(i)
            delta_t_values.append(delta_t[i])

        return (np.array(subsequences),
                np.array(indices),
                np.array(delta_t_values))

    def _time_aware_consistency(self, labels, delta_t_values, series_ids):
        """
        Compute the time-aware consistency penalty.

        The consistency term encourages consecutive events to be in the
        same cluster, with the penalty WEIGHTED by the time interval:

            c(X_{t-1}, P_k, Δt) = β * 1{t-1 ∉ P_k} / log(e + Δt)

        Key insight: The decay function 1/log(e + Δt) means:
        - Small Δt (events close in time) → large penalty for cluster switches
        - Large Δt (events far apart) → smaller penalty, allowing natural transitions

        This is crucial for EHRs where intervals range from seconds to days.

        Parameters:
        -----------
        labels : cluster assignments array
        delta_t_values : time intervals between consecutive events
        series_ids : series (admission) identifiers

        Returns:
        --------
        penalty : total time-aware consistency penalty
        """
        penalty = 0.0
        for i in range(1, len(labels)):
            # Only penalize within the same series (admission)
            if series_ids[i] == series_ids[i - 1]:
                if labels[i] != labels[i - 1]:
                    # Time-aware decay: shorter intervals = stronger penalty
                    dt = delta_t_values[i]
                    time_weight = 1.0 / np.log(np.e + dt)
                    penalty += self.beta * time_weight
        return penalty

    def _e_step(self, subsequences, delta_t_values, series_ids):
        """
        E-Step: Assign each subsequence to its optimal cluster.

        In the full TICC, this is solved via dynamic programming (Viterbi)
        to find the globally optimal assignment path. Here we use a
        simplified approach that combines GMM likelihood with the
        time-aware consistency penalty via local optimization.

        The assignment for each event minimizes:
            -log_likelihood(x_t | cluster_k) + consistency_penalty

        Parameters:
        -----------
        subsequences : array of flattened subsequences
        delta_t_values : time intervals
        series_ids : series identifiers

        Returns:
        --------
        labels : optimal cluster assignments
        """
        n_samples = len(subsequences)
        n_clusters = self.n_clusters

        # Compute log-likelihoods under each cluster's distribution
        log_likelihoods = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            mean_k = self.cluster_means_[k]
            cov_k = self.cluster_covs_[k]
            # Regularize covariance for numerical stability
            cov_reg = cov_k + self.lambda_reg * np.eye(cov_k.shape[0])

            try:
                # Multivariate Gaussian log-likelihood
                diff = subsequences - mean_k
                inv_cov = np.linalg.inv(cov_reg)
                log_det = np.linalg.slogdet(cov_reg)[1]
                log_likelihoods[:, k] = -0.5 * (
                    np.sum(diff @ inv_cov * diff, axis=1) +
                    log_det +
                    mean_k.shape[0] * np.log(2 * np.pi)
                )
            except np.linalg.LinAlgError:
                log_likelihoods[:, k] = -np.inf

        # Forward pass with time-aware consistency (simplified Viterbi)
        labels = np.zeros(n_samples, dtype=int)
        labels[0] = np.argmax(log_likelihoods[0])

        for i in range(1, n_samples):
            best_score = -np.inf
            best_k = 0

            for k in range(n_clusters):
                score = log_likelihoods[i, k]

                # Add time-aware consistency bonus for staying in same cluster
                if series_ids[i] == series_ids[i - 1]:
                    dt = delta_t_values[i]
                    time_weight = 1.0 / np.log(np.e + dt)
                    if labels[i - 1] == k:
                        score += self.beta * time_weight
                    else:
                        score -= self.beta * time_weight

                if score > best_score:
                    best_score = score
                    best_k = k

            labels[i] = best_k

        return labels

    def _m_step(self, subsequences, labels):
        """
        M-Step: Update cluster parameters (means and covariances).

        In the full TICC, this step solves a graphical lasso problem with
        Toeplitz constraints using ADMM (Alternating Direction Method of
        Multipliers). Here we estimate standard Gaussian parameters with
        L1-inspired sparsification of the precision matrix.

        The Toeplitz constraint ensures that temporal dependencies at the
        same lag are modeled identically regardless of window position,
        capturing TIME-INVARIANT structural patterns.

        Parameters:
        -----------
        subsequences : array of flattened subsequences
        labels : current cluster assignments
        """
        dim = subsequences.shape[1]

        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() < 2:
                # Skip clusters with too few members
                continue

            cluster_data = subsequences[mask]

            # Update mean
            self.cluster_means_[k] = cluster_data.mean(axis=0)

            # Update covariance with regularization
            cov = np.cov(cluster_data.T) + self.lambda_reg * np.eye(dim)

            # Apply soft-thresholding to enforce sparsity (simplified L1)
            # This approximates the graphical lasso penalty on Θ_k
            precision = np.linalg.inv(cov)
            threshold = self.lambda_reg
            precision_sparse = np.sign(precision) * np.maximum(
                np.abs(precision) - threshold, 0
            )
            # Keep diagonal elements unpenalized (standard practice)
            np.fill_diagonal(precision_sparse, np.diag(precision))

            # Reconstruct covariance from sparsified precision
            try:
                self.cluster_covs_[k] = np.linalg.inv(precision_sparse)
                self.cluster_inv_covs_[k] = precision_sparse
            except np.linalg.LinAlgError:
                self.cluster_covs_[k] = cov
                self.cluster_inv_covs_[k] = precision

    def fit(self, data, feature_cols, series_col='hadm_id'):
        """
        Fit the MT-TICC model to clinical time-series data.

        The algorithm alternates between:
        1. E-step: Assign events to clusters (considering time-aware consistency)
        2. M-step: Update cluster parameters (with sparsity constraints)

        Until convergence or max_iter is reached.

        Parameters:
        -----------
        data : DataFrame with clinical time-series data
        feature_cols : list of feature column names
        series_col : column identifying different series (admissions)

        Returns:
        --------
        self : fitted model
        """
        print("\n" + "=" * 60)
        print("FITTING MT-TICC MODEL")
        print("=" * 60)
        print(f"  K={self.n_clusters}, ω={self.window_size}, "
              f"β={self.beta}, λ={self.lambda_reg}")

        # Create subsequences from all series
        all_subsequences = []
        all_delta_t = []
        all_series_ids = []
        all_indices = []

        for series_id, group in data.groupby(series_col):
            subseq, idx, dt = self._create_subsequences(group, feature_cols)
            all_subsequences.append(subseq)
            all_delta_t.append(dt)
            all_series_ids.extend([series_id] * len(subseq))
            all_indices.extend(idx)

        subsequences = np.vstack(all_subsequences)
        delta_t_values = np.concatenate(all_delta_t)
        series_ids = np.array(all_series_ids)

        print(f"  Created {len(subsequences)} subsequences "
              f"(dim={subsequences.shape[1]})")

        # Initialize with GMM (provides good starting clusters)
        print("  Initializing with Gaussian Mixture Model...")
        dim = subsequences.shape[1]
        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            n_init=3,
            random_state=42
        )
        initial_labels = gmm.fit_predict(subsequences)

        # Initialize cluster parameters from GMM
        self.cluster_means_ = [gmm.means_[k] for k in range(self.n_clusters)]
        self.cluster_covs_ = [gmm.covariances_[k] for k in range(self.n_clusters)]
        self.cluster_inv_covs_ = [
            np.linalg.inv(cov + self.lambda_reg * np.eye(dim))
            for cov in self.cluster_covs_
        ]

        # EM iterations with time-aware consistency
        prev_labels = initial_labels.copy()
        for iteration in range(self.max_iter):
            # E-step: assign clusters with time-aware consistency
            labels = self._e_step(subsequences, delta_t_values, series_ids)

            # M-step: update cluster parameters
            self._m_step(subsequences, labels)

            # Check convergence
            changed = np.sum(labels != prev_labels)
            pct_changed = changed / len(labels) * 100
            print(f"  Iteration {iteration + 1}: {changed} assignments changed "
                  f"({pct_changed:.2f}%)")

            if pct_changed < self.tol * 100:
                print(f"  Converged after {iteration + 1} iterations!")
                break

            prev_labels = labels.copy()

        self.labels_ = labels
        self.subsequences_ = subsequences
        self.series_ids_ = series_ids
        self.delta_t_values_ = delta_t_values
        self.indices_ = np.array(all_indices)

        # Print cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        print("\n  Cluster distribution:")
        for k, count in zip(unique, counts):
            print(f"    Cluster {k + 1}: {count} events "
                  f"({count / len(labels) * 100:.1f}%)")

        return self

    def predict(self, data, feature_cols, series_col='hadm_id'):
        """
        Predict cluster assignments for new data.

        For test data, we compute the probability of each event belonging
        to each cluster using the learned parameters, then assign to the
        cluster with maximum probability (Eq. 2 in the paper).

        Parameters:
        -----------
        data : DataFrame with clinical time-series
        feature_cols : list of feature columns
        series_col : series identifier column

        Returns:
        --------
        labels : predicted cluster assignments
        probabilities : cluster membership probabilities
        """
        all_subsequences = []
        all_delta_t = []
        all_series_ids = []

        for series_id, group in data.groupby(series_col):
            subseq, idx, dt = self._create_subsequences(group, feature_cols)
            all_subsequences.append(subseq)
            all_delta_t.append(dt)
            all_series_ids.extend([series_id] * len(subseq))

        subsequences = np.vstack(all_subsequences)
        delta_t_values = np.concatenate(all_delta_t)
        series_ids = np.array(all_series_ids)

        labels = self._e_step(subsequences, delta_t_values, series_ids)

        # Compute cluster probabilities
        n_samples = len(subsequences)
        probs = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            diff = subsequences - self.cluster_means_[k]
            cov_reg = (self.cluster_covs_[k] +
                       self.lambda_reg * np.eye(self.cluster_covs_[k].shape[0]))
            inv_cov = np.linalg.inv(cov_reg)
            log_det = np.linalg.slogdet(cov_reg)[1]
            probs[:, k] = np.exp(-0.5 * (
                np.sum(diff @ inv_cov * diff, axis=1) + log_det
            ))

        # Normalize to get probabilities
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)

        return labels, probs

    def get_cluster_features(self, data, labels, probs, feature_cols,
                             series_col='hadm_id'):
        """
        Generate cluster-based features for downstream prediction.

        Following the paper's approach, we create K additional features
        (one per cluster) representing the probability of each event
        belonging to each cluster. These are then used as supplementary
        features for the LSTM prediction model.

        Parameters:
        -----------
        data : original DataFrame
        labels : cluster assignments
        probs : cluster membership probabilities
        feature_cols : feature column names
        series_col : series identifier

        Returns:
        --------
        augmented_data : DataFrame with original + cluster features
        """
        cluster_feature_names = [f'cluster_{k+1}_prob' for k in range(self.n_clusters)]

        # Map cluster probabilities back to original data rows.
        # The cluster probabilities correspond to SUBSEQUENCES (one per
        # sliding window), which are fewer than the original data rows.
        # We map them by series, aligning from the end of each series
        # (the first ω-1 rows of each series don't have cluster assignments).
        augmented_data = data.copy()
        for col in cluster_feature_names:
            augmented_data[col] = 0.0

        prob_idx = 0
        for series_id, group in data.groupby(series_col):
            n_rows = len(group)
            omega = self.window_size
            n_windows = max(0, n_rows - omega + 1)

            if n_windows == 0:
                continue

            # Get the original indices for this series
            original_indices = group.index.tolist()

            # The first (omega - 1) rows don't have cluster assignments;
            # assign them the probabilities of the first window.
            for i in range(omega - 1):
                if prob_idx < len(probs):
                    for k, col in enumerate(cluster_feature_names):
                        augmented_data.loc[original_indices[i], col] = probs[prob_idx, k]

            # Assign probabilities to the remaining rows
            for i in range(n_windows):
                row_idx = original_indices[omega - 1 + i]
                if prob_idx + i < len(probs):
                    for k, col in enumerate(cluster_feature_names):
                        augmented_data.loc[row_idx, col] = probs[prob_idx + i, k]

            prob_idx += n_windows

        augmented_features = feature_cols + cluster_feature_names
        return augmented_data, augmented_features


# ==============================================================================
# SECTION 6: CLUSTER INTERPRETATION
# ==============================================================================
# One of the most valuable aspects of the MT-TICC approach is that the
# discovered clusters are INTERPRETABLE. Each cluster represents a distinct
# clinical state in the sepsis progression pathway.
#
# We analyze clusters through:
# 1. Mean feature deviations from normal ranges
# 2. Missing rate patterns (features with high missing rates often have
#    high PageRank in the structural pattern graph)
# 3. Transition probabilities between clusters

# Normal ranges for interpreting cluster deviations
NORMAL_RANGES = {
    'sbp':        (90, 120),       # mmHg
    'map':        (70, 100),       # mmHg
    'hr':         (60, 100),       # bpm
    'rr':         (12, 20),        # breaths/min
    'spo2':       (95, 100),       # %
    'temp':       (36.1, 37.2),    # °C
    'wbc':        (4.5, 11.0),     # K/uL
    'bilirubin':  (0.1, 1.2),      # mg/dL
    'bun':        (7, 20),         # mg/dL
    'lactate':    (0.5, 2.0),      # mmol/L
    'creatinine': (0.6, 1.2),      # mg/dL
    'platelet':   (150, 400),      # K/uL
    'neutrophils':(40, 70),        # %
    'fio2':       (21, 50),        # % (room air=21 to moderate O2 support)
}

# Clinical interpretations for the 6 clusters discovered in the paper
CLUSTER_INTERPRETATIONS = {
    1: "Metabolic Dysfunction - Mild abnormalities in metabolic markers",
    2: "Renal Dysfunction - Elevated BUN and creatinine indicating kidney stress",
    3: ("Non-temperature Physiological Response to Infection, "
        "Cellular Response, Renal Dysfunction"),
    4: ("Non-temperature Physiological Response to Infection, "
        "Metabolic Dysfunction"),
    5: ("Non-temperature Physiological Response to Infection, "
        "Metabolic Dysfunction, Renal Dysfunction, "
        "Gastrointestinal Dysfunction"),
    6: ("Non-temperature Physiological Response to Infection, "
        "Cellular Response, Metabolic Dysfunction, Renal Dysfunction"),
}


def analyze_cluster_patterns(data, labels, feature_cols, n_clusters,
                              scaler=None):
    """
    Analyze the clinical patterns captured by each cluster.

    For each cluster, we compute:
    1. Mean values of each feature IN ORIGINAL SCALE (un-normalized)
    2. Deviation from normal ranges (how "abnormal" each feature is)
    3. The severity ranking (more deviations = more severe cluster)

    IMPORTANT: The data passed to this function is z-score normalized.
    We must inverse-transform it back to original physiological units
    before comparing against NORMAL_RANGES. Without this, every cluster
    appears identical because z-scored means are all near zero.

    Parameters:
    -----------
    data : DataFrame with clinical measurements (z-score normalized)
    labels : cluster assignments
    feature_cols : list of feature column names
    n_clusters : number of clusters
    scaler : fitted StandardScaler used during preprocessing (required)

    Returns:
    --------
    deviation_df : DataFrame of normalized deviations from normal ranges
    severity_order : clusters ranked from least to most severe
    """
    deviations = np.zeros((len(feature_cols), n_clusters))

    # Labels correspond to subsequences (one per sliding window), which may
    # be fewer than data rows. Extract the feature values for each label
    # using numpy indexing for alignment safety.
    feature_values = data[feature_cols].values
    # If labels and data lengths differ, truncate data from the end
    # (each series loses the first ω-1 rows to the sliding window)
    if len(labels) < len(feature_values):
        feature_values = feature_values[-len(labels):]

    # CRITICAL: Inverse-transform back to original physiological units.
    # Without this, we'd compare z-scored values (mean≈0) against raw
    # normal ranges (e.g., SBP 90-120 mmHg), making all clusters identical.
    if scaler is not None:
        feature_values_orig = scaler.inverse_transform(feature_values)
    else:
        feature_values_orig = feature_values

    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        cluster_means = pd.Series(
            feature_values_orig[mask].mean(axis=0), index=feature_cols
        )

        for j, feat in enumerate(feature_cols):
            normal_low, normal_high = NORMAL_RANGES[feat]
            normal_mid = (normal_low + normal_high) / 2.0
            normal_range = normal_high - normal_low
            if normal_range == 0:
                normal_range = 1.0
            # Deviation = how far from normal midpoint, normalized by range
            deviations[j, k] = abs(cluster_means[feat] - normal_mid) / normal_range

    # Normalize the full matrix to [0, 1] for visualization.
    # Use global max so relative differences between clusters AND features
    # are preserved - this lets the reader see which clusters are more
    # abnormal across the board.
    global_max = deviations.max()
    if global_max > 0:
        deviations = deviations / global_max

    deviation_df = pd.DataFrame(
        deviations,
        index=feature_cols,
        columns=[f'Cluster {k+1}' for k in range(n_clusters)]
    )

    # Rank clusters by total deviation (severity)
    severity_scores = deviations.sum(axis=0)
    severity_order = np.argsort(severity_scores) + 1  # 1-indexed

    return deviation_df, severity_order


def compute_transition_matrix(labels, series_ids, n_clusters):
    """
    Compute the transition probability matrix between clusters.

    This reveals how patients typically progress through disease states.
    For example, the paper found that:
    - Shock patients transition to MORE SEVERE clusters more often
    - Non-shock patients transition to LESS SEVERE clusters more often
    - This differential pattern is one of the key clinical insights

    Parameters:
    -----------
    labels : cluster assignments
    series_ids : series (admission) identifiers
    n_clusters : number of clusters

    Returns:
    --------
    transition_matrix : (K × K) matrix of transition probabilities
    """
    transitions = np.zeros((n_clusters, n_clusters))

    for i in range(1, len(labels)):
        if series_ids[i] == series_ids[i - 1]:
            from_cluster = labels[i - 1]
            to_cluster = labels[i]
            transitions[from_cluster, to_cluster] += 1

    # Normalize rows to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transitions / row_sums

    return transition_matrix


# ==============================================================================
# SECTION 7: SEPTIC SHOCK EARLY PREDICTION WITH LSTM
# ==============================================================================
# The clusters from MT-TICC serve as additional features for predicting
# septic shock. We use a Long Short-Term Memory (LSTM) network because:
#
# 1. LSTMs handle VARIABLE-LENGTH sequences naturally
# 2. They capture LONG-RANGE dependencies in clinical trajectories
# 3. They've shown strong performance in EHR prediction tasks
#
# We use PyTorch for its native Apple Silicon (MPS) acceleration,
# portability across platforms, and clean API for custom training loops.
#
# PREDICTION TASK:
#   Given a patient's observations up to τ hours before the endpoint,
#   predict whether the visit will develop into septic shock.
#
#   - For shock visits: endpoint = onset time of septic shock
#   - For non-shock visits: endpoint = discharge time
#   - τ (hold-off window): we test τ ∈ [12, 24] and [24, 36] hours
#
#   This setup tests whether we can predict shock EARLY ENOUGH for
#   clinical intervention.

def prepare_sequences_for_lstm(data, cohort, feature_cols, max_seq_len=100,
                               tau_min=12, tau_max=24):
    """
    Prepare padded sequences and labels for LSTM training.

    Each patient's admission becomes one sequence. We truncate to the
    observation window (everything before τ hours from endpoint) and
    pad/truncate to a fixed length for batched training.

    Parameters:
    -----------
    data : DataFrame with clinical time-series (+ cluster features)
    cohort : DataFrame with admission info and shock labels
    feature_cols : list of feature column names
    max_seq_len : maximum sequence length (for padding)
    tau_min : minimum hold-off window (hours)
    tau_max : maximum hold-off window (hours)

    Returns:
    --------
    X : array of shape (n_patients, max_seq_len, n_features)
    y : binary labels (0=non-shock, 1=shock)
    hadm_ids : corresponding admission IDs
    """
    sequences = []
    seq_lengths = []
    labels = []
    hadm_ids = []

    for _, row in cohort.iterrows():
        hadm_id = row['hadm_id']
        shock_flag = row['septic_shock_flag']

        # Get this admission's time series
        admission_data = data[data['hadm_id'] == hadm_id].sort_values('charttime')

        if len(admission_data) < 5:
            continue  # Skip admissions with too few data points

        # Apply hold-off window
        # Only use data from τ hours before the endpoint.
        # Use the midpoint of the τ range for consistency across patients.
        endpoint = admission_data['charttime'].max()
        tau = (tau_min + tau_max) / 2.0  # Fixed τ for reproducibility
        cutoff = endpoint - timedelta(hours=tau)
        observation_data = admission_data[admission_data['charttime'] <= cutoff]

        if len(observation_data) < 3:
            continue  # Need minimum observations

        # Extract feature values
        seq = observation_data[feature_cols].values

        # Truncate to max_seq_len (keep the most recent events)
        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]

        actual_len = len(seq)

        # Pad to max_seq_len. We use the sequence's own mean for padding
        # instead of zeros, because in z-scored data, zero = population mean,
        # which introduces misleading "average patient" signal. Padding with
        # the patient's own mean is a neutral signal that won't bias the LSTM.
        if len(seq) < max_seq_len:
            pad_value = seq.mean(axis=0, keepdims=True)  # (1, n_features)
            pad_rows = np.repeat(pad_value, max_seq_len - len(seq), axis=0)
            seq = np.vstack([pad_rows, seq])  # Pre-pad

        sequences.append(seq)
        seq_lengths.append(actual_len)
        labels.append(shock_flag)
        hadm_ids.append(hadm_id)

    X = np.array(sequences)
    y = np.array(labels)
    hadm_ids = np.array(hadm_ids)
    lengths = np.array(seq_lengths)

    print(f"  Prepared {len(X)} sequences for LSTM")
    print(f"  Sequence shape: {X.shape}")
    print(f"  Label distribution: {np.bincount(y)}")
    print(f"  Sequence lengths: min={lengths.min()}, "
          f"median={int(np.median(lengths))}, max={lengths.max()}")

    return X, y, hadm_ids, lengths


class SepticShockLSTM(nn.Module):
    """
    Bidirectional LSTM model for septic shock prediction (PyTorch).

    Architecture:
    - Bidirectional LSTM (reads forward AND backward through time)
    - Dropout for regularization (prevents overfitting)
    - Dense layers for classification
    - Sigmoid output for binary prediction

    Why Bidirectional? While clinical events happen sequentially,
    reading the sequence in both directions allows the model to
    understand context from both past and future within the
    observation window.

    Parameters:
    -----------
    n_features : int - Number of input features per time step
    hidden_size : int - Number of LSTM hidden units
    num_layers : int - Number of stacked LSTM layers
    dropout : float - Dropout rate for regularization
    """

    def __init__(self, n_features, hidden_size=32, num_layers=1, dropout=0.3):
        super().__init__()

        # Bidirectional LSTM
        # bidirectional=True reads the sequence forwards AND backwards,
        # effectively doubling the hidden representation.
        # We use a single layer with 32 hidden units to keep the model
        # small enough for the ~1700 sample dataset (avoids overfitting).
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_size = hidden_size

        # Dense layers for classification
        # The LSTM output is 2 * hidden_size because of bidirectionality
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        # With pre-padding, real data is at the END of the sequence,
        # so the last time step output contains the most recent clinical state.
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden*2)
        last_hidden = lstm_out[:, -1, :] # Take last time step output
        return self.classifier(last_hidden).squeeze(-1)


def build_lstm_model(n_features, device='cpu'):
    """
    Build and return the Bidirectional LSTM model.

    Parameters:
    -----------
    n_features : int - Number of input features per time step
    device : str - 'cpu', 'cuda', or 'mps' (Apple Silicon GPU)

    Returns:
    --------
    model : SepticShockLSTM on the specified device
    """
    model = SepticShockLSTM(n_features=n_features).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLSTM Model Summary:")
    print(f"  Architecture: Bidirectional LSTM ({model.lstm.num_layers} layer, {model.hidden_size} hidden) → Sigmoid")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    return model


def _get_device():
    """
    Select the best available compute device.

    Priority: MPS (Apple Silicon GPU) > CUDA (NVIDIA GPU) > CPU
    MPS acceleration is available on M1/M2/M3 Macs with PyTorch >= 1.12.
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_and_evaluate(X, y, n_folds=3, epochs=50, batch_size=32):
    """
    Train and evaluate the LSTM using stratified k-fold cross-validation.

    Following the paper, we use 3-fold cross-validation. This provides
    robust estimates of model performance and allows us to compute
    mean +/- standard deviation for each metric.

    Training includes:
    - Early stopping (patience=5) to prevent overfitting
    - Learning rate reduction on plateau (patience=3)
    - 15% validation split from training data for monitoring

    Metrics reported (matching the paper):
    - Accuracy (Acc): Overall correct predictions
    - Recall (Rec): Sensitivity - fraction of actual shocks caught
    - Precision (Prec): Positive predictive value
    - F1 Score: Harmonic mean of precision and recall
    - AUC: Area Under the ROC Curve

    Parameters:
    -----------
    X : input sequences (n_samples, seq_len, n_features) - numpy array
    y : binary labels - numpy array
    n_folds : number of CV folds
    epochs : maximum training epochs
    batch_size : batch size for training

    Returns:
    --------
    results_df : DataFrame with metrics for each fold
    """
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATION")
    print("=" * 60)

    device = _get_device()
    print(f"  Using device: {device}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        torch.manual_seed(42 + fold)

        # Split data
        X_train_np, X_test_np = X[train_idx], X[test_idx]
        y_train_np, y_test_np = y[train_idx], y[test_idx]

        # Further split training into train/validation (85%/15%)
        n_val = int(len(X_train_np) * 0.15)
        perm = np.random.RandomState(42).permutation(len(X_train_np))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        X_tr = torch.FloatTensor(X_train_np[tr_idx]).to(device)
        y_tr = torch.FloatTensor(y_train_np[tr_idx]).to(device)
        X_val = torch.FloatTensor(X_train_np[val_idx]).to(device)
        y_val = torch.FloatTensor(y_train_np[val_idx]).to(device)
        X_te = torch.FloatTensor(X_test_np).to(device)

        # Create DataLoader for batched training
        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)

        # Build fresh model for each fold
        model = build_lstm_model(n_features=X.shape[2], device=str(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,
                                      weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=False
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                # Gradient clipping prevents exploding gradients in LSTMs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
            epoch_loss /= len(X_tr)

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 8:
                    print(f"  Early stopping at epoch {epoch + 1} "
                          f"(val_loss: {best_val_loss:.4f})")
                    break

        # Restore best model weights
        if best_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_state.items()}
            )

        # --- Evaluate on test set ---
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_te).cpu().numpy()
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'fold': fold + 1,
            'accuracy': accuracy_score(y_test_np, y_pred),
            'recall': recall_score(y_test_np, y_pred),
            'precision': precision_score(y_test_np, y_pred, zero_division=0),
            'f1': f1_score(y_test_np, y_pred),
            'auc': roc_auc_score(y_test_np, y_pred_prob)
        }
        all_metrics.append(metrics)

        print(f"  Acc: {metrics['accuracy']:.3f}  "
              f"Rec: {metrics['recall']:.3f}  "
              f"Prec: {metrics['precision']:.3f}  "
              f"F1: {metrics['f1']:.3f}  "
              f"AUC: {metrics['auc']:.3f}")

    results_df = pd.DataFrame(all_metrics)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Mean +/- Std)")
    print("=" * 60)
    for metric in ['accuracy', 'recall', 'precision', 'f1', 'auc']:
        mean = results_df[metric].mean()
        std = results_df[metric].std()
        print(f"  {metric.upper():>10s}: {mean:.3f} +/- {std:.3f}")

    return results_df


# ==============================================================================
# SECTION 8: VISUALIZATION
# ==============================================================================
# Visualization is crucial for interpreting disease progression patterns.
# We create several plots that mirror the figures in the paper.

def plot_cluster_deviations(deviation_df, save_path=None):
    """
    Visualize feature deviations from normal ranges for each cluster.

    This creates a heatmap similar to Figure 3(a) in the paper.
    Darker colors indicate more abnormal values, helping clinicians
    understand what each cluster represents.

    Parameters:
    -----------
    deviation_df : DataFrame of normalized deviations
    save_path : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        deviation_df,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'label': 'Deviation from Normal'}
    )

    ax.set_title('Feature Deviations from Normal Ranges by Cluster',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Clinical Feature', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_transition_heatmap(transition_matrix, title="Cluster Transitions",
                            save_path=None):
    """
    Visualize cluster transition probabilities as a heatmap.

    Parameters:
    -----------
    transition_matrix : (K × K) transition probability matrix
    title : plot title
    save_path : optional path to save
    """
    n_clusters = transition_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[f'C{k+1}' for k in range(n_clusters)],
        yticklabels=[f'C{k+1}' for k in range(n_clusters)],
        ax=ax,
        vmin=0,
        vmax=0.6,
        linewidths=0.5
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('To Cluster', fontsize=12)
    ax.set_ylabel('From Cluster', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_transition_comparison(shock_matrix, nonshock_matrix, save_path=None):
    """
    Compare transition matrices for shock vs non-shock patients side by side.

    This recreates Figure 4(b) from the paper. The key clinical insight:
    - Shock patients transition to MORE SEVERE clusters more often
    - Non-shock patients transition to LESS SEVERE clusters or remain stable

    Parameters:
    -----------
    shock_matrix : (K × K) transition probability matrix for shock patients
    nonshock_matrix : (K × K) transition probability matrix for non-shock patients
    save_path : optional path to save
    """
    n_clusters = shock_matrix.shape[0]
    cluster_labels = [f'C{k+1}' for k in range(n_clusters)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        shock_matrix, annot=True, fmt='.2f', cmap='Reds',
        xticklabels=cluster_labels, yticklabels=cluster_labels,
        ax=ax1, vmin=0, vmax=0.6, linewidths=0.5
    )
    ax1.set_title('Shock Patients\nTransition Probabilities',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('To Cluster', fontsize=12)
    ax1.set_ylabel('From Cluster', fontsize=12)

    sns.heatmap(
        nonshock_matrix, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=cluster_labels, yticklabels=cluster_labels,
        ax=ax2, vmin=0, vmax=0.6, linewidths=0.5
    )
    ax2.set_title('Non-Shock Patients\nTransition Probabilities',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('To Cluster', fontsize=12)
    ax2.set_ylabel('From Cluster', fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_early_prediction_comparison(tau_values, results_dict, save_path=None):
    """
    Compare prediction performance across different methods and τ values.

    This recreates Figure 2 from the paper, showing how F1 and AUC
    change as we increase the hold-off window τ. Key insight:
    the MT-TICC cluster features provide the biggest advantage
    for LARGER τ values (earlier prediction is harder, so the
    additional structural information from clustering helps more).

    Parameters:
    -----------
    tau_values : array of hold-off window values
    results_dict : dict mapping method names to (f1_scores, auc_scores)
    save_path : optional path to save
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'Original': '#1f77b4', 'TICC': '#2ca02c',
              'M-TICC': '#ff7f0e', 'MT-TICC': '#d62728'}

    for method, (f1_scores, auc_scores) in results_dict.items():
        color = colors.get(method, 'gray')
        ax1.plot(tau_values, f1_scores, 'o-', label=method,
                 color=color, linewidth=2, markersize=6)
        ax2.plot(tau_values, auc_scores, 'o-', label=method,
                 color=color, linewidth=2, markersize=6)

    ax1.set_xlabel('τ (hours)', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Early Prediction F1 Score', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('τ (hours)', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('Early Prediction AUC', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


def plot_cluster_timeline(labels, series_ids, n_clusters, n_series=5,
                          cohort=None, save_path=None):
    """
    Visualize the cluster assignment timeline for sample patients.

    This shows how patients transition between disease states over time,
    making the temporal patterns discovered by MT-TICC visually apparent.

    Parameters:
    -----------
    labels : cluster assignments
    series_ids : series identifiers
    n_clusters : number of clusters
    n_series : number of patient timelines to show
    save_path : optional path to save
    """
    # Select a mix of shock and non-shock patients for comparison
    unique_series = np.unique(series_ids)
    if cohort is not None and 'septic_shock_flag' in cohort.columns:
        shock_ids = set(cohort[cohort['septic_shock_flag'] == 1]['hadm_id'].values)
        shock_series = [s for s in unique_series if s in shock_ids]
        nonshock_series = [s for s in unique_series if s not in shock_ids]
        # Pick alternating shock/non-shock patients
        n_each = n_series // 2
        selected = []
        for i in range(max(n_each, n_series - n_each)):
            if i < len(shock_series) and len(selected) < n_series:
                selected.append(('Shock', shock_series[i]))
            if i < len(nonshock_series) and len(selected) < n_series:
                selected.append(('Non-Shock', nonshock_series[i]))
        # Ensure we have exactly n_series
        selected = selected[:n_series]
    else:
        selected = [('Patient', sid) for sid in unique_series[:n_series]]

    fig, axes = plt.subplots(len(selected), 1,
                              figsize=(14, 2 * len(selected)),
                              sharex=False)
    if len(selected) == 1:
        axes = [axes]

    cmap = plt.cm.RdYlGn_r  # Red = severe, Green = mild

    for idx, ((ptype, series_id), ax) in enumerate(zip(selected, axes)):
        mask = series_ids == series_id
        series_labels = labels[mask]

        # Color each time point by its cluster
        for t in range(len(series_labels)):
            color = cmap(series_labels[t] / (n_clusters - 1))
            ax.barh(0, 1, left=t, height=0.8, color=color, edgecolor='white',
                    linewidth=0.3)

        label_text = f'{ptype} {idx // 2 + 1}' if ptype != 'Patient' else f'Patient {idx + 1}'
        ax.set_ylabel(label_text, fontsize=10, rotation=0,
                       labelpad=80, va='center')
        ax.set_xlim(0, len(series_labels))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])

    axes[-1].set_xlabel('Time (event index)', fontsize=12)
    fig.suptitle('Disease Progression: Cluster Assignments Over Time',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_clusters))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                         fraction=0.05, pad=0.1)
    cbar.set_label('Cluster (1=Mild → 6=Severe)', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.show()


# ==============================================================================
# SECTION 9: PUTTING IT ALL TOGETHER - MAIN PIPELINE
# ==============================================================================

def run_full_pipeline():
    """
    Execute the complete disease progression modeling pipeline.

    This function orchestrates the entire workflow:
    1. Connect to MIMIC-III and extract data
    2. Preprocess the clinical time series
    3. Fit the MT-TICC clustering model
    4. Analyze and interpret the discovered clusters
    5. Generate cluster-based features
    6. Train and evaluate the LSTM prediction model
    7. Create all visualizations

    To run this pipeline, ensure you have:
    - MIMIC-III database set up locally (PostgreSQL)
    - All required Python packages installed
    - Sufficient RAM (16+ GB recommended for full dataset)
    """
    print("=" * 70)
    print("DISEASE PROGRESSION MODELING FOR SEPSIS")
    print("Using MT-TICC with MIMIC-III Data")
    print("=" * 70)

    # ----- STEP 1: Data Extraction -----
    print("\n[STEP 1] Loading clinical data...")

    # ---------------------------------------------------------------
    # DATA SOURCE CONFIGURATION
    # ---------------------------------------------------------------
    # Set USE_SYNTHETIC = False to connect to a local MIMIC-III database.
    # The synthetic data generator creates realistic ICU data with
    # 6 discrete clinical states that mirror the paper's findings,
    # allowing the simplified MT-TICC implementation to demonstrate
    # all concepts effectively.
    #
    # The full MT-TICC algorithm (graphical lasso + Viterbi) is needed
    # for meaningful results on real MIMIC-III data.
    # ---------------------------------------------------------------
    USE_SYNTHETIC = True

    use_real_data = False
    if not USE_SYNTHETIC:
        try:
            conn = connect_to_mimic()
            cohort_df, vitals_df, labs_df = extract_mimic_data(conn)
            conn.close()
            use_real_data = True
            print("  Successfully loaded real MIMIC-III data.")
        except Exception as e:
            print(f"\n  Could not connect to MIMIC-III database: {e}")
            print("  Falling back to synthetic data for demonstration.\n")

    if not use_real_data:
        print("  Using synthetic clinical data (200 patients, 6 disease states)")
        print("  To use real MIMIC-III data, set USE_SYNTHETIC = False above")
        print("  and configure MIMIC_DB_CONFIG.\n")
        cohort_df, vitals_df, labs_df = generate_synthetic_mimic_data()

    # ----- STEP 2: Preprocessing -----
    print("\n[STEP 2] Preprocessing clinical data...")
    data_dict = preprocess_pipeline(vitals_df, labs_df, cohort_df)
    data = data_dict['data']
    feature_cols = data_dict['feature_columns']
    cohort = data_dict['cohort']
    scaler = data_dict['scaler']

    # ----- STEP 3: Fit MT-TICC -----
    print("\n[STEP 3] Fitting MT-TICC clustering model...")
    mt_ticc = SimplifiedMTTICC(
        n_clusters=6,       # K=6 (determined by BIC in the paper)
        window_size=3,       # ω=3 (window of 3 consecutive events)
        beta=2.0,            # β=2 (consistency weight - lower values allow more cluster diversity)
        lambda_reg=1e-4,     # λ=1e-4 (sparsity coefficient)
        max_iter=50
    )
    mt_ticc.fit(data, feature_cols)

    # ----- STEP 4: Cluster Analysis -----
    print("\n[STEP 4] Analyzing cluster patterns...")
    labels = mt_ticc.labels_
    series_ids = mt_ticc.series_ids_

    deviation_df, severity_order = analyze_cluster_patterns(
        data, labels, feature_cols, n_clusters=6, scaler=scaler
    )
    print("\n  Cluster Interpretations (least to most severe):")
    for rank, cluster_idx in enumerate(severity_order):
        interp = CLUSTER_INTERPRETATIONS.get(cluster_idx, "Unknown")
        print(f"    {rank + 1}. Cluster {cluster_idx}: {interp}")

    transition_matrix = compute_transition_matrix(labels, series_ids, n_clusters=6)

    # Compute separate transition matrices for shock vs non-shock patients.
    # This is a key analysis from the paper (Figure 4b): shock patients
    # transition toward MORE severe clusters, non-shock toward LESS severe.
    shock_hadm_ids = set(cohort[cohort['septic_shock_flag'] == 1]['hadm_id'].values)
    shock_mask = np.array([sid in shock_hadm_ids for sid in series_ids])
    nonshock_mask = ~shock_mask

    shock_labels = labels[shock_mask]
    shock_series = series_ids[shock_mask]
    nonshock_labels = labels[nonshock_mask]
    nonshock_series = series_ids[nonshock_mask]

    shock_transitions = compute_transition_matrix(shock_labels, shock_series, n_clusters=6)
    nonshock_transitions = compute_transition_matrix(nonshock_labels, nonshock_series, n_clusters=6)

    # ----- STEP 5: Visualization -----
    print("\n[STEP 5] Generating visualizations...")
    plot_cluster_deviations(deviation_df, save_path='cluster_deviations.png')
    plot_transition_comparison(shock_transitions, nonshock_transitions,
                               save_path='transitions.png')
    plot_cluster_timeline(labels, series_ids, n_clusters=6,
                          cohort=cohort, save_path='timeline.png')

    # ----- STEP 6: LSTM Prediction -----
    print("\n[STEP 6] Training LSTM for septic shock prediction...")
    # Generate cluster features
    _, probs = mt_ticc.predict(data, feature_cols)
    augmented_data, augmented_features = mt_ticc.get_cluster_features(
        data, labels, probs, feature_cols
    )

    # Compare Original (O) vs Original + Clusters (O+C)
    for feature_set_name, feat_cols in [
        ("Original (O)", feature_cols),
        ("Original + Clusters (O+C)", feature_cols + [
            f'cluster_{k+1}_prob' for k in range(6)
        ])
    ]:
        print(f"\n--- {feature_set_name} ---")
        X, y, _, lengths = prepare_sequences_for_lstm(
            augmented_data, cohort, feat_cols,
            max_seq_len=50, tau_min=12, tau_max=24
        )

        if len(X) > 0:
            results = train_and_evaluate(X, y, n_folds=3, epochs=50)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. MT-TICC discovers interpretable disease progression states")
    print("  2. Cluster features improve septic shock prediction accuracy")
    print("  3. Time-awareness captures irregular EHR sampling patterns")
    print("  4. Multi-series input learns shared patterns across patients")
    print("\nImportant Limitations (see Section 11 for details):")
    print("  - This tutorial uses a SIMPLIFIED MT-TICC implementation")
    print("  - The full algorithm requires graphical lasso, Viterbi DP,")
    print("    and Toeplitz constraints for real clinical data")
    print("  - Synthetic data is used to demonstrate concepts; real MIMIC-III")
    print("    data would require the full algorithmic pipeline")


# ==============================================================================
# SECTION 10: SYNTHETIC DATA GENERATION (for demonstration)
# ==============================================================================
# When MIMIC-III database access is not available, this function generates
# realistic synthetic clinical data that mirrors the structure, distributions,
# and patterns of real ICU data.

def generate_synthetic_mimic_data(n_patients=200, seed=42):
    """
    Generate synthetic MIMIC-III-like clinical data for demonstration.

    The synthetic data replicates key properties of real ICU data:
    - Irregular time intervals between measurements
    - High missing rates (especially for lab values)
    - Physiologically correlated features
    - Different trajectories for shock vs non-shock patients

    Parameters:
    -----------
    n_patients : number of synthetic patients
    seed : random seed

    Returns:
    --------
    cohort_df, vitals_df, labs_df : DataFrames matching MIMIC-III structure
    """
    np.random.seed(seed)

    n_shock = n_patients // 2
    n_non_shock = n_patients - n_shock

    cohort_records = []
    vitals_records = []
    labs_records = []

    # ---------------------------------------------------------------
    # Define 6 discrete clinical states with distinct feature profiles
    # matching the paper's discovered clusters (Table II).
    # Each state is a dict of (mean, std) per feature.
    # ---------------------------------------------------------------
    STATES = {
        0: {  # State 1: Near-normal (mild metabolic)
            'sbp': (115, 8), 'map': (82, 6), 'hr': (78, 8), 'rr': (16, 2),
            'spo2': (97, 1.5), 'temp': (37.0, 0.3),
            'wbc': (9, 2), 'bilirubin': (0.8, 0.3), 'bun': (14, 4),
            'lactate': (1.2, 0.3), 'creatinine': (0.9, 0.2), 'platelet': (260, 30),
            'neutrophils': (58, 6), 'fio2': (21, 1),
        },
        1: {  # State 2: Renal dysfunction
            'sbp': (108, 10), 'map': (75, 8), 'hr': (85, 10), 'rr': (18, 2),
            'spo2': (96, 1.5), 'temp': (37.2, 0.4),
            'wbc': (11, 3), 'bilirubin': (1.0, 0.4), 'bun': (28, 6),
            'lactate': (1.8, 0.4), 'creatinine': (1.8, 0.4), 'platelet': (220, 35),
            'neutrophils': (65, 7), 'fio2': (25, 3),
        },
        2: {  # State 3: Physiological response + cellular + renal
            'sbp': (100, 12), 'map': (68, 8), 'hr': (95, 12), 'rr': (22, 3),
            'spo2': (94, 2), 'temp': (37.0, 0.4),  # Normal temp (non-febrile)
            'wbc': (15, 3), 'bilirubin': (1.5, 0.5), 'bun': (32, 8),
            'lactate': (2.5, 0.5), 'creatinine': (2.2, 0.5), 'platelet': (180, 30),
            'neutrophils': (72, 6), 'fio2': (35, 5),
        },
        3: {  # State 4: Physiological response + metabolic
            'sbp': (95, 12), 'map': (62, 8), 'hr': (105, 12), 'rr': (24, 3),
            'spo2': (92, 2), 'temp': (38.2, 0.5),
            'wbc': (17, 4), 'bilirubin': (2.5, 0.8), 'bun': (35, 8),
            'lactate': (3.2, 0.6), 'creatinine': (2.0, 0.5), 'platelet': (150, 30),
            'neutrophils': (78, 5), 'fio2': (45, 8),
        },
        4: {  # State 5: Severe - multi-organ
            'sbp': (85, 12), 'map': (55, 8), 'hr': (115, 12), 'rr': (28, 3),
            'spo2': (89, 3), 'temp': (38.8, 0.6),
            'wbc': (20, 4), 'bilirubin': (3.5, 1.0), 'bun': (45, 10),
            'lactate': (4.5, 0.8), 'creatinine': (3.0, 0.6), 'platelet': (100, 25),
            'neutrophils': (82, 5), 'fio2': (60, 10),
        },
        5: {  # State 6: Most severe - organ failure
            'sbp': (75, 10), 'map': (48, 6), 'hr': (125, 10), 'rr': (32, 3),
            'spo2': (85, 3), 'temp': (39.2, 0.6),
            'wbc': (22, 5), 'bilirubin': (5.0, 1.5), 'bun': (60, 12),
            'lactate': (6.0, 1.0), 'creatinine': (4.0, 0.8), 'platelet': (60, 20),
            'neutrophils': (88, 4), 'fio2': (80, 10),
        },
    }

    def sample_from_state(state_idx):
        """Sample vital + lab values from a discrete clinical state."""
        s = STATES[state_idx]
        vitals = {
            'sbp': max(60, np.random.normal(*s['sbp'])),
            'map': max(40, np.random.normal(*s['map'])),
            'hr': max(40, np.random.normal(*s['hr'])),
            'rr': max(8, np.random.normal(*s['rr'])),
            'spo2': min(100, max(70, np.random.normal(*s['spo2']))),
            'temp': np.random.normal(*s['temp']),
        }
        labs = {
            'wbc': max(1, np.random.normal(*s['wbc'])),
            'bilirubin': max(0.1, np.random.normal(*s['bilirubin'])),
            'bun': max(5, np.random.normal(*s['bun'])),
            'lactate': max(0.5, np.random.normal(*s['lactate'])),
            'creatinine': max(0.3, np.random.normal(*s['creatinine'])),
            'platelet': max(20, np.random.normal(*s['platelet'])),
            'neutrophils': min(95, max(10, np.random.normal(*s['neutrophils']))),
            'fio2': min(100, max(21, np.random.normal(*s['fio2']))),
        }
        return vitals, labs

    for i in range(n_patients):
        subject_id = 10000 + i
        hadm_id = 20000 + i
        is_shock = 1 if i < n_shock else 0

        # Demographics
        age = np.random.normal(65, 15)
        gender = np.random.choice(['M', 'F'])
        stay_hours = np.random.uniform(48, 200)

        cohort_records.append({
            'subject_id': subject_id,
            'hadm_id': hadm_id,
            'septic_shock_flag': is_shock,
            'age': max(18, age),
            'gender': gender,
            'ethnicity': np.random.choice(['WHITE', 'BLACK', 'HISPANIC', 'ASIAN']),
            'stay_hours': stay_hours,
            'admittime': pd.Timestamp('2013-01-01') + timedelta(hours=i * 24)
        })

        # ---------------------------------------------------------------
        # Generate a state trajectory for this patient.
        # Shock patients: start mild, escalate to severe states.
        # Non-shock patients: stay in mild states, may briefly worsen then recover.
        # ---------------------------------------------------------------
        n_events = int(stay_hours * np.random.uniform(0.8, 1.5))
        base_time = pd.Timestamp('2013-01-01') + timedelta(hours=i * 24)

        if is_shock:
            # Shock: start in states 0-1, escalate through to severe states.
            # Transition probability is high enough to ensure clear escalation
            # within the stay duration, matching clinical septic shock trajectories.
            max_state = np.random.choice([4, 5], p=[0.3, 0.7])
            state_trajectory = []
            current_state = np.random.choice([0, 1], p=[0.7, 0.3])
            for t in range(n_events):
                progression = t / max(n_events - 1, 1)
                target = min(max_state, int(progression * (max_state + 1)))
                # High escalation probability - shock patients deteriorate
                if current_state < target and np.random.random() < 0.08:
                    current_state = min(current_state + 1, 5)
                # Very low recovery probability - shock rarely reverses
                elif current_state > target and np.random.random() < 0.01:
                    current_state = max(current_state - 1, 0)
                state_trajectory.append(current_state)
        else:
            # Non-shock: mostly states 0-1, may briefly reach state 2, then recover.
            # These patients' immune systems handle the infection.
            state_trajectory = []
            current_state = 0
            peak_time = np.random.uniform(0.2, 0.4)
            for t in range(n_events):
                progression = t / max(n_events - 1, 1)
                if progression < peak_time:
                    target = min(1, int((progression / peak_time) * 2))
                else:
                    target = 0
                if current_state < target and np.random.random() < 0.06:
                    current_state = min(current_state + 1, 2)
                elif current_state > target and np.random.random() < 0.10:
                    current_state = max(current_state - 1, 0)
                state_trajectory.append(current_state)

        # Generate vitals and labs from the state trajectory
        lab_interval = np.random.uniform(4, 8)
        next_lab_time = 0
        for t, state in enumerate(state_trajectory):
            time_offset = timedelta(hours=t * np.random.exponential(1.0))
            charttime = base_time + time_offset
            vitals, labs = sample_from_state(state)

            vitals_records.append({
                'subject_id': subject_id, 'hadm_id': hadm_id,
                'charttime': charttime, **vitals,
            })

            # Labs are less frequent than vitals
            elapsed = time_offset.total_seconds() / 3600
            if elapsed >= next_lab_time:
                labs_records.append({
                    'subject_id': subject_id, 'hadm_id': hadm_id,
                    'charttime': charttime, **labs,
                })
                next_lab_time = elapsed + lab_interval

    cohort_df = pd.DataFrame(cohort_records)
    vitals_df = pd.DataFrame(vitals_records)
    labs_df = pd.DataFrame(labs_records)

    print(f"  Generated synthetic data:")
    print(f"    {len(cohort_df)} patients ({n_shock} shock, {n_non_shock} non-shock)")
    print(f"    {len(vitals_df)} vital sign records")
    print(f"    {len(labs_df)} lab result records")

    return cohort_df, vitals_df, labs_df


# ==============================================================================
# SECTION 11: LIMITATIONS AND DISCUSSION
# ==============================================================================
#
# This section documents the key limitations of our simplified MT-TICC
# implementation compared to the full algorithm described by Yang et al. (2021).
# Understanding these differences is critical for interpreting the results.
#
# ---------------------------------------------------------------------------
# WHY SYNTHETIC DATA?
# ---------------------------------------------------------------------------
#
# This tutorial defaults to synthetic data (USE_SYNTHETIC = True) because our
# simplified implementation cannot reliably discover clinically meaningful
# clusters on real MIMIC-III data. After extensive testing with real MIMIC-III
# data, the simplified clustering produced undifferentiated clusters (nearly
# identical feature profiles) and the LSTM prediction model remained near
# chance level (AUC ~0.55-0.60). The synthetic data generator plants 6
# discrete clinical states with known structure, allowing the simplified
# algorithm to demonstrate all pipeline concepts effectively.
#
# ---------------------------------------------------------------------------
# LIMITATION 1: No Graphical Lasso (Sparse Inverse Covariance Estimation)
# ---------------------------------------------------------------------------
#
# The full MT-TICC learns a SPARSE INVERSE COVARIANCE (precision) matrix for
# each cluster using the Alternating Direction Method of Multipliers (ADMM).
# The precision matrix captures CONDITIONAL dependencies between features:
# which variables directly influence each other after accounting for all
# other variables.
#
# Our simplified version uses standard covariance matrices from GMM, which
# capture MARGINAL correlations - not the same thing. Marginal correlations
# are dominated by confounders (e.g., "sicker patients have worse values on
# everything"), obscuring the sparse dependency structures that distinguish
# clinical states.
#
# Impact: The simplified version sees "similar mean values" rather than
# "similar dependency structures," so clusters collapse into near-identical
# profiles on real data.
#
# ---------------------------------------------------------------------------
# LIMITATION 2: No Viterbi Dynamic Programming (Global Optimal Assignment)
# ---------------------------------------------------------------------------
#
# The full MT-TICC uses a Viterbi-like dynamic programming E-step to find the
# GLOBALLY optimal sequence of cluster assignments for each patient stay.
# This considers the entire trajectory simultaneously, enforcing temporal
# coherence: the algorithm pays a penalty (beta) for switching clusters but
# finds the assignment path that MINIMIZES total cost across the full stay.
#
# Our simplified version uses a GREEDY FORWARD PASS: at each time point, it
# picks the locally best cluster considering only the previous assignment.
# This myopic approach can get stuck in suboptimal assignments early on that
# cascade through the rest of the sequence.
#
# Impact: Without global optimization, the simplified version produces
# noisier, less temporally coherent cluster assignments, especially for
# longer ICU stays.
#
# ---------------------------------------------------------------------------
# LIMITATION 3: No Block Toeplitz Constraint
# ---------------------------------------------------------------------------
#
# The full MT-TICC enforces that learned precision matrices have a block
# Toeplitz structure. This means temporal dependencies at the same lag are
# constrained to be identical regardless of where in the window they occur.
# For example, if heart rate at lag-1 depends on blood pressure at lag-0,
# this relationship is the same whether we look at positions (0,1) or (1,2).
#
# Our simplified version does not enforce this structural constraint,
# allowing the model to overfit to position-specific patterns rather than
# learning generalizable temporal relationships.
#
# Impact: Without Toeplitz constraints, the model has more parameters to
# estimate and less inductive bias for learning time-invariant patterns.
#
# ---------------------------------------------------------------------------
# WHAT WOULD BE NEEDED FOR REAL MIMIC-III RESULTS
# ---------------------------------------------------------------------------
#
# To achieve the paper's results on real clinical data, one would need:
#
# 1. ADMM solver for graphical lasso with L1 penalty on inverse covariance
# 2. Dynamic programming E-step (Viterbi) with temporal consistency penalty
# 3. Block Toeplitz constraints on the precision matrices
# 4. Proper BIC-based model selection for number of clusters (K) and window
#    size (omega)
# 5. Significantly more data (the paper used thousands of ICU stays)
#
# These are non-trivial to implement correctly and constitute the core
# algorithmic contribution of the MT-TICC paper. Libraries such as TICC
# (https://github.com/davidhallac/TICC) implement some of these components.
#
# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
#
# Our simplified implementation is essentially K-Means on windowed
# subsequences with a temporal consistency penalty. This is sufficient for:
#   - Demonstrating the MT-TICC pipeline concept
#   - Showing how cluster features augment prediction
#   - Visualizing disease progression patterns
#
# But insufficient for:
#   - Discovering meaningful clusters from raw real-world EHR data
#   - Reproducing the paper's quantitative results
#   - Clinical deployment
#
# ==============================================================================


# ==============================================================================
# ENTRY POINT
# ==============================================================================
# Run the full pipeline by executing this script.
# To use with real MIMIC-III data, set USE_SYNTHETIC = False in run_full_pipeline()
# and configure MIMIC_DB_CONFIG. Note: the simplified implementation may not produce
# meaningful results on real data without the full algorithmic components (see Section 11).

if __name__ == "__main__":
    run_full_pipeline()
