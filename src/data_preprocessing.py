"""
data_preprocessing.py
----------------------
Loads the raw Kaggle "Freelancer Earnings & Job Trends" dataset and
cleans it into an analysis-ready DataFrame.

Dataset source (download manually or via Kaggle API):
    https://www.kaggle.com/datasets/shohinurpervezshohan/freelancer-earnings-and-job-trends

Expected raw columns:
    Freelancer_ID, Job_Category, Platform, Experience_Level,
    Client_Region, Payment_Method, Job_Completed, Earnings_USD,
    Hourly_Rate, Job_Success_Rate, Client_Rating, Job_Duration_Days,
    Project_Type, Rehire_Rate, Marketing_Spend

The loader is intentionally defensive: it does not assume every
column is present or perfectly named, because real-world exports
(and future dataset versions) are messy. Anything it can't find it
skips with a warning instead of crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Columns we expect, and a sensible dtype family for each.
NUMERIC_COLUMNS = [
    "Job_Completed",
    "Earnings_USD",
    "Hourly_Rate",
    "Job_Success_Rate",
    "Client_Rating",
    "Job_Duration_Days",
    "Rehire_Rate",
    "Marketing_Spend",
]

CATEGORICAL_COLUMNS = [
    "Job_Category",
    "Platform",
    "Experience_Level",
    "Client_Region",
    "Payment_Method",
    "Project_Type",
]

ID_COLUMN = "Freelancer_ID"


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the dataset at '{path}'.\n"
            "Download 'Freelancer Earnings & Job Trends' from Kaggle and place "
            "the CSV in data/raw/ (see README for instructions)."
        )
    df = pd.read_csv(path)
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _strip_strings(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pass:
      1. Drop exact duplicate rows.
      2. Coerce numeric columns, strip categorical text.
      3. Drop rows with a missing/invalid ID.
      4. Impute missing numerics with the column median (robust to outliers).
      5. Impute missing categoricals with 'Unknown'.
      6. Remove physically impossible values (negative earnings, rates
         outside 0-100%, ratings outside 1-5, etc.).
      7. Winsorize extreme outliers in earnings-related columns at the
         1st/99th percentile so a handful of freak entries don't distort
         the clustering geometry.
    """
    df = df.copy()
    before = len(df)

    df = df.drop_duplicates()

    present_numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]
    present_categorical = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    df = _coerce_numeric(df, present_numeric)
    df = _strip_strings(df, present_categorical)

    if ID_COLUMN in df.columns:
        df = df.dropna(subset=[ID_COLUMN])

    # Impute
    for col in present_numeric:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in present_categorical:
        df[col] = df[col].replace({"nan": np.nan, "": np.nan})
        df[col] = df[col].fillna("Unknown")

    # Sanity bounds
    if "Job_Success_Rate" in df.columns:
        df = df[df["Job_Success_Rate"].between(0, 100)]
    if "Rehire_Rate" in df.columns:
        df = df[df["Rehire_Rate"].between(0, 100)]
    if "Client_Rating" in df.columns:
        df = df[df["Client_Rating"].between(1, 5)]
    for col in ["Earnings_USD", "Hourly_Rate", "Job_Completed",
                "Job_Duration_Days", "Marketing_Spend"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    # Winsorize heavy-tailed money columns
    for col in ["Earnings_USD", "Hourly_Rate", "Marketing_Spend"]:
        if col in df.columns:
            lower, upper = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower, upper)

    df = df.reset_index(drop=True)
    logger.info(
        "Cleaned data: %s -> %s rows (%s removed)",
        before, len(df), before - len(df),
    )
    return df


def load_and_clean(path: str | Path) -> pd.DataFrame:
    """Convenience wrapper: load raw CSV then clean it in one call."""
    return clean_data(load_raw_data(path))
