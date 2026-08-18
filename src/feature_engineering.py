"""
feature_engineering.py
------------------------
Turns the cleaned raw table into the numeric feature matrix that the
clustering model actually trains on.

Design notes
------------
The niche for this project is *global freelancer segmentation* -
grouping freelancers not by who they are demographically, but by how
they operate economically: how much they earn, how efficiently they
convert jobs into income, how reliable clients find them, and how
much they spend acquiring work. That is a much more useful business
lens than raw earnings alone, and it's also what makes the segments
transferable across countries - a "Premium Specialist" in Dhaka and
one in Nairobi look the same in feature space even though their
home markets are completely different.

Engineered features
--------------------
- earnings_per_job        : Earnings_USD / Job_Completed
- earnings_per_day         : Earnings_USD / Job_Duration_Days
- marketing_efficiency     : Earnings_USD / (Marketing_Spend + 1)
- rate_consistency_gap     : Hourly_Rate vs. earnings_per_job spread
- experience_score         : ordinal encoding of Experience_Level
- client_trust_index       : weighted blend of Job_Success_Rate,
                              Client_Rating (scaled to 100) and
                              Rehire_Rate - a single 0-100 "would a new
                              client trust this freelancer" score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPERIENCE_ORDER = {"Beginner": 1, "Intermediate": 2, "Expert": 3}

# The final numeric feature set the clustering model is trained on.
CLUSTER_FEATURES = [
    "Earnings_USD",
    "Hourly_Rate",
    "Job_Completed",
    "Job_Success_Rate",
    "Client_Rating",
    "Rehire_Rate",
    "Job_Duration_Days",
    "Marketing_Spend",
    "earnings_per_job",
    "marketing_efficiency",
    "client_trust_index",
    "experience_score",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Earnings_USD" in df.columns and "Job_Completed" in df.columns:
        df["earnings_per_job"] = df["Earnings_USD"] / df["Job_Completed"].replace(0, np.nan)
        df["earnings_per_job"] = df["earnings_per_job"].fillna(0)

    if "Earnings_USD" in df.columns and "Job_Duration_Days" in df.columns:
        df["earnings_per_day"] = df["Earnings_USD"] / df["Job_Duration_Days"].replace(0, np.nan)
        df["earnings_per_day"] = df["earnings_per_day"].fillna(0)

    if "Earnings_USD" in df.columns and "Marketing_Spend" in df.columns:
        df["marketing_efficiency"] = df["Earnings_USD"] / (df["Marketing_Spend"] + 1)

    if "Experience_Level" in df.columns:
        df["experience_score"] = df["Experience_Level"].map(EXPERIENCE_ORDER).fillna(1)

    trust_components = []
    if "Job_Success_Rate" in df.columns:
        trust_components.append(df["Job_Success_Rate"])
    if "Client_Rating" in df.columns:
        trust_components.append(df["Client_Rating"] / 5 * 100)
    if "Rehire_Rate" in df.columns:
        trust_components.append(df["Rehire_Rate"])
    if trust_components:
        df["client_trust_index"] = np.mean(trust_components, axis=0)
    else:
        df["client_trust_index"] = 0

    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (feature_dataframe, feature_column_names) using only the
    engineered numeric columns that are actually present, so the
    pipeline degrades gracefully if a future dataset version is
    missing one or two fields.
    """
    df = add_engineered_features(df)
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    if len(available) < 4:
        raise ValueError(
            "Not enough usable numeric features to cluster on. "
            f"Found only: {available}"
        )
    return df, available
