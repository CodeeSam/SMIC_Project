"""
src/load_data.py

Responsibilities:
- Load the raw CSV (data/raw_data.csv)
- Validate required columns and types
- Normalize date, follower, and engagement columns
- Save cleaned dataset to data/cleaned_posts.csv
- Create an 80/20 train/test split and save to:
    data/train_sentiment.csv
    data/test_sentiment.csv

Usage (from project root):
$ python -m src.load_data              # runs default path load
or from Python:
from src.load_data import prepare_data_for_project
prepare_data_for_project("data/raw_data.csv")
"""

import os
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]   # project root (smcis/)
DATA_DIR = ROOT / "data"


# -------------------------
# Helpers
# -------------------------
def ensure_data_folders():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    uploads = DATA_DIR / "uploads"
    uploads.mkdir(exist_ok=True)


def parse_date_col(df: pd.DataFrame, date_col: str = "Date", output_col: str = "Date"):
    """
    Parse various date formats into pandas datetime.
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    df[output_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
    return df


def normalize_engagement(val):
    """Return engagement as decimal (0.02 for 2%)."""
    if pd.isna(val):
        return None
    try:
        if isinstance(val, str):
            v = val.strip()
            if v.endswith("%"):
                return float(v.rstrip("%")) / 100.0
            return float(v)
        return float(val)
    except Exception:
        return None


def coerce_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")


# -------------------------
# Core functions
# -------------------------
REQUIRED_COLS = ["Competitor_handle", "Date", "Post_text", "Followers", "Avg_Engagement_Rate"]


def validate_columns(df: pd.DataFrame) -> Tuple[bool, list]:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    return (len(missing) == 0, missing)


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert known variants to canonical column names.
    e.g. avg_engagement_rate, avg_eng -> Avg_Engagement_Rate
    """
    df = df.copy()
    col_map = {}
    lower_map = {c.lower(): c for c in df.columns}
    # Detect common variants
    variants = {
        "competitor_handle": "Competitor_handle",
        "competitor": "Competitor_handle",
        "handle": "Competitor_handle",
        "date": "Date",
        "post_text": "Post_text",
        "text": "Post_text",
        "followers": "Followers",
        "follower_count": "Followers",
        "avg_engagement_rate": "Avg_Engagement_Rate",
        "avg_eng": "Avg_Engagement_Rate",
        "engagement_rate": "Avg_Engagement_Rate",
        "target_label": "target_label",
        "label": "target_label",
    }
    for low, col in lower_map.items():
        if low in variants:
            col_map[col] = variants[low]

    if col_map:
        df = df.rename(columns=col_map)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned dataframe with canonical columns, parsed dates,
    numeric follower counts and normalized engagement.
    """
    df = canonicalize_columns(df)
    ok, missing = validate_columns(df)
    if not ok:
        raise ValueError(f"Missing required columns: {missing}")

    # trim whitespace on handles and text
    df["Competitor_handle"] = df["Competitor_handle"].astype(str).str.strip()
    df["Post_text"] = df["Post_text"].astype(str).fillna("")

    # parse date
    df = parse_date_col(df, date_col="Date", output_col="Date")
    # drop rows with invalid dates
    df = df[~df["Date"].isna()].copy()

    # numeric conversions
    df["Followers"] = coerce_numeric_series(df["Followers"])
    df["Avg_Engagement_Rate"] = df["Avg_Engagement_Rate"].apply(normalize_engagement)

    # Optional: ensure target_label exists; if not, create placeholder -1
    if "target_label" not in df.columns:
        df["target_label"] = -1

    # Standardize column order
    cols = ["Competitor_handle", "Date", "Post_text", "target_label", "Followers", "Avg_Engagement_Rate"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    # Sort by competitor and date
    df = df.sort_values(["Competitor_handle", "Date"]).reset_index(drop=True)
    return df


def train_test_split_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified-ish split if possible (stratify by target_label if valid),
    else random split while preserving competitor distribution via group split fallback.
    Saves train_sentiment.csv and test_sentiment.csv to data/
    """
    df = df.copy()
    # If target_label has at least 2 classes, try stratify
    stratify_col = None
    if df["target_label"].nunique() > 1 and df["target_label"].notna().sum() > 0:
        try:
            stratify_col = df["target_label"]
        except Exception:
            stratify_col = None

    if stratify_col is not None:
        train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)
    else:
        # fallback: simple random split
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    train_path = DATA_DIR / "train_sentiment.csv"
    test_path = DATA_DIR / "test_sentiment.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def prepare_data_for_project(raw_csv_path: str = "data/raw_data.csv") -> dict:
    """
    Full pipeline:
      - ensure folders
      - load CSV
      - clean it
      - save cleaned_posts.csv
      - create train/test split and save
      - aggregate daily timeseries and save timeseries_data.csv (for Prophet)
    Returns a dict with paths to saved files.
    """
    ensure_data_folders()
    raw_path = Path(raw_csv_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"raw data not found at {raw_csv_path}")

    df = pd.read_csv(raw_path)
    cleaned = clean_dataframe(df)

    cleaned_path = DATA_DIR / "cleaned_posts.csv"
    cleaned.to_csv(cleaned_path, index=False)

    # create train/test for sentiment
    train_path, test_path = train_test_split_save(cleaned)

    # create timeseries aggregation (daily max follower per competitor)
    ts = (
        cleaned[["Competitor_handle", "Date", "Followers", "Avg_Engagement_Rate"]]
        .copy()
        .rename(columns={"Date": "ds", "Followers": "follower_count", "Avg_Engagement_Rate": "avg_engagement_rate"})
    )
    ts["ds"] = pd.to_datetime(ts["ds"]).dt.normalize()
    agg = ts.groupby(["Competitor_handle", "ds"], as_index=False).agg(
        {"follower_count": "max", "avg_engagement_rate": "mean"}
    )
    timeseries_path = DATA_DIR / "timeseries_data.csv"
    agg.to_csv(timeseries_path, index=False)

    return {
        "cleaned_path": str(cleaned_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "timeseries_path": str(timeseries_path),
    }


# -------------------------
# CLI entrypoint (so we can run python -m src.load_data)
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare and clean raw social media CSV data for SMCIS")
    parser.add_argument("--raw", type=str, default="data/raw_data.csv", help="Path to raw CSV (default: data/raw_data.csv)")
    args = parser.parse_args()
    print("Preparing data from:", args.raw)
    out = prepare_data_for_project(args.raw)
    print("Saved files:")
    for k, v in out.items():
        print(f" - {k}: {v}")