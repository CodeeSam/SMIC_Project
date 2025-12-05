"""
src/model_timeseries.py

Responsibilities:
- Load preprocessed time series CSV (timeseries_data.csv)
- Train Prophet model for each competitor
- Include avg_engagement_rate as an extra regressor
- Save 90-day forecasts for all competitors to forecast_results.csv
"""

import pandas as pd
from pathlib import Path
from prophet import Prophet

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TS_CSV = DATA_DIR / "timeseries_data.csv"
FORECAST_CSV = DATA_DIR / "forecast_results.csv"

FORECAST_DAYS = 90  # number of days to forecast

def load_data(path):
    df = pd.read_csv(path)
    # Rename columns to match Prophet's expectations
    df = df.rename(columns={
        "Date": "ds",
        "follower_count": "y_followers",  # fix typo here
        "Avg_Engagement_Rate": "avg_engagement_rate"
    })
    df["ds"] = pd.to_datetime(df["ds"])
    df["y_followers"] = pd.to_numeric(df["y_followers"], errors="coerce")
    df["avg_engagement_rate"] = pd.to_numeric(df["avg_engagement_rate"], errors="coerce")
    return df


def train_forecast(df):
    results = []

    competitors = df["Competitor_handle"].unique()

    for comp in competitors:
        print(f"Training Prophet model for: {comp}")
        comp_df = df[df["Competitor_handle"] == comp][["ds", "y_followers", "avg_engagement_rate"]].copy()

        # Initialize Prophet model
        m = Prophet(daily_seasonality=True, yearly_seasonality=True)
        m.add_regressor("avg_engagement_rate")

        # Fit model
        m.fit(comp_df.rename(columns={"ds": "ds", "y_followers": "y"}))

        # Create future dataframe
        future = m.make_future_dataframe(periods=FORECAST_DAYS)
        # Use last available engagement rate for future days
        last_eng_rate = comp_df["avg_engagement_rate"].iloc[-1]
        future["avg_engagement_rate"] = last_eng_rate

        # Make forecast
        forecast = m.predict(future)
        forecast["Competitor_handle"] = comp

        # Keep relevant columns
        forecast_subset = forecast[["ds", "Competitor_handle", "yhat", "yhat_lower", "yhat_upper"]]
        results.append(forecast_subset)

    # Combine all forecasts
    all_forecasts = pd.concat(results)
    all_forecasts.to_csv(FORECAST_CSV, index=False)
    print(f"Saved all forecasts to: {FORECAST_CSV}")

def main():
    df = load_data(TS_CSV)
    train_forecast(df)
    print("✅ Time series forecasting complete.")

if __name__ == "__main__":
    main()
