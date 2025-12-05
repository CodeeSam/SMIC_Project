# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from transformers import pipeline

# -----------------------
# Paths
# -----------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TS_CSV = DATA_DIR / "timeseries_data.csv"
FORECAST_CSV = DATA_DIR / "forecast_results.csv"
SENTIMENT_CSV = DATA_DIR / "pred_test_sentiment.csv"

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_timeseries():
    df = pd.read_csv(TS_CSV)
    df = df.rename(columns={"Date":"ds", "follower_count":"y_followers", "Avg_Engagement_Rate":"avg_engagement_rate"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df

@st.cache_data
def load_forecast():
    df = pd.read_csv(FORECAST_CSV)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

@st.cache_data
def load_sentiment():
    df = pd.read_csv(SENTIMENT_CSV)
    return df

timeseries_df = load_timeseries()
forecast_df = load_forecast()
sentiment_df = load_sentiment()

# -----------------------
# Streamlit App Layout
# -----------------------
st.set_page_config(page_title="SMCIS Dashboard", layout="wide")
st.title("Social Media Competitor Intelligence System (SMCIS)")

# Competitor Selection
competitors = timeseries_df["Competitor_handle"].unique()
selected_comp = st.selectbox("Select Competitor", competitors)

# Date Range Selection
min_date = timeseries_df["ds"].min()
max_date = timeseries_df["ds"].max()
date_range = st.date_input("Select Date Range", [min_date, max_date])

# Filtered Data
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
ts_filtered = timeseries_df[(timeseries_df["Competitor_handle"]==selected_comp) &
                            (timeseries_df["ds"] >= start_date) & (timeseries_df["ds"] <= end_date)]

forecast_filtered = forecast_df[(forecast_df["Competitor_handle"]==selected_comp) &
                                (forecast_df["ds"] >= start_date) & (forecast_df["ds"] <= end_date)]

sentiment_filtered = sentiment_df[sentiment_df["Competitor_handle"]==selected_comp]

# -----------------------
# Follower Trend Plot
# -----------------------
st.subheader("Follower Count (Historical + Forecast)")
fig = px.line()
fig.add_scatter(x=ts_filtered["ds"], y=ts_filtered["y_followers"], mode="lines+markers", name="Historical")
fig.add_scatter(x=forecast_filtered["ds"], y=forecast_filtered["yhat"], mode="lines+markers", name="Forecast")
fig.add_scatter(x=forecast_filtered["ds"], y=forecast_filtered["yhat_upper"], mode="lines", line=dict(dash="dash"), name="Upper Bound")
fig.add_scatter(x=forecast_filtered["ds"], y=forecast_filtered["yhat_lower"], mode="lines", line=dict(dash="dash"), name="Lower Bound")
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Sentiment Analysis Plot
# -----------------------
st.subheader("Sentiment Analysis")
sentiment_counts = sentiment_filtered["sentiment_label"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Count"]
fig2 = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment", text="Count")
st.plotly_chart(fig2, use_container_width=True)

# Show posts flagged as anomalies (negative sentiment)
st.subheader("Strategic Anomaly Posts (Negative Sentiment)")
anomaly_posts = sentiment_filtered[sentiment_filtered["sentiment_label"]=="negative"][["Date","Post_text","sentiment_label"]]
st.dataframe(anomaly_posts)

# -----------------------
# Real-Time Sentiment Prediction
# -----------------------
st.subheader("Real-Time Sentiment Prediction")
st.write("Enter a social media post and get sentiment prediction:")

user_post = st.text_area("Post Text")
if st.button("Predict Sentiment"):
    if user_post.strip() != "":
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["positive", "neutral", "negative"]
        result = classifier(user_post, candidate_labels)
        st.write(f"Predicted Sentiment: **{result['labels'][0]}** with confidence {result['scores'][0]:.2f}")
    else:
        st.warning("Please enter some text!")

# -----------------------
# Download Buttons
# -----------------------
st.subheader("Download Data")
st.download_button("Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast_results.csv")
st.download_button("Download Sentiment CSV", sentiment_df.to_csv(index=False), file_name="pred_test_sentiment.csv")