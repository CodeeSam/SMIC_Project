"""
src/model_sentiment.py

Responsibilities:
- Load cleaned train/test CSV files
- Perform zero-shot sentiment classification on Post_text
- Save sentiment predictions back to CSV
- Works with DistilBERT (no training required)
"""

import pandas as pd
from pathlib import Path
from transformers import pipeline

# Paths
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "train_sentiment.csv"
TEST_CSV = DATA_DIR / "test_sentiment.csv"

# Output paths
PRED_TRAIN_CSV = DATA_DIR / "pred_train_sentiment.csv"
PRED_TEST_CSV = DATA_DIR / "pred_test_sentiment.csv"

# Define candidate labels for zero-shot
CANDIDATE_LABELS = ["positive", "neutral", "negative"]

def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna({"Post_text": ""})
    return df

def predict_sentiment(df, classifier):
    """
    Runs zero-shot classification on the 'Post_text' column
    Returns a DataFrame with added columns:
        - sentiment_label
        - sentiment_score (confidence of the predicted label)
    """
    sentiments = []
    scores = []

    for text in df["Post_text"]:
        if not isinstance(text, str) or text.strip() == "":
            # Default for empty post
            sentiments.append("neutral")
            scores.append(1.0)
            continue

        result = classifier(text, candidate_labels=CANDIDATE_LABELS)
        # pick the label with the highest score
        label = result["labels"][0]
        score = result["scores"][0]
        sentiments.append(label)
        scores.append(score)

    df["sentiment_label"] = sentiments
    df["sentiment_score"] = scores
    return df

def main():
    print("Loading zero-shot sentiment classifier...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Process train set
    print("Processing train_sentiment.csv...")
    train_df = load_data(TRAIN_CSV)
    train_pred = predict_sentiment(train_df, classifier)
    train_pred.to_csv(PRED_TRAIN_CSV, index=False)
    print(f"Saved predicted train sentiment to: {PRED_TRAIN_CSV}")

    # Process test set
    print("Processing test_sentiment.csv...")
    test_df = load_data(TEST_CSV)
    test_pred = predict_sentiment(test_df, classifier)
    test_pred.to_csv(PRED_TEST_CSV, index=False)
    print(f"Saved predicted test sentiment to: {PRED_TEST_CSV}")

    print("✅ Sentiment prediction complete.")

if __name__ == "__main__":
    main()