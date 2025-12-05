"""
src/evaluate_sentiment.py

Responsibilities:
- Load predicted sentiment CSV (`pred_test_sentiment.csv`)
- Compare predictions against ground truth (`target_label`)
- Compute and display key evaluation metrics
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PRED_TEST_CSV = DATA_DIR / "pred_test_sentiment.csv"

# Map your target_label to the same labels as zero-shot output
# Assuming your target_label uses 0=negative, 1=neutral, 2=positive
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_data(path):
    df = pd.read_csv(path)
    # Ensure target_label is mapped to string labels
    df["target_label_str"] = df["target_label"].map(LABEL_MAP)
    return df

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutral", "negative"])
    report = classification_report(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1, cm, report

def main():
    df = load_data(PRED_TEST_CSV)
    y_true = df["target_label_str"]
    y_pred = df["sentiment_label"]

    acc, prec, rec, f1, cm, report = compute_metrics(y_true, y_pred)

    print("✅ Sentiment Classification Evaluation Metrics\n")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Detailed Classification Report:")
    print(report)

if __name__ == "__main__":
    main()