"""
ZeTheta Project 1C — baseline machine-learning experiment.

Target:
    Whether the next available bar for the same currency pair closes above
    the current bar's Close.

Improvements:
    - Missing next-bar targets are kept as missing and removed before training.
    - Rows are split chronologically by DateTime, including multi-pair datasets.
    - Training rows whose target would cross into the test period are excluded.
    - Required columns and class counts are checked before fitting.
    - Evaluation results include a majority-class baseline for comparison.

This is an educational experiment, not a live trading strategy or financial advice.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "02_Raw_Data" / "forex_price_data.csv"
OUT = HERE
TEST_FRACTION = 0.20
RANDOM_STATE = 42

# ---------- Load and validate data ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find input data at:\n{DATA_PATH}\n"
        "Update DATA_PATH near the top of train_models.py to point to your CSV."
    )

df = pd.read_csv(DATA_PATH)

required = ["DateTime", "Open", "High", "Low", "Close", "Volume"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

if "PairName" not in df.columns:
    # Support single-pair files without a PairName column.
    df["PairName"] = "UNKNOWN_PAIR"

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Spread"]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

df["PairName"] = df["PairName"].astype(str).str.strip().str.upper()
df = df.dropna(subset=["DateTime", "Open", "High", "Low", "Close"])
df = df.sort_values(["PairName", "DateTime"])
df = df.drop_duplicates(subset=["PairName", "DateTime"], keep="last")
df = df.reset_index(drop=True)

if len(df) < 50:
    raise ValueError(
        f"Only {len(df)} usable rows were found. Check the input CSV before training."
    )

# ---------- Create features using current and past observations only ----------
grouped = df.groupby("PairName", sort=False)

df["return_1"] = grouped["Close"].pct_change(1)
df["return_2"] = grouped["Close"].pct_change(2)
df["return_3"] = grouped["Close"].pct_change(3)
df["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
df["body_pct"] = (df["Close"] - df["Open"]) / df["Open"]
df["close_vs_sma10"] = (
    df["Close"]
    / grouped["Close"].transform(
        lambda series: series.rolling(10, min_periods=10).mean()
    )
    - 1
)
df["close_vs_sma20"] = (
    df["Close"]
    / grouped["Close"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    - 1
)
df["volatility_10"] = grouped["Close"].transform(
    lambda series: series.pct_change().rolling(10, min_periods=10).std()
)
df["volume_change"] = grouped["Volume"].pct_change()

# Correct target construction:
# the last row of each pair has no next close, so its target remains NaN.
next_close = grouped["Close"].shift(-1)
next_datetime = grouped["DateTime"].shift(-1)
df["target_up_next_bar"] = np.nan
valid_next = next_close.notna()
df.loc[valid_next, "target_up_next_bar"] = (
    next_close.loc[valid_next] > df.loc[valid_next, "Close"]
).astype(int)
df["target_next_datetime"] = next_datetime

features = [
    "return_1",
    "return_2",
    "return_3",
    "range_pct",
    "body_pct",
    "close_vs_sma10",
    "close_vs_sma20",
    "volatility_10",
    "volume_change",
]
if "Spread" in df.columns:
    features.append("Spread")

# Remove rows with unavailable feature values or unavailable next-bar targets.
data = df.dropna(subset=features + ["target_up_next_bar"]).copy()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=features)
data["target_up_next_bar"] = data["target_up_next_bar"].astype(int)

# ---------- Chronological train/test split ----------
# Split by timestamp, not by arbitrary row order, so observations at the same
# timestamp do not end up on both sides of the split.
unique_times = np.array(sorted(data["DateTime"].dropna().unique()))
if len(unique_times) < 2:
    raise ValueError("Not enough distinct timestamps for a chronological split.")

split_position = int(len(unique_times) * (1 - TEST_FRACTION))
split_position = min(max(split_position, 1), len(unique_times) - 1)
split_time = pd.Timestamp(unique_times[split_position])

train_data = data[data["DateTime"] < split_time].copy()
test_data = data[data["DateTime"] >= split_time].copy()

# Avoid boundary leakage: do not train on a row whose next-bar target occurs
# in the held-out test period.
train_data = train_data[train_data["target_next_datetime"] < split_time].copy()

if train_data.empty or test_data.empty:
    raise ValueError(
        "The chronological split produced an empty train or test set. "
        "Check the dataset size and timestamps."
    )
if train_data["target_up_next_bar"].nunique() < 2:
    raise ValueError(
        "Training data contains only one target class. More data or a different "
        "time range is needed to train a classifier."
    )
if test_data["target_up_next_bar"].nunique() < 2:
    print(
        "WARNING: Test data contains only one target class; ROC-AUC will be undefined."
    )

X_train = train_data[features].replace([np.inf, -np.inf], np.nan)
X_test = test_data[features].replace([np.inf, -np.inf], np.nan)
y_train = train_data["target_up_next_bar"]
y_test = test_data["target_up_next_bar"]

# ---------- Models ----------
models = {
    "logistic_regression": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=4,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

def evaluate_predictions(name, y_true, pred, prob):
    return {
        "Model": name,
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Split_DateTime": str(split_time),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision_Up": precision_score(y_true, pred, zero_division=0),
        "Recall_Up": recall_score(y_true, pred, zero_division=0),
        "F1_Up": f1_score(y_true, pred, zero_division=0),
        "ROC_AUC": (
            roc_auc_score(y_true, prob)
            if pd.Series(y_true).nunique() == 2
            else np.nan
        ),
    }

results = []

# Majority-class baseline: useful for checking whether the trained models add value.
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:, 1]
results.append(
    evaluate_predictions("majority_class_baseline", y_test, baseline_pred, baseline_prob)
)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    results.append(evaluate_predictions(name, y_test, pred, prob))
    joblib.dump(model, OUT / f"{name}_model.joblib")

    # Save predictions for independent inspection.
    prediction_output = test_data[
        ["DateTime", "PairName", "Close", "target_up_next_bar"]
    ].copy()
    prediction_output["predicted_up_next_bar"] = pred
    prediction_output["probability_up"] = prob
    prediction_output.to_csv(OUT / f"{name}_holdout_predictions.csv", index=False)

results_df = pd.DataFrame(results)
results_df.to_csv(OUT / "model_evaluation.csv", index=False)

# Save the feature list and split details so the run is reproducible.
pd.DataFrame({"feature": features}).to_csv(OUT / "model_features.csv", index=False)
run_summary = pd.DataFrame(
    [
        {"Item": "Input_File", "Value": str(DATA_PATH)},
        {"Item": "Usable_Rows", "Value": len(data)},
        {"Item": "Training_Rows", "Value": len(train_data)},
        {"Item": "Testing_Rows", "Value": len(test_data)},
        {"Item": "Split_DateTime", "Value": str(split_time)},
        {"Item": "Training_Up_Count", "Value": int((y_train == 1).sum())},
        {"Item": "Training_Down_Count", "Value": int((y_train == 0).sum())},
        {"Item": "Testing_Up_Count", "Value": int((y_test == 1).sum())},
        {"Item": "Testing_Down_Count", "Value": int((y_test == 0).sum())},
    ]
)
run_summary.to_csv(OUT / "training_run_summary.csv", index=False)

print("\nModel evaluation:")
print(results_df.to_string(index=False))
print(f"\nChronological split timestamp: {split_time}")
print(f"Training rows: {len(train_data)} | Test rows: {len(test_data)}")
print(f"Results saved to: {OUT}")
print("\nNote: this is a baseline research experiment, not financial advice.")
