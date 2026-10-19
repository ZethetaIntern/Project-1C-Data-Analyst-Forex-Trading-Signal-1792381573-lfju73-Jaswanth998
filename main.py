"""
Forex Trading Signal Analysis - Python project
Expected input columns (case-insensitive): DateTime, Open, High, Low, Close, Volume
Optional: PairName, PairID, Spread

Usage:
  python main.py --input "E:/Data Analyst/ZeTheta Projects/Project 1C Data Analyst Forex Trading Signal/03_Cleaned_Data/forex_price_data_cleaned.csv"
  python main.py --input "path/to/data.csv" --output "outputs"

If --input is omitted, the script searches nearby project folders for a CSV file.
This script is for learning and analysis only, not financial advice.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


COLUMN_ALIASES = {
    "datetime": ["datetime", "date_time", "timestamp", "date", "time"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close", "closing_price", "price"],
    "volume": ["volume", "tick_volume", "trading_volume"],
    "pairname": ["pairname", "pair_name", "currency_pair", "pair"],
    "pairid": ["pairid", "pair_id"],
    "spread": ["spread"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common column variants to consistent names."""
    rename = {}
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_original:
                rename[lower_to_original[alias]] = standard.title() if standard != "pairname" else "PairName"
                break
    return df.rename(columns=rename)


def find_input_file(project_root: Path) -> Path:
    """Search likely project data folders for a CSV file."""
    search_roots = [
        project_root / "03_Cleaned_Data",
        project_root / "02_Raw_Data",
        project_root,
    ]
    candidates = []
    for root in search_roots:
        if root.exists():
            candidates.extend(p for p in root.rglob("*.csv") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            "No CSV file found. Pass --input with the full path to your forex CSV file."
        )
    # Prefer a filename that looks like the cleaned dataset.
    candidates.sort(
        key=lambda p: (
            "clean" not in p.name.lower(),
            "forex" not in p.name.lower(),
            len(str(p)),
        )
    )
    return candidates[0]


def load_and_clean(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = standardize_columns(df)

    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume", "Spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    df = df.sort_values("Datetime")
    if "PairName" in df.columns:
        df["PairName"] = df["PairName"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["Datetime", "PairName"] if "PairName" in df.columns else ["Datetime"])
    df = df.reset_index(drop=True)

    # Basic quality checks; rows are retained unless essential fields are missing.
    df["Price_Range"] = df["High"] - df["Low"]
    df["Daily_or_Row_Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["Price_Change"] = df["Close"].diff()
    df["Volatility_5"] = df["Daily_or_Row_Return"].rolling(window=5, min_periods=2).std()

    # Simple descriptive signal based on Close vs. MA_5.
    df["Market_Trend"] = "Neutral"
    df.loc[df["Close"] > df["MA_5"], "Market_Trend"] = "Uptrend"
    df.loc[df["Close"] < df["MA_5"], "Market_Trend"] = "Downtrend"

    # Next-observation direction label, useful for exploratory ML only.
    next_close = df["Close"].shift(-1)
    df["Next_Period_Direction"] = (next_close > df["Close"]).astype("float")
    df.loc[next_close.isna(), "Next_Period_Direction"] = pd.NA
    return df


def make_outputs(df: pd.DataFrame, output_dir: Path, source_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "forex_analysis_output.csv"
    df.to_csv(cleaned_path, index=False)

    # Summary report
    summary = {
        "source_file": str(source_path),
        "rows_after_cleaning": int(len(df)),
        "start_datetime": str(df["Datetime"].min()),
        "end_datetime": str(df["Datetime"].max()),
        "average_close": float(df["Close"].mean()),
        "highest_price": float(df["High"].max()),
        "lowest_price": float(df["Low"].min()),
        "average_volume": float(df["Volume"].mean()),
        "latest_close": float(df["Close"].iloc[-1]),
        "latest_ma_5": float(df["MA_5"].iloc[-1]),
        "latest_market_trend": str(df["Market_Trend"].iloc[-1]),
    }
    pd.Series(summary, name="value").to_csv(output_dir / "summary_report.csv", header=True)

    # Plot 1: close and moving averages
    plt.figure(figsize=(12, 6))
    plt.plot(df["Datetime"], df["Close"], label="Close", linewidth=1.2)
    plt.plot(df["Datetime"], df["MA_5"], label="MA_5", linewidth=1.8)
    plt.plot(df["Datetime"], df["MA_20"], label="MA_20", linewidth=1.8)
    plt.title("EUR/USD Closing Price and Moving Averages")
    plt.xlabel("DateTime")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "closing_price_moving_averages.png", dpi=150)
    plt.close()

    # Plot 2: high and low
    plt.figure(figsize=(12, 6))
    plt.plot(df["Datetime"], df["High"], label="High", linewidth=1)
    plt.plot(df["Datetime"], df["Low"], label="Low", linewidth=1)
    plt.title("High and Low Prices Over Time")
    plt.xlabel("DateTime")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "high_low_prices.png", dpi=150)
    plt.close()

    # Plot 3: volume
    plt.figure(figsize=(12, 5))
    plt.plot(df["Datetime"], df["Volume"], linewidth=0.8)
    plt.title("Trading Volume Over Time")
    plt.xlabel("DateTime")
    plt.ylabel("Volume")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "trading_volume.png", dpi=150)
    plt.close()

    # Plot 4: distribution of returns
    returns = df["Daily_or_Row_Return"].dropna()
    if not returns.empty:
        plt.figure(figsize=(10, 5))
        plt.hist(returns, bins=40)
        plt.title("Distribution of Period Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / "return_distribution.png", dpi=150)
        plt.close()

    print("\nAnalysis completed.")
    print(f"Input file: {source_path}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"Average Close: {df['Close'].mean():.5f}")
    print(f"Highest High: {df['High'].max():.5f}")
    print(f"Lowest Low: {df['Low'].min():.5f}")
    print(f"Average Volume: {df['Volume'].mean():.2f}")
    print(f"Latest Market Trend (Close vs MA_5): {df['Market_Trend'].iloc[-1]}")
    print(f"Outputs saved to: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forex market analysis and signal exploration.")
    parser.add_argument("--input", type=str, default=None, help="Full path to the forex CSV file.")
    parser.add_argument("--output", type=str, default="outputs", help="Output folder (default: outputs).")
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root used to search for CSVs when --input is omitted.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser()
    input_path = Path(args.input).expanduser() if args.input else find_input_file(project_root)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_dir = Path(args.output).expanduser()
    df = load_and_clean(input_path)
    make_outputs(df, output_dir, input_path)


if __name__ == "__main__":
    main()
