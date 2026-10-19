# 10_Python_Code — Forex Trading Signal Analysis

This folder contains a Python script that complements the Power BI Forex Market Analysis Dashboard.

## Files
- `main.py` — loads the forex CSV, validates essential columns, creates moving averages and a simple trend label, exports a summary, and saves charts.
- `requirements.txt` — Python libraries needed.
- `outputs/` — created automatically when you run the script.

## Expected data
The CSV should contain these columns (case-insensitive):
`DateTime`, `Open`, `High`, `Low`, `Close`, `Volume`

Optional columns: `PairName`, `PairID`, `Spread`.

## Setup on Windows
1. Install Python 3.10 or newer from the official Python website.
2. Open Command Prompt in this `10_Python_Code` folder.
3. Install dependencies:

   ```bash
   py -m pip install -r requirements.txt
   ```

4. Run the script, replacing the sample path with the actual CSV path:

   ```bash
   py main.py --input "E:\Data Analyst\ZeTheta Projects\Project 1C Data Analyst Forex Trading Signal\03_Cleaned_Data\forex_price_data_cleaned.csv"
   ```

If the CSV has a different filename, use its exact filename. You can also specify an output folder:

```bash
py main.py --input "FULL_PATH_TO_YOUR_CSV.csv" --output "outputs"
```

## Outputs
The script creates:
- `forex_analysis_output.csv`
- `summary_report.csv`
- `closing_price_moving_averages.png`
- `high_low_prices.png`
- `trading_volume.png`
- `return_distribution.png`

## Important notes
- The script assumes each row is one observation and calculates MA_5 as the previous five rows, unlike a five-day time window.
- `Market_Trend` is a simple Close-versus-MA_5 indicator, not a trading recommendation.
- The next-period direction column is an exploratory label only; no predictive performance is claimed.
- Do not use the output as financial advice or for live trading without independent validation.
