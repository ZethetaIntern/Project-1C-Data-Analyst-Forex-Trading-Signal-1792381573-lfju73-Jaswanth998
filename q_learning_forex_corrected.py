"""
ZeTheta Project 1C — simple tabular Q-learning Forex simulation.
This is an educational simulation, not a live trading bot or investment advice.
Uses historical/project price data and a small discrete state/action space.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "02_Raw_Data" / "forex_price_data.csv"
if not DATA_PATH.exists():
    # Allows testing if the raw CSV is copied into this folder.
    DATA_PATH = HERE / "forex_price_data.csv"
OUT = HERE

df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"])
df = df.sort_values(["PairName", "DateTime"]).drop_duplicates().reset_index(drop=True)
results = []
for pair, raw in df.groupby("PairName", sort=False):
    d = raw.copy().reset_index(drop=True)
    d["return_1"] = d["Close"].pct_change()
    d["SMA_10"] = d["Close"].rolling(10, min_periods=10).mean()
    d["momentum"] = d["Close"] / d["SMA_10"] - 1
    d["volatility"] = d["return_1"].rolling(10, min_periods=10).std()
    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(d) < 60:
        print(f"Skipping {pair}: too few usable observations ({len(d)}).")
        continue

    # State features are computed from current/past bars only.
    mom_state = pd.cut(d["momentum"], bins=[-np.inf, -0.002, 0.002, np.inf],
                       labels=[0, 1, 2]).astype(int).to_numpy()
    vol_cut = d["volatility"].median()
    vol_state = (d["volatility"].to_numpy() > vol_cut).astype(int)
    states = mom_state * 2 + vol_state  # six possible states
    actions = [0, 1, 2]  # flat, long, short
    q = np.zeros((6, len(actions)), dtype=float)
    rng = np.random.default_rng(42)

    # Chronological 70/30 train/test split. Train rewards use next-bar returns.
    split = int(len(d) * 0.70)
    train_returns = d["return_1"].to_numpy()
    close = d["Close"].to_numpy()
    spread = pd.to_numeric(d["Spread"], errors="coerce").fillna(0).to_numpy() if "Spread" in d else np.zeros(len(d))
    # Spread units are not documented/verified in the supplied files.
    # Do not silently treat spread as a price-unit value: that produced a
    # cost rate > 1 in the supplied run and forced the policy to stay FLAT.
    # Set SPREAD_IS_PRICE_FRACTION = True only after confirming the CSV's units.
    SPREAD_IS_PRICE_FRACTION = False
    if SPREAD_IS_PRICE_FRACTION:
        cost_rate = np.nanmedian(np.abs(spread / close))
        if not np.isfinite(cost_rate):
            cost_rate = 0.0
        cost_rate = max(float(cost_rate), 0.0)
        cost_note = "Spread cost proxy applied; verify source units."
    else:
        cost_rate = 0.0
        cost_note = "Transaction costs excluded because spread units are unverified."
        print(f"WARNING {pair}: {cost_note}")

    alpha, gamma = 0.10, 0.90
    epsilon = 0.20
    episodes = 30
    for _ in range(episodes):
        for t in range(1, max(split - 1, 1)):
            s = int(states[t])
            if rng.random() < epsilon:
                a = int(rng.integers(0, 3))
            else:
                a = int(np.argmax(q[s]))
            next_s = int(states[t + 1])
            # Reward is next-bar return for long, negative return for short, zero for flat,
            # less a simple spread-based penalty when position changes.
            next_ret = float(train_returns[t + 1]) if np.isfinite(train_returns[t + 1]) else 0.0
            previous_a = int(np.argmax(q[int(states[t-1])]))
            turnover_cost = cost_rate if a != previous_a else 0.0
            reward = (next_ret if a == 1 else (-next_ret if a == 2 else 0.0)) - turnover_cost
            q[s, a] += alpha * (reward + gamma * np.max(q[next_s]) - q[s, a])
        epsilon = max(0.02, epsilon * 0.95)

    # Greedy evaluation on the later 30% test period.
    test_rows = []
    previous_action = 0
    for t in range(split, len(d) - 1):
        state = int(states[t])
        action = int(np.argmax(q[state]))
        next_ret = float(train_returns[t + 1]) if np.isfinite(train_returns[t + 1]) else 0.0
        turnover_cost = cost_rate if action != previous_action else 0.0
        reward = (next_ret if action == 1 else (-next_ret if action == 2 else 0.0)) - turnover_cost
        test_rows.append({
            "PairName": pair, "DateTime": d.loc[t, "DateTime"],
            "Close": float(close[t]), "Momentum_State": int(mom_state[t]),
            "Volatility_State": int(vol_state[t]), "Action_Code": action,
            "Action": {0:"FLAT",1:"LONG",2:"SHORT"}[action],
            "Next_Bar_Return": next_ret, "Estimated_Reward_After_Cost": reward
        })
        previous_action = action

    test = pd.DataFrame(test_rows)
    if test.empty:
        continue
    test["Strategy_Cumulative_Return"] = (1 + test["Estimated_Reward_After_Cost"]).cumprod() - 1
    test["Buy_Hold_Cumulative_Return"] = (1 + test["Next_Bar_Return"]).cumprod() - 1
    test.to_csv(OUT / f"q_learning_test_results_{pair.replace('/','_')}.csv", index=False)
    results.append({
        "PairName": pair, "Usable_Observations": len(d), "Train_Observations": split,
        "Test_Observations": len(test), "Episodes": episodes,
        "Estimated_Cost_Rate": cost_rate,
        "Strategy_Cumulative_Return": float(test["Strategy_Cumulative_Return"].iloc[-1]),
        "Buy_Hold_Cumulative_Return": float(test["Buy_Hold_Cumulative_Return"].iloc[-1]),
        "Mean_Reward_Per_Test_Bar": float(test["Estimated_Reward_After_Cost"].mean()),
        "Unique_Actions_Test": int(test["Action_Code"].nunique()),
        "Note": "Illustrative only; cost units and method require validation."
    })

if not results:
    raise RuntimeError("No pair had enough usable observations to run the simulation.")
pd.DataFrame(results).to_csv(OUT / "q_learning_summary.csv", index=False)
print(pd.DataFrame(results).to_string(index=False))
