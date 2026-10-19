ZeTheta Project 1C — RL correction note

The supplied RL output showed Estimated_Cost_Rate = 1.301926, Strategy_Cumulative_Return = 0.0,
Buy_Hold_Cumulative_Return = 0.010543, and Unique_Actions_Test = 1. The policy selected only one
action (apparently FLAT), so the original result does not demonstrate a functioning strategy.

q_learning_forex_corrected.py avoids applying a spread-based transaction cost unless the source spread
units have been confirmed. Costs are excluded by default. This is not a claim that the strategy is
profitable. Re-run with the original forex_price_data.csv in ../02_Raw_Data, then inspect the fresh
q_learning_summary.csv and test results. If the source spread is in price units, review the setting
SPREAD_IS_PRICE_FRACTION before enabling costs.
