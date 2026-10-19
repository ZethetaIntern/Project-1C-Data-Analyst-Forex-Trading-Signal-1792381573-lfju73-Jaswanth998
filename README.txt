ZeTheta Project 1C — Reinforcement Learning simulation package

Files:
- q_learning_forex.py: educational tabular Q-learning simulation
- q_learning_summary.csv: summary produced by the script
- q_learning_test_results_<pair>.csv: later-period test results per currency pair

Setup:
1. Save the extracted package contents in 07_Reinforcement_Learning.
2. Keep the original forex_price_data.csv in ../02_Raw_Data.
3. Install Python packages: pandas and numpy.
4. Run from a terminal: python q_learning_forex.py

Important:
- This is a simple learning demonstration, not a production-ready trading agent.
- Results are sensitive to the reward function, state definitions, sample size and transaction-cost assumptions.
- Spread values in the source dataset must be checked for units before treating cost estimates as realistic.
- Do not connect this to a live broker or use it for real-money trading.
- Check the official ZeTheta instructions for the exact expected deliverables.
