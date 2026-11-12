# False Discoveries in Backtested Trading Strategies

Undergraduate research project at Queen's University, supervised by Professor Mehmet Beceren.

This project studies how often trading strategies appear profitable in backtests by chance alone, and how well multiple testing corrections separate real discoveries from false ones.

## Notebooks
1. **Data Collection**: downloads daily price data and defines the in-sample (2006–2020) and out-of-sample (2021–2025) periods.
2. **Strategy Grid**: builds the backtesting framework and runs 10,000 random strategies with no skill as a control group.

Further notebooks covering significance testing, real strategy families, and out-of-sample evaluation are in progress.

Reusable functions are in `notebooks/functions.py`.