# Importing the necessary packages
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Function to retrieve data for any asset in any time period
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, multi_level_index=False)
    data = data[['Open', 'Close']]
    return data

# Function to calculate a strategy's daily returns
def strategy_returns(positions, open_prices, close_prices, cost=0.0):
    current_position = positions.shift(1).fillna(0)                                               # today's position is based on yesterday's close
    previous_position = current_position.shift(1).fillna(0)                                       # what you held before today's market open
    trade_size = (current_position - previous_position).abs()                                     # how much you bought or sold at today's open
    intraday_returns = (close_prices - open_prices) / open_prices                                 # return from today's open to today's close
    overnight_returns = (open_prices / close_prices.shift(1) - 1).fillna(0)                       # return from yesterday's close to today's open
    gross_returns = (1 + previous_position * overnight_returns) * (1 + current_position * intraday_returns) - 1     # overnight and intraday returns compound
    trading_costs = cost * trade_size                                                             # cost paid on the amount traded
    net_returns = gross_returns - trading_costs
    return net_returns
    
# Function to calculate cumulative % growth
def cumulative_return(returns):
    daily_growth = 1 + returns                         # e.g. a +2% day becomes 1.02
    cumulative_growth = daily_growth.cumprod()         # value of $1 invested at the start
    percent_return = 100 * (cumulative_growth - 1)     # convert to % gain
    return percent_return

# Function to calculate annualized return
def annualized_return(returns):
    total_growth = (1 + returns).prod()                # final value of $1
    years = (len(returns)) / 252                   # 252 trading days in a year
    ann_return = total_growth ** (1 / years) - 1
    return ann_return

# Function to calculate annualized volatility
def annualized_volatility(returns):
    daily_volatility = returns.std()
    ann_vol = daily_volatility * np.sqrt(252)
    return ann_vol

# Function to calculate the Sharpe ratio
def sharpe_ratio(returns):
    annual_mean = returns.mean() * 252
    annual_volatility = annualized_volatility(returns)
    if annual_volatility == 0:                            # no movement in the strategy
        return 0.0
    sharpe = annual_mean / annual_volatility
    return sharpe

# Function to calculate maximum drawdown
def max_drawdown(returns):
    cumulative_growth = (1 + returns).cumprod()
    running_peak = cumulative_growth.cummax()             # highest value reached up to each day
    drawdown = cumulative_growth / running_peak - 1       # % below that peak
    worst_drawdown = drawdown.min()                       # most negative = worst drawdown
    return worst_drawdown

# Function to calculate all performance metrics at once
def performance_metrics(returns):
    ann_return = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    sharpe = sharpe_ratio(returns)
    worst_drawdown = max_drawdown(returns)
    return ann_return, ann_vol, sharpe, worst_drawdown

# Function to summarize all metrics for one strategy
def summary(returns):
    ann_return, ann_vol, sharpe, worst_drawdown = performance_metrics(returns)
    return {'Ann. Return': ann_return, 
            'Ann. Volatility': ann_vol, 
            'Sharpe': sharpe, 
            'Max Drawdown': worst_drawdown}

# Function to generate random positions of -1 or +1 with equal probability
def random_long_short_positions(index, rng):
    positions = rng.choice([-1.0, 1.0], size=len(index))
    long_short_positions = pd.Series(positions, index=index)
    return long_short_positions

# Function to generate random positions in [-1, 1] with density f(x) = 3x^2/2
def random_continuous_positions(index, rng):
    y = rng.uniform(0, 1, size=len(index))
    positions = np.cbrt(2*y - 1)              # inverse CDF of f(x) = 3x^2/2 (derived in report)
    continuous_positions = pd.Series(positions, index=index)
    return continuous_positions
    
# Function to run every random strategy through the backtest (and keep a sample for visualization)
def run_random_grid(ticker, ticker_data, n_strategies, seed, sample_every=40):
    rng = np.random.default_rng(seed)                 # same seed gives the same strategies every run
    results = {}
    sample_returns = {}
    for k in range(n_strategies):
        positions = random_long_short_positions(ticker_data.index, rng)
        returns = strategy_returns(positions, ticker_data['Open'], ticker_data['Close'])
        results[f'{ticker}_Long_Short_{k}'] = summary(returns)
        if k % sample_every == 0:
            sample_returns[f'{ticker}_Long_Short_{k}'] = returns
    for k in range(n_strategies):
        positions = random_continuous_positions(ticker_data.index, rng)
        returns = strategy_returns(positions, ticker_data['Open'], ticker_data['Close'])
        results[f'{ticker}_Continuous_{k}'] = summary(returns)
        if k % sample_every == 0:
            sample_returns[f'{ticker}_Continuous_{k}'] = returns
    results_df = pd.DataFrame(results).T
    return results_df, sample_returns
