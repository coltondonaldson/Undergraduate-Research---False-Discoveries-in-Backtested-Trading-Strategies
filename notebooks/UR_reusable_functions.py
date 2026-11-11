import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

def get_spy_data():
    spy_data = yf.download("SPY", start="2009-01-01", end="2020-01-01", interval="1d", auto_adjust=True)
    return spy_data

def get_variables(dataframe):
    opening_prices = dataframe[('Open', 'SPY')]
    closing_prices = dataframe[('Close', 'SPY')]
    dates = dataframe.index
    model_start = dates.get_loc(pd.Timestamp("2010-01-04"))
    return opening_prices, closing_prices, dates, model_start

def compute_strategy_balance(dataframe, strategy, starting_balance):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    strategy_balance = []
    strategy_balance.append(starting_balance)
    for day in range(model_start, len(opening_prices) - 1):
        position_size = strategy.shift(1).iloc[day]
        current_balance = strategy_balance[-1] * (1 + position_size * opening_prices.pct_change().iloc[day + 1])
        strategy_balance.append(current_balance)
    final_day_pct_change = (closing_prices.iloc[-1] - opening_prices.iloc[-1]) / opening_prices.iloc[-1]
    position_size = strategy.shift(1).iloc[-1]
    current_balance = strategy_balance[-1] * (1 + position_size * final_day_pct_change)
    strategy_balance.append(current_balance)
    return strategy_balance

def performance_metrics(dataframe, strategy_signal):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    strategy_balance = compute_strategy_balance(dataframe, strategy_signal, 1)
    daily_returns = []
    for day in range(len(strategy_balance)-1):
        daily_return = (strategy_balance[day+1] - strategy_balance[day]) / strategy_balance[day]
        daily_returns.append(daily_return)
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    volatility = np.std(daily_returns) * np.sqrt(252)
    drawdowns = []
    max_balance = strategy_balance[0]
    for day in range(len(strategy_balance)-1):
        if strategy_balance[day+1] > max_balance:
            max_balance = strategy_balance[day+1]
        drawdowns.append((strategy_balance[day+1] - max_balance) / max_balance)
    max_drawdown = min(drawdowns)
    cumulative_return = strategy_balance[-1] / strategy_balance[0] - 1
    annualized_return = (1 + cumulative_return)**(252 / (len(strategy_balance)-1)) - 1
    performance_metrics = {'Sharpe Ratio': sharpe_ratio, 
                           'Max Drawdown': max_drawdown, 
                           'Annualized Return': annualized_return, 
                           'Volatility': volatility}
    return performance_metrics

def momentum_condition(dataframe, lookback_period):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    momentum_lookback_period = lookback_period
    momentum_lookback_pct_change = closing_prices.pct_change(momentum_lookback_period)
    return momentum_lookback_pct_change

def momentum_signal(dataframe, lookback_period):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    momentum_lookback_pct_change = momentum_condition(dataframe, lookback_period)
    mom_signal = []
    for day in range(len(dates)):
        if momentum_lookback_pct_change.iloc[day] > 0:
            mom_signal.append(1)
        else:
            mom_signal.append(0)
    return mom_signal

def mean_reversion_variables(dataframe, lookback_period, entry_band_width):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    mr_lookback_period = lookback_period
    mr_lookback_mean = []
    mr_lookback_std = []
    for day in range(mr_lookback_period):
        mr_lookback_mean.append(0)
        mr_lookback_std.append(0)
    for day in range(mr_lookback_period, len(closing_prices)):
        mr_lookback_mean.append(closing_prices.iloc[day - mr_lookback_period : day].mean())
        mr_lookback_std.append(closing_prices.iloc[day - mr_lookback_period : day].std())
    mr_entry_band = []
    for day in range(len(mr_lookback_mean)):
        mr_entry_band.append(mr_lookback_mean[day] - entry_band_width*mr_lookback_std[day])
    return mr_lookback_mean, mr_entry_band

def mean_reversion_signal(dataframe, lookback_period, entry_band_width):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    mr_lookback_mean, mr_entry_band = mean_reversion_variables(dataframe, lookback_period, entry_band_width)
    mr_signal = []
    signal_active = False
    for day in range(len(mr_lookback_mean)):
        if closing_prices.iloc[day] > mr_lookback_mean[day]:
            signal_active = False
        elif closing_prices.iloc[day] < mr_entry_band[day]:
            signal_active = True
        if signal_active is False:
            mr_signal.append(0)
        elif signal_active is True:
            mr_signal.append(1)
    return mr_signal

def table_formatter(model):
    formatted_data = model.copy()
    decimal_rows = ['Sharpe Ratio']
    percent_rows = ['Max Drawdown', 'Annualized Return', 'Volatility']
    for row in decimal_rows:
        formatted_data[row] = f"{formatted_data[row]:.3f}"
    for row in percent_rows:
        formatted_data[row] = f"{formatted_data[row]*100:.2f}%"
    return formatted_data

def market_volatility(dataframe, window):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    market_balance = []
    day_1_return = (closing_prices.iloc[0] - opening_prices.iloc[0]) / opening_prices.iloc[0] + 1
    market_balance.append(day_1_return)
    for day in range(len(closing_prices) - 1):
        current_balance = market_balance[-1] * ((closing_prices.iloc[day + 1] - closing_prices.iloc[day]) / closing_prices.iloc[day] + 1)
        market_balance.append(current_balance)
    daily_returns = []
    for day in range(len(market_balance)-1):
        daily_return = (market_balance[day+1] - market_balance[day]) / market_balance[day]
        daily_returns.append(daily_return)
    annualized_volatility = []
    for day in range(len(closing_prices)-model_start):
        window_returns = daily_returns[day+model_start-window: day+model_start]
        rolling_volatility = np.std(window_returns)
        annualized_volatility.append(rolling_volatility * np.sqrt(252))
    return annualized_volatility

def market_EWMA(dataframe, Lambda):
    opening_prices, closing_prices, dates, model_start = get_variables(dataframe)
    market_balance = []
    day_1_return = (closing_prices.iloc[0] - opening_prices.iloc[0]) / opening_prices.iloc[0] + 1
    market_balance.append(day_1_return)
    for day in range(1, len(closing_prices)):
        current_balance = market_balance[-1] * ((closing_prices.iloc[day] - closing_prices.iloc[day-1]) / closing_prices.iloc[day-1] + 1)
        market_balance.append(current_balance)
    daily_returns = []
    for day in range(1, len(market_balance)):
        daily_return = (market_balance[day] - market_balance[day-1]) / market_balance[day-1]
        daily_returns.append(daily_return)
    half_year_window = 126
    initial_returns = daily_returns[:half_year_window - 1]
    initial_variance = np.var(initial_returns)
    ewma_var = []
    ewma_var.append(initial_variance)
    for day in range(half_year_window - 1, len(daily_returns)):
        daily_return = daily_returns[day]
        daily_var = Lambda * ewma_var[-1] + (1 - Lambda) * daily_return**2
        ewma_var.append(daily_var)
    ewma_vol = np.sqrt(ewma_var)
    ewma_z_score = []
    for day in range(len(ewma_vol) - model_start + half_year_window - 1):
        rolling_mean = np.mean(ewma_vol[day: day + half_year_window])
        rolling_std = np.std(ewma_vol[day: day + half_year_window])
        z_score = (ewma_vol[day + model_start - half_year_window + 1] - rolling_mean) / rolling_std
        ewma_z_score.append(z_score)
    ewma_vol = ewma_vol * np.sqrt(252)
    ewma_vol = ewma_vol[model_start - half_year_window + 1:]    
    return ewma_vol, ewma_z_score
    