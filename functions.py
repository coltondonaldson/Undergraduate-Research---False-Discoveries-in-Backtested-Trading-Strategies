
# Function to retrieve data for any asset in any time period
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, multi_level_index=False)
    data = data[['Open', 'Close']]
    return data

# Function to calculate the close-to-close returns for a ticker
def daily_returns(ticker_data):
    current = ticker_data['Close']
    previous = current.shift(1)                  # yesterday's close lined up with today's
    returns = (current - previous) / previous    # % change from yesterday to today
    returns = returns.fillna(0)                  # first day has no previous close
    return returns

# Function to calculate a strategy's daily returns
def strategy_returns(positions, asset_returns, cost=0.0):
    position = positions.shift(1).fillna(0)              # today you hold what you decided at yesterday's close
    previous_position = position.shift(1).fillna(0)      # what you held the day before
    trade_size = (position - previous_position).abs()    # how much you bought or sold to get to today's position
    gross_returns = position * asset_returns             # return earned on the position held
    trading_costs = cost * trade_size                    # cost paid on the amount traded
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
    years = (len(returns) - 1) / 252                   # 252 trading days in a year
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
    return {'Ann. Return': annualized_return(returns), 
            'Ann. Volatility': annualized_volatility(returns), 
            'Sharpe': sharpe_ratio(returns), 
            'Max Drawdown': max_drawdown(returns)}
