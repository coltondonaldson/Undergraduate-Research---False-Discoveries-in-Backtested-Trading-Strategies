# Importing the necessary packages
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Defining a function to retrieve data for any given time period
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True, multi_level_index=False)
    data = data[['Open', 'Close']]
    return data
