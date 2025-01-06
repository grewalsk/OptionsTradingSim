# DataFetcher.py

import requests
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def fetch_intraday_stock_data(symbol, interval="5min", outputsize="compact"):
    """
    Fetches intraday stock data from Alpha Vantage API.
    
    Parameters:
        symbol (str): Stock ticker symbol (e.g., 'AAPL').
        interval (str): Time interval between data points (e.g., '5min').
        outputsize (str): 'compact' for the latest 100 data points or 'full' for the full-length time series.
    
    Returns:
        dict or None: JSON data as a dictionary if successful, else None.
    """
    API_KEY = st.secrets["alphavantage"]["api_key"]  # Ensure this is set in secrets.toml
    API_URL = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": outputsize
    }
    try:
        response = requests.get(API_URL, params=params)
        data = response.json()
        time_series_key = f"Time Series ({interval})"
        if time_series_key in data:
            return data[time_series_key]
        else:
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None
