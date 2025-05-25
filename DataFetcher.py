# OptionTrading/DataFetcher.py

import requests
import streamlit as st

API_URL = "https://www.alphavantage.co/query"

# Configuration for the API call
INTERVAL = "5min"       # Interval between data points (1min, 5min, 15min, 30min, 60min)
ADJUSTED = "true"       # Adjust for splits and dividends
EXTENDED_HOURS = "true" # Include pre/post-market data
OUTPUT_SIZE = "compact" # Data output size (compact or full)
DATA_TYPE = "json"      # Data format (json or csv)

@st.cache_data
def fetch_intraday_stock_data(symbol):
    """
    Fetches intraday stock data for a given symbol from the Alpha Vantage API.

    Parameters:
        symbol (str): The stock symbol to fetch data for (e.g., 'AAPL').

    Returns:
        dict: A dictionary containing the intraday time series data.
    """
    api_key = st.secrets["alphavantage"]["api_key"]

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": INTERVAL,
        "apikey": api_key,
        "adjusted": ADJUSTED,
        "extended_hours": EXTENDED_HOURS,
        "outputsize": OUTPUT_SIZE,
        "datatype": DATA_TYPE
    }
    response = requests.get(API_URL, params=params)
    data = response.json()

    # Log the API request and response (optional)
    st.write(f"**API Request URL:** {response.url}\n")
    st.write(f"**API Response for {symbol}:**\n{data}\n")

    # Handle API rate limits and errors
    if "Note" in data:
        st.warning("API rate limit reached. Please wait a minute before making new requests.")
        return {}
    elif "Error Message" in data:
        st.error(f"Error fetching data for symbol {symbol}: {data['Error Message']}")
        return {}

    time_series_key = f"Time Series ({INTERVAL})"
    if time_series_key not in data:
        st.error(f"Unexpected data format received from API for symbol {symbol}.")
        return {}

    return data[time_series_key]
