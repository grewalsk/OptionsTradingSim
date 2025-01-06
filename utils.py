# utils.py

import pandas as pd
import streamlit as st

def initialize_session_state():
    if 'stock_data_eu' not in st.session_state:
        st.session_state['stock_data_eu'] = pd.DataFrame(columns=['timestamp', 'close_price'])
    if 'stock_data_am' not in st.session_state:
        st.session_state['stock_data_am'] = pd.DataFrame(columns=['timestamp', 'close_price'])
    # Initialize other session_state variables as needed
    if 'volatility_eu' not in st.session_state:
        st.session_state['volatility_eu'] = 0.0
    if 'latest_close_price_eu' not in st.session_state:
        st.session_state['latest_close_price_eu'] = 100.0  # Default value
    if 'volatility_am' not in st.session_state:
        st.session_state['volatility_am'] = 0.0
    if 'latest_close_price_am' not in st.session_state:
        st.session_state['latest_close_price_am'] = 100.0  # Default value
    if 'model_nn' not in st.session_state:
        st.session_state['model_nn'] = None
    if 'history_nn' not in st.session_state:
        st.session_state['history_nn'] = {}
    if 'stock_data_cmp' not in st.session_state:
        st.session_state['stock_data_cmp'] = pd.DataFrame(columns=['timestamp', 'close_price'])
    if 'volatility_cmp' not in st.session_state:
        st.session_state['volatility_cmp'] = 0.0
    if 'latest_close_price_cmp' not in st.session_state:
        st.session_state['latest_close_price_cmp'] = 100.0  # Default value
