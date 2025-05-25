# OptionTrading/utils.py

import pandas as pd
import streamlit as st

def initialize_session_state():
    if 'stock_data_eu' not in st.session_state:
        st.session_state['stock_data_eu'] = pd.DataFrame()
    if 'stock_data_am' not in st.session_state:
        st.session_state['stock_data_am'] = pd.DataFrame()
    if 'stock_data_nn' not in st.session_state:
        st.session_state['stock_data_nn'] = pd.DataFrame()
    if 'volatility_eu' not in st.session_state:
        st.session_state['volatility_eu'] = 20.0  # Default 20% volatility
    if 'latest_close_price_eu' not in st.session_state:
        st.session_state['latest_close_price_eu'] = 100.0
    if 'volatility_am' not in st.session_state:
        st.session_state['volatility_am'] = 20.0
    if 'latest_close_price_am' not in st.session_state:
        st.session_state['latest_close_price_am'] = 100.0
    if 'volatility_nn' not in st.session_state:
        st.session_state['volatility_nn'] = 20.0
    if 'latest_close_price_nn' not in st.session_state:
        st.session_state['latest_close_price_nn'] = 100.0
    if 'volatility_3d' not in st.session_state:
        st.session_state['volatility_3d'] = 20.0
    if 'latest_close_price_3d' not in st.session_state:
        st.session_state['latest_close_price_3d'] = 100.0
    if 'model_nn' not in st.session_state:
        st.session_state['model_nn'] = None
    if 'history_nn' not in st.session_state:
        st.session_state['history_nn'] = {'loss': []}
    if 'training_features' not in st.session_state:
        st.session_state['training_features'] = None
    if 'training_prices' not in st.session_state:
        st.session_state['training_prices'] = None
    if 'training_data_type' not in st.session_state:
        st.session_state['training_data_type'] = None