# OptionTrading/app.py
# Option Pricing Application with Black-Scholes, Crank-Nicolson, FOBSM, and Neural Networks
# Featuring 3D visualizations, sensitivity analysis, and Monte Carlo simulations

# Standard libraries
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit for the web interface
import streamlit as st

# Machine learning
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Custom modules
from DataFetcher import fetch_intraday_stock_data
from pricing_models import (
    black_scholes_call,
    black_scholes_put,
    crank_nicolson_american_option,
    monte_carlo_simulation,
    fobs_m_option_pricing
)
from visualizations import generate_heatmap, plot_training_history, plot_prediction_vs_actual
from surface_plots import plot_option_price_surface, plot_option_price_vs_strike_expiry
from neural_networks import (
    FOBSMNeuralNetwork,
    OptionPricingDataset, 
    OptionPricingTrainer,
    predict_option_price,
    calculate_prediction_intervals
)
from utils import initialize_session_state


# Initialize session state variables if they don't exist
initialize_session_state()

# Set Seaborn theme for consistent styling
sns.set_style("whitegrid")


def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Option Pricing Application", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    selected_tab = st.sidebar.radio("Go to", ["Introduction", "European Options", "American Options", "Neural Networks"])
    
    # Display the selected tab
    if selected_tab == "Introduction":
        display_introduction()
    elif selected_tab == "European Options":
        display_european_options()
    elif selected_tab == "American Options":
        display_american_options()
    elif selected_tab == "Neural Networks":
        display_neural_networks()


def display_introduction():
    st.title("Option Pricing Application")
    st.header("Introduction")
    st.markdown("""
    ### **Project Goal**
    
    This application provides a comprehensive analysis of option pricing using the **Black-Scholes Model**, **Neural Networks**, and extends it to accommodate **American options** through the **Crank-Nicolson numerical scheme** and **FOBSM (Fractional Order Black-Scholes Model)**. By comparing various pricing methods, users can gain insights into the impact of different models on option valuation and understand the practical applications of these models in the financial market.
    
    ### **Why This Project is Interesting and Market Applicable**
    
    - **Informed Decision-Making:** Investors and traders can utilize these pricing models to make informed decisions about buying and selling options.
    - **Risk Management:** Understanding option pricing helps in assessing the risk and potential returns associated with different option strategies.
    - **Market Efficiency:** Accurate pricing models contribute to the overall efficiency of financial markets by ensuring options are fairly valued.
    - **Educational Value:** The application serves as an educational tool for those looking to understand the mathematical foundations and modern machine learning approaches to option pricing.
    
    ### **Model Limitations**
    
    - **Black-Scholes Model:**
      - **Applicability:** Designed primarily for **European-style options** which can only be exercised at expiration.
      - **Assumptions:** Assumes constant volatility and risk-free rate, log-normal distribution of asset prices, and no dividends.
      - **Limitations:** Does not account for early exercise features inherent to **American options**, especially **puts** and **calls on dividend-paying stocks**.
    
    - **Crank-Nicolson Scheme:**
      - **Purpose:** Extends the Black-Scholes Model to price **American options** by accommodating the possibility of early exercise.
      - **Complexity:** Involves numerical methods and computational resources.
      - **Accuracy:** Provides more accurate pricing for American options but requires careful implementation to ensure stability and convergence.
    
    - **FOBSM (Fractional Order Black-Scholes Model):**
      - **Purpose:** Incorporates fractional calculus to model memory effects and long-range dependencies in financial markets, providing more realistic option pricing for certain market conditions.
      - **Complexity:** Uses fractional derivatives which require specialized numerical methods and deeper mathematical understanding.
    
    - **Neural Networks:**
      - **Data Dependence:** Performance heavily relies on the quality and quantity of training data.
      - **Black Box Nature:** Less interpretable compared to traditional models.
      - **Overfitting Risk:** Requires proper validation to ensure generalization.
    """)


def display_european_options():
    st.header("European Options Pricing with Black-Scholes Model")
    st.markdown("""
        The **Black-Scholes Model** provides a mathematical framework for pricing **European-style options**. This section allows you to:
        - **Calculate Option Prices:** Using the Black-Scholes formula.
        - **Conduct Sensitivity Analysis:** Visualize how option prices respond to changes in key parameters through heatmaps.
        - **Assess Uncertainty:** Generate confidence intervals to understand the reliability of option price estimates.
        - **3D Visualization:** Explore option price surfaces in 3D for deeper insights.
    """)
    
    # Add tabs for different visualizations
    eu_tabs = st.tabs(["Option Calculator", "3D Visualization"])
    
    with eu_tabs[0]:
        with st.container():
            st.subheader("Fetch Stock Data for European Option")
            # 1. Add a Search Bar for Ticker Symbols
            ticker_symbol_eu = st.text_input(
                "European Option - Stock Ticker Symbol (e.g., AAPL)",
                value="AAPL",
                help="Enter the stock ticker symbol to fetch real-time data."
            )

            fetch_data_button_eu = st.button("Fetch Stock Data for European Option", key="fetch_eu")

            if fetch_data_button_eu:
                # Fetch data and store in session_state
                try:
                    with st.spinner('Fetching data...'):
                        stock_data_eu = fetch_intraday_stock_data(ticker_symbol_eu)
                    if not stock_data_eu:
                        st.error(f"No intraday data returned for {ticker_symbol_eu}. Please check the symbol and try again.")
                    else:
                        # Extract closing prices
                        closing_prices_eu = [float(entry['4. close']) for entry in stock_data_eu.values()]
                        closing_prices_eu.reverse()  # Chronological order

                        if len(closing_prices_eu) < 2:
                            st.error("Not enough data points to calculate volatility.")
                        else:
                            # Calculate log returns
                            log_returns_eu = np.diff(np.log(closing_prices_eu))
                            volatility_eu = np.std(log_returns_eu) * np.sqrt(252) * 100  # Annualized volatility in %

                            # Latest close price
                            latest_close_price_eu = closing_prices_eu[-1]

                            # Store in session_state
                            st.session_state['stock_data_eu'] = pd.DataFrame({
                                'timestamp': list(stock_data_eu.keys()),
                                'close_price': closing_prices_eu
                            })
                            st.session_state['volatility_eu'] = volatility_eu
                            st.session_state['latest_close_price_eu'] = latest_close_price_eu

                            st.success(f"**Latest Close Price for {ticker_symbol_eu}:** ${latest_close_price_eu:.2f}")
                            st.write(f"**Annualized Historical Volatility:** {volatility_eu:.2f}%")

                            # Update input fields with fetched data
                            S_eu = st.number_input(
                                "European Option - Underlying Asset Price (S)",
                                min_value=0.01,
                                value=latest_close_price_eu,
                                step=1.0,
                                format="%.2f",
                                help="Current price of the underlying asset."
                            )
                            sigma_eu = st.number_input(
                                "European Option - Volatility (σ) (%)",
                                min_value=0.01,
                                value=volatility_eu,
                                step=0.1,
                                format="%.2f",
                                help="Annual volatility of the underlying asset."
                            ) / 100
                            # Update other inputs as necessary or keep them as manual inputs
                            K_eu = st.number_input(
                                "European Option - Strike Price (K)",
                                min_value=0.01,
                                value=100.0,
                                step=1.0,
                                format="%.2f",
                                help="Strike price of the option."
                            )
                            T_eu = st.number_input(
                                "European Option - Time to Maturity (T) in years",
                                min_value=0.01,
                                value=1.0,
                                step=0.1,
                                format="%.2f",
                                help="Time remaining until the option's expiration."
                            )
                            r_eu = st.number_input(
                                "European Option - Risk-Free Rate (r) (%)",
                                min_value=0.0,
                                value=5.0,
                                step=0.1,
                                format="%.2f",
                                help="Annual risk-free interest rate."
                            ) / 100
                            option_type_eu = st.selectbox(
                                "European Option - Option Type",
                                ("Call", "Put"),
                                help="Select the type of European option."
                            )
                except Exception as e:
                    st.error(f"An error occurred while fetching data: {e}")
            else:
                # Default input fields if no data is fetched
                S_eu = st.number_input(
                    "European Option - Underlying Asset Price (S)",
                    min_value=0.01,
                    value=st.session_state.get('latest_close_price_eu', 100.0),
                    step=1.0,
                    format="%.2f",
                    help="Current price of the underlying asset."
                )
                K_eu = st.number_input(
                    "European Option - Strike Price (K)",
                    min_value=0.01,
                    value=100.0,
                    step=1.0,
                    format="%.2f",
                    help="Strike price of the option."
                )
                T_eu = st.number_input(
                    "European Option - Time to Maturity (T) in years",
                    min_value=0.01,
                    value=1.0,
                    step=0.1,
                    format="%.2f",
                    help="Time remaining until the option's expiration."
                )
                r_eu = st.number_input(
                    "European Option - Risk-Free Rate (r) (%)",
                    min_value=0.0,
                    value=5.0,
                    step=0.1,
                    format="%.2f",
                    help="Annual risk-free interest rate."
                ) / 100
                sigma_eu = st.number_input(
                    "European Option - Volatility (σ) (%)",
                    min_value=0.01,
                    value=st.session_state.get('volatility_eu', 20.0),
                    step=0.1,
                    format="%.2f",
                    help="Annual volatility of the underlying asset."
                ) / 100
                option_type_eu = st.selectbox(
                    "European Option - Option Type",
                    ("Call", "Put"),
                    help="Select the type of European option."
                )

        st.subheader("Calculate Option Price")
        # Calculate Option Price
        calculate_price_eu = st.button("Calculate European Option Price", key="calc_eu")
        if calculate_price_eu:
            try:
                if option_type_eu.lower() == 'call':
                    option_price_eu = black_scholes_call(S_eu, K_eu, T_eu, r_eu, sigma_eu)
                    st.success(f"**Black-Scholes Call Option Price:** ${option_price_eu:.4f}")
                else:
                    option_price_eu = black_scholes_put(S_eu, K_eu, T_eu, r_eu, sigma_eu)
                    st.success(f"**Black-Scholes Put Option Price:** ${option_price_eu:.4f}")
            except Exception as e:
                st.error(f"An error occurred during option pricing: {e}")

        # Monte Carlo Simulation for Confidence Interval
        st.subheader("Monte Carlo Simulation for Confidence Interval")
        num_simulations_eu = st.number_input(
            "European Option - Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of Monte Carlo simulation runs to estimate confidence intervals."
        )
        confidence_level_eu = st.slider(
            "European Option - Confidence Level (%)",
            min_value=50,
            max_value=99,
            value=95,
            step=1,
            help="Probability that the true option price lies within the confidence interval."
        )
        
        run_mc_eu = st.button("Run Monte Carlo Simulation for European Option", key="mc_eu")
        if run_mc_eu:
            try:
                lower_eu, upper_eu, prices_eu = monte_carlo_simulation(
                    S=S_eu,
                    K=K_eu,
                    T=T_eu,
                    r=r_eu,
                    sigma=sigma_eu,
                    option_type=option_type_eu.lower(),
                    num_simulations=int(num_simulations_eu),
                    confidence_level=confidence_level_eu / 100
                )
                st.markdown(f"**{confidence_level_eu}% Confidence Interval:**")
                st.write(f"Lower Bound: ${lower_eu:.4f}")
                st.write(f"Upper Bound: ${upper_eu:.4f}")
                
                # Plot histogram
                fig_hist_eu, ax_hist_eu = plt.subplots(figsize=(6, 4))
                sns.histplot(prices_eu, bins=50, kde=True, ax=ax_hist_eu, color='skyblue')
                ax_hist_eu.axvline(lower_eu, color='red', linestyle='--', label=f'Lower {confidence_level_eu}% Bound')
                ax_hist_eu.axvline(upper_eu, color='green', linestyle='--', label=f'Upper {confidence_level_eu}% Bound')
                ax_hist_eu.set_title(f"Distribution of Simulated {option_type_eu.capitalize()} Option Prices")
                ax_hist_eu.set_xlabel("Option Price")
                ax_hist_eu.set_ylabel("Frequency")
                ax_hist_eu.legend()
                plt.tight_layout()
                st.pyplot(fig_hist_eu)
            except Exception as e:
                st.error(f"An error occurred during simulation: {e}")

        # Heatmap Sensitivity Analysis
        st.subheader("Sensitivity Analysis with Heatmaps")
        st.markdown("""
        Visualize how the option price changes with variations in two parameters simultaneously.
        """)
        
        # Parameter selection for heatmap
        available_params_eu = ['S', 'T', 'r', 'sigma']  # Excluding 'K' as it's held constant
        param1_eu = st.selectbox(
            "European Option - Select First Parameter (X-axis)",
            available_params_eu,
            index=0,
            help="Choose the first parameter to vary."
        )
        param2_eu = st.selectbox(
            "European Option - Select Second Parameter (Y-axis)",
            [param for param in available_params_eu if param != param1_eu],
            index=1,
            help="Choose the second parameter to vary."
        )
        
        range_percent_eu = st.slider(
            "European Option - Select the percentage range around the current values:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Defines the range for each parameter as ±[percentage]% of its current value."
        )
        
        generate_heatmap_eu = st.button("Generate Heatmap for European Option", key="heatmap_eu")
        if generate_heatmap_eu:
            try:
                fig_heatmap_eu = generate_heatmap(
                    param1=param1_eu,
                    param2=param2_eu,
                    S=S_eu,
                    K=K_eu,
                    T=T_eu,
                    r=r_eu,
                    sigma=sigma_eu,
                    option_type=option_type_eu.lower(),
                    range_percent=range_percent_eu
                )
                plt.tight_layout()
                st.pyplot(fig_heatmap_eu)
            except Exception as e:
                st.error(f"An error occurred while generating the heatmap: {e}")
    
    # 3D Visualization Tab - provides interactive 3D surfaces of option pricing relationships
    with eu_tabs[1]:
        st.subheader("3D Option Price Surface Visualization")
        st.markdown("""
        Explore how option prices change across different parameter combinations in 3D space. 
        Adjust the parameters below to customize the visualization.
        """)
        
        # Add ticker data fetching for 3D visualization
        st.subheader("Fetch Stock Data for 3D Visualization")
        ticker_symbol_3d = st.text_input(
            "3D Viz - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data for 3D visualization."
        )

        fetch_data_button_3d = st.button("Fetch Stock Data for 3D Visualization", key="fetch_3d")

        if fetch_data_button_3d:
            try:
                with st.spinner('Fetching data for 3D visualization...'):
                    stock_data_3d = fetch_intraday_stock_data(ticker_symbol_3d)
                if not stock_data_3d:
                    st.error(f"No data returned for {ticker_symbol_3d}. Using default values.")
                else:
                    # Extract closing prices
                    closing_prices_3d = [float(entry['4. close']) for entry in stock_data_3d.values()]
                    closing_prices_3d.reverse()  # Chronological order

                    if len(closing_prices_3d) >= 2:
                        # Calculate log returns and volatility
                        log_returns_3d = np.diff(np.log(closing_prices_3d))
                        volatility_3d = np.std(log_returns_3d) * np.sqrt(252) * 100  # Annualized volatility in %
                        latest_close_price_3d = closing_prices_3d[-1]

                        # Store in session_state
                        st.session_state['volatility_3d'] = volatility_3d
                        st.session_state['latest_close_price_3d'] = latest_close_price_3d

                        st.success(f"**Latest Close Price for {ticker_symbol_3d}:** ${latest_close_price_3d:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_3d:.2f}%")
                    else:
                        st.warning("Not enough data points. Using default values.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")
        
        # Visualization type selection - offers two different 3D perspectives on option pricing
        viz_type = st.radio(
            "Select Visualization Type",
            ["Price vs Volatility & Time", "Price vs Strike & Expiry"],
            help="Choose the type of 3D visualization to display."
        )
        
        # Parameters for 3D visualization
        col1_3d, col2_3d = st.columns(2)
        
        with col1_3d:
            # Common parameters for both visualization types
            S_3d = st.number_input(
                "3D Viz - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state.get('latest_close_price_3d', st.session_state.get('latest_close_price_eu', 100.0)),
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_3d = st.number_input(
                "3D Viz - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            r_3d = st.number_input(
                "3D Viz - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
        
        with col2_3d:
            option_type_3d = st.selectbox(
                "3D Viz - Option Type",
                ("Call", "Put"),
                help="Select the type of option for visualization."
            )
            
            # Parameters specific to the first visualization type
            if viz_type == "Price vs Volatility & Time":
                sigma_min = st.number_input(
                    "Minimum Volatility (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=5.0,
                    step=1.0,
                    help="Minimum volatility for the 3D surface."
                ) / 100
                
                sigma_max = st.number_input(
                    "Maximum Volatility (%)",
                    min_value=10.0,
                    max_value=100.0,
                    value=60.0,
                    step=5.0,
                    help="Maximum volatility for the 3D surface."
                ) / 100
                
                T_min = st.number_input(
                    "Minimum Time to Maturity (years)",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.1,
                    step=0.05,
                    help="Minimum time to maturity for the 3D surface."
                )
                
                T_max = st.number_input(
                    "Maximum Time to Maturity (years)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Maximum time to maturity for the 3D surface."
                )
            
            # Parameters specific to the second visualization type
            else:
                K_range_percent = st.slider(
                    "Strike Price Range (%)",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Range of strike prices to display as percentage of current price."
                )
                
                sigma_3d = st.number_input(
                    "3D Viz - Volatility (σ) (%)",
                    min_value=0.01,
                    value=st.session_state.get('volatility_3d', st.session_state.get('volatility_eu', 20.0)),
                    step=0.1,
                    format="%.2f",
                    help="Volatility for the 3D surface."
                ) / 100
                
                T_max_2 = st.number_input(
                    "Maximum Time to Maturity (years)",
                    min_value=0.5,
                    max_value=5.0,
                    value=2.0,
                    step=0.5,
                    help="Maximum time to maturity for the 3D surface."
                )
                
                num_points = st.slider(
                    "Number of Points (Grid Size)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Number of points to calculate for each axis. Higher values give smoother surfaces but take longer to compute."
                )
        
        # Button to generate the 3D visualization
        generate_3d_button = st.button("Generate 3D Visualization", key="generate_3d")
        
        if generate_3d_button:
            try:
                with st.spinner('Generating 3D visualization...'):
                    if viz_type == "Price vs Volatility & Time":
                        # Validate input ranges
                        if sigma_min >= sigma_max:
                            st.error("Minimum volatility must be less than maximum volatility.")
                            st.stop()
                        if T_min >= T_max:
                            st.error("Minimum maturity time must be less than maximum maturity time.")
                            st.stop()
                            
                        try:
                            fig_3d = plot_option_price_surface(
                                S=S_3d,
                                K=K_3d,
                                r=r_3d,
                                option_type=option_type_3d.lower(),
                                sigma_range=(sigma_min, sigma_max),
                                T_range=(T_min, T_max),
                                sigma_points=20, 
                                T_points=20
                            )
                        except ValueError as ve:
                            st.error(f"Invalid input for visualization: {ve}")
                            st.stop()
                        except Exception as ex:
                            st.error(f"Error generating surface plot: {ex}")
                            st.stop()
                    else:
                        # Calculate strike price range based on percentage
                        K_min = S_3d * (1 - K_range_percent/100)
                        K_max = S_3d * (1 + K_range_percent/100)
                        
                        # Validate inputs
                        if K_min <= 0:
                            st.error("Strike price range is too large. Minimum strike price would be negative or zero.")
                            st.stop()
                        
                        # Create arrays for strike prices and times
                        K_array = np.linspace(K_min, K_max, num_points)
                        T_array = np.linspace(0.1, T_max_2, num_points)
                        
                        try:
                            fig_3d = plot_option_price_vs_strike_expiry(
                                S=S_3d,
                                K_array=K_array,
                                T_array=T_array,
                                r=r_3d,
                                sigma=sigma_3d,
                                option_type=option_type_3d.lower()
                            )
                        except ValueError as ve:
                            st.error(f"Invalid input for visualization: {ve}")
                            st.stop()
                        except Exception as ex:
                            st.error(f"Error generating strike/expiry plot: {ex}")
                            st.stop()
                    
                    # Display the figure if we got here without errors
                    st.pyplot(fig_3d)
                    st.success("3D visualization generated successfully!")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        
        # Add information about interpreting the 3D visualization
        st.info("""
        **How to Interpret the 3D Visualization:**
        
        1. **Color gradient** indicates option price levels - darker colors typically represent higher prices.
        2. **X and Y axes** show the varying parameters as specified in your selection.
        3. **Z axis** represents the option price.
        4. **Red reference point** shows a specific combination of parameters for easy comparison.
        
        **Tips for Interaction:**
        - Click and drag to rotate the 3D view
        - Use the scroll wheel to zoom in/out
        - Hold Shift while dragging to pan the view
        """)


def display_american_options():
    st.header("American Options Pricing with Advanced Numerical Schemes")
    st.markdown("""
        The **Crank-Nicolson numerical scheme** and **FOBSM (Fractional Order Black-Scholes Model)** extend the Black-Scholes Model to accommodate **American-style options**, which can be exercised at any time before expiration. This section allows you to:
        - **Calculate Option Prices:** Using chosen numerical methods.
        - **Conduct Sensitivity Analysis:** Visualize how option prices respond to changes in key parameters through heatmaps.
        - **Assess Uncertainty:** Generate confidence intervals to understand the reliability of option price estimates.
    """)
    
    with st.container():
        st.subheader("Fetch Stock Data for American Option")
        # 1. Add a Search Bar for Ticker Symbols
        ticker_symbol_am = st.text_input(
            "American Option - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data."
        )

        fetch_data_button_am = st.button("Fetch Stock Data for American Option", key="fetch_am")

        if fetch_data_button_am:
            # Fetch data and store in session_state
            try:
                with st.spinner('Fetching data...'):
                    stock_data_am = fetch_intraday_stock_data(ticker_symbol_am)
                if not stock_data_am:
                    st.error(f"No intraday data returned for {ticker_symbol_am}. Please check the symbol and try again.")
                else:
                    # Extract closing prices
                    closing_prices_am = [float(entry['4. close']) for entry in stock_data_am.values()]
                    closing_prices_am.reverse()  # Chronological order

                    if len(closing_prices_am) < 2:
                        st.error("Not enough data points to calculate volatility.")
                    else:
                        # Calculate log returns
                        log_returns_am = np.diff(np.log(closing_prices_am))
                        volatility_am = np.std(log_returns_am) * np.sqrt(252) * 100  # Annualized volatility in %

                        # Latest close price
                        latest_close_price_am = closing_prices_am[-1]

                        # Store in session_state
                        st.session_state['stock_data_am'] = pd.DataFrame({
                            'timestamp': list(stock_data_am.keys()),
                            'close_price': closing_prices_am
                        })
                        st.session_state['volatility_am'] = volatility_am
                        st.session_state['latest_close_price_am'] = latest_close_price_am

                        st.success(f"**Latest Close Price for {ticker_symbol_am}:** ${latest_close_price_am:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_am:.2f}%")

                        # Update input fields with fetched data
                        S_am = st.number_input(
                            "American Option - Underlying Asset Price (S)",
                            min_value=0.01,
                            value=latest_close_price_am,
                            step=1.0,
                            format="%.2f",
                            help="Current price of the underlying asset."
                        )
                        sigma_am = st.number_input(
                            "American Option - Volatility (σ) (%)",
                            min_value=0.01,
                            value=volatility_am,
                            step=0.1,
                            format="%.2f",
                            help="Annual volatility of the underlying asset."
                        ) / 100
                        # Update other inputs as necessary or keep them as manual inputs
                        K_am = st.number_input(
                            "American Option - Strike Price (K)",
                            min_value=0.01,
                            value=100.0,
                            step=1.0,
                            format="%.2f",
                            help="Strike price of the option."
                        )
                        T_am = st.number_input(
                            "American Option - Time to Maturity (T) in years",
                            min_value=0.01,
                            value=1.0,
                            step=0.1,
                            format="%.2f",
                            help="Time remaining until the option's expiration."
                        )
                        r_am = st.number_input(
                            "American Option - Risk-Free Rate (r) (%)",
                            min_value=0.0,
                            value=5.0,
                            step=0.1,
                            format="%.2f",
                            help="Annual risk-free interest rate."
                        ) / 100
                        option_type_am = st.selectbox(
                            "American Option - Option Type",
                            ("Call", "Put"),
                            help="Select the type of American option."
                        )
                        M_am = st.number_input(
                            "American Option - Number of Asset Price Steps (M)",
                            min_value=50,
                            max_value=500,
                            value=100,
                            step=10,
                            help="Number of steps in the asset price dimension for the finite difference grid."
                        )
                        N_am = st.number_input(
                            "American Option - Number of Time Steps (N)",
                            min_value=50,
                            max_value=500,
                            value=100,
                            step=10,
                            help="Number of steps in the time dimension for the finite difference grid."
                        )
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
        else:
            # Default input fields if no data is fetched
            S_am = st.number_input(
                "American Option - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state.get('latest_close_price_am', 100.0),
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_am = st.number_input(
                "American Option - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_am = st.number_input(
                "American Option - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_am = st.number_input(
                "American Option - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_am = st.number_input(
                "American Option - Volatility (σ) (%)",
                min_value=0.01,
                value=st.session_state.get('volatility_am', 20.0),
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_am = st.selectbox(
                "American Option - Option Type",
                ("Call", "Put"),
                help="Select the type of American option."
            )
            M_am = st.number_input(
                "American Option - Number of Asset Price Steps (M)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of steps in the asset price dimension for the finite difference grid."
            )
            N_am = st.number_input(
                "American Option - Number of Time Steps (N)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of steps in the time dimension for the finite difference grid."
            )

    st.subheader("Calculate Option Price")
    # Calculate Option Price
    calculate_price_am = st.button("Calculate American Option Price", key="calc_am")
    if calculate_price_am:
        try:
            if option_type_am.lower() == 'call':
                option_price_am = crank_nicolson_american_option(
                    S=S_am,
                    K=K_am,
                    T=T_am,
                    r=r_am,
                    sigma=sigma_am,
                    option_type='call',
                    M=int(M_am),
                    N=int(N_am)
                )
                st.success(f"**Crank-Nicolson Call Option Price:** ${option_price_am:.4f}")
            else:
                option_price_am = crank_nicolson_american_option(
                    S=S_am,
                    K=K_am,
                    T=T_am,
                    r=r_am,
                    sigma=sigma_am,
                    option_type='put',
                    M=int(M_am),
                    N=int(N_am)
                )
                st.success(f"**Crank-Nicolson Put Option Price:** ${option_price_am:.4f}")
        except Exception as e:
            st.error(f"An error occurred during option pricing: {e}")

    # Monte Carlo Simulation for Confidence Interval
    st.subheader("Monte Carlo Simulation for Confidence Interval")
    num_simulations_am = st.number_input(
        "American Option - Number of Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of Monte Carlo simulation runs to estimate confidence intervals."
    )
    confidence_level_am = st.slider(
        "American Option - Confidence Level (%)",
        min_value=50,
        max_value=99,
        value=95,
        step=1,
        help="Probability that the true option price lies within the confidence interval."
    )
    
    run_mc_am = st.button("Run Monte Carlo Simulation for American Option", key="mc_am")
    if run_mc_am:
        try:
            lower_am, upper_am, prices_am = monte_carlo_simulation(
                S=S_am,
                K=K_am,
                T=T_am,
                r=r_am,
                sigma=sigma_am,
                option_type=option_type_am.lower(),
                num_simulations=int(num_simulations_am),
                confidence_level=confidence_level_am / 100
            )
            st.markdown(f"**{confidence_level_am}% Confidence Interval:**")
            st.write(f"Lower Bound: ${lower_am:.4f}")
            st.write(f"Upper Bound: ${upper_am:.4f}")
            
            # Plot histogram
            fig_hist_am, ax_hist_am = plt.subplots(figsize=(6, 4))
            sns.histplot(prices_am, bins=50, kde=True, ax=ax_hist_am, color='salmon')
            ax_hist_am.axvline(lower_am, color='red', linestyle='--', label=f'Lower {confidence_level_am}% Bound')
            ax_hist_am.axvline(upper_am, color='green', linestyle='--', label=f'Upper {confidence_level_am}% Bound')
            ax_hist_am.set_title(f"Distribution of Simulated {option_type_am.capitalize()} Option Prices")
            ax_hist_am.set_xlabel("Option Price")
            ax_hist_am.set_ylabel("Frequency")
            ax_hist_am.legend()
            plt.tight_layout()
            st.pyplot(fig_hist_am)
        except Exception as e:
            st.error(f"An error occurred during simulation: {e}")

    # Heatmap Sensitivity Analysis
    st.subheader("Sensitivity Analysis with Heatmaps")
    st.markdown("""
    Visualize how the option price changes with variations in two parameters simultaneously.
    """)
    
    # Parameter selection for heatmap
    available_params_am = ['S', 'T', 'r', 'sigma']  # Excluding 'K' as it's held constant
    param1_am = st.selectbox(
        "American Option - Select First Parameter (X-axis)",
        available_params_am,
        index=0,
        help="Choose the first parameter to vary."
    )
    param2_am = st.selectbox(
        "American Option - Select Second Parameter (Y-axis)",
        [param for param in available_params_am if param != param1_am],
        index=1,
        help="Choose the second parameter to vary."
    )
    
    range_percent_am = st.slider(
        "American Option - Select the percentage range around the current values:",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Defines the range for each parameter as ±[percentage]% of its current value."
    )
    
    generate_heatmap_am = st.button("Generate Heatmap for American Option", key="heatmap_am")
    if generate_heatmap_am:
        try:
            fig_heatmap_am = generate_heatmap(
                param1=param1_am,
                param2=param2_am,
                S=S_am,
                K=K_am,
                T=T_am,
                r=r_am,
                sigma=sigma_am,
                option_type=option_type_am.lower(),
                range_percent=range_percent_am
            )
            plt.tight_layout()
            st.pyplot(fig_heatmap_am)
        except Exception as e:
            st.error(f"An error occurred while generating the heatmap: {e}")


def display_neural_networks():
    st.header("Option Pricing with Neural Networks")
    st.markdown("""
    This section allows you to:
    - **Train a Neural Network Model** on historical option pricing data.
    - **Predict Option Prices** using the trained model.
    - **Visualize Model Performance** and predictions.
    """)

    with st.container():
        st.subheader("Fetch Stock Data for Neural Network")
        # Add ticker data fetching for neural networks
        ticker_symbol_nn = st.text_input(
            "Neural Network - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data for neural network training."
        )

        fetch_data_button_nn = st.button("Fetch Stock Data for Neural Network", key="fetch_nn")

        if fetch_data_button_nn:
            try:
                with st.spinner('Fetching data for neural network...'):
                    stock_data_nn = fetch_intraday_stock_data(ticker_symbol_nn)
                if not stock_data_nn:
                    st.error(f"No data returned for {ticker_symbol_nn}. Please check the symbol and try again.")
                else:
                    # Extract closing prices
                    closing_prices_nn = [float(entry['4. close']) for entry in stock_data_nn.values()]
                    closing_prices_nn.reverse()  # Chronological order

                    if len(closing_prices_nn) >= 2:
                        # Calculate log returns and volatility
                        log_returns_nn = np.diff(np.log(closing_prices_nn))
                        volatility_nn = np.std(log_returns_nn) * np.sqrt(252) * 100  # Annualized volatility in %
                        latest_close_price_nn = closing_prices_nn[-1]

                        # Store in session_state
                        st.session_state['stock_data_nn'] = pd.DataFrame({
                            'timestamp': list(stock_data_nn.keys()),
                            'close_price': closing_prices_nn
                        })
                        st.session_state['volatility_nn'] = volatility_nn
                        st.session_state['latest_close_price_nn'] = latest_close_price_nn

                        st.success(f"**Latest Close Price for {ticker_symbol_nn}:** ${latest_close_price_nn:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_nn:.2f}%")
                        st.write(f"**Data Points Available:** {len(closing_prices_nn)}")
                    else:
                        st.warning("Not enough data points to calculate volatility.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

        st.subheader("Prepare Data for Neural Network")
        # Assume the user selects whether to use European or American option data
        option_type_nn = st.selectbox(
            "Select Option Type for Neural Network",
            ("European", "American"),
            help="Choose the option type to train the Neural Network."
        )

        if option_type_nn == "European":
            # Use neural network data if available, otherwise fall back to European data
            default_price = st.session_state.get('latest_close_price_nn', 
                          st.session_state.get('latest_close_price_eu', 100.0))
            default_vol = st.session_state.get('volatility_nn', 
                         st.session_state.get('volatility_eu', 20.0))
            
            # Retrieve parameters
            S_nn = st.number_input(
                "Neural Network - Underlying Asset Price (S)",
                min_value=0.01,
                value=default_price,
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_nn = st.number_input(
                "Neural Network - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_nn = st.number_input(
                "Neural Network - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_nn = st.number_input(
                "Neural Network - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_nn = st.number_input(
                "Neural Network - Volatility (σ) (%)",
                min_value=0.01,
                value=default_vol,
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_selected_nn = st.selectbox(
                "Neural Network - Option Type",
                ("Call", "Put"),
                help="Select the type of option."
            )
        else:
            # Use neural network data if available, otherwise fall back to American data
            default_price = st.session_state.get('latest_close_price_nn', 
                          st.session_state.get('latest_close_price_am', 100.0))
            default_vol = st.session_state.get('volatility_nn', 
                         st.session_state.get('volatility_am', 20.0))
            
            # Retrieve parameters
            S_nn = st.number_input(
                "Neural Network - Underlying Asset Price (S)",
                min_value=0.01,
                value=default_price,
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_nn = st.number_input(
                "Neural Network - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_nn = st.number_input(
                "Neural Network - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_nn = st.number_input(
                "Neural Network - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_nn = st.number_input(
                "Neural Network - Volatility (σ) (%)",
                min_value=0.01,
                value=default_vol,
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_selected_nn = st.selectbox(
                "Neural Network - Option Type",
                ("Call", "Put"),
                help="Select the type of option."
            )

        # Fetch training data
        if option_type_nn == "European":
            # Use neural network data if available, otherwise use European data
            if 'stock_data_nn' in st.session_state and not st.session_state['stock_data_nn'].empty:
                pricing_data_eu = st.session_state['stock_data_nn'].copy()
            elif not st.session_state['stock_data_eu'].empty:
                pricing_data_eu = st.session_state['stock_data_eu'].copy()
            else:
                st.warning("No stock data available. Please fetch data first.")
                st.stop()
                
            # Limit data size for faster training (use only first 100 rows)
            if len(pricing_data_eu) > 100:
                pricing_data_eu = pricing_data_eu.head(100)
                
            # Ensure 'price' column exists for training
            if 'price' not in pricing_data_eu.columns:
                # Simulate option prices using Black-Scholes for synthetic training data
                pricing_data_eu['price'] = pricing_data_eu.apply(
                    lambda row: black_scholes_call(row['close_price'], K_nn, T_nn, r_nn, sigma_nn) if option_type_selected_nn.lower() == 'call' else black_scholes_put(row['close_price'], K_nn, T_nn, r_nn, sigma_nn),
                    axis=1
                )
            # Add 'K', 'T', 'r', 'sigma' columns based on user input
            pricing_data_eu['K'] = K_nn
            pricing_data_eu['T'] = T_nn
            pricing_data_eu['r'] = r_nn
            pricing_data_eu['sigma'] = sigma_nn
            # Rename 'close_price' to 'S'
            df_eu = pricing_data_eu[['close_price', 'K', 'T', 'r', 'sigma', 'price']].rename(columns={'close_price': 'S'})
        else:
            # Use neural network data if available, otherwise use American data
            if 'stock_data_nn' in st.session_state and not st.session_state['stock_data_nn'].empty:
                pricing_data_am = st.session_state['stock_data_nn'].copy()
            elif not st.session_state['stock_data_am'].empty:
                pricing_data_am = st.session_state['stock_data_am'].copy()
            else:
                st.warning("No stock data available. Please fetch data first.")
                st.stop()
                
            # Limit data size for faster training (use only first 50 rows for American options as they're slower to compute)
            if len(pricing_data_am) > 50:
                pricing_data_am = pricing_data_am.head(50)
                
            if 'price' not in pricing_data_am.columns:
                # Use simpler Black-Scholes instead of slow Crank-Nicolson for faster training
                pricing_data_am['price'] = pricing_data_am.apply(
                    lambda row: black_scholes_call(row['close_price'], K_nn, T_nn, r_nn, sigma_nn) if option_type_selected_nn.lower() == 'call' else black_scholes_put(row['close_price'], K_nn, T_nn, r_nn, sigma_nn),
                    axis=1
                )
            # Add 'K', 'T', 'r', 'sigma' columns based on user input
            pricing_data_am['K'] = K_nn
            pricing_data_am['T'] = T_nn
            pricing_data_am['r'] = r_nn
            pricing_data_am['sigma'] = sigma_nn
            # Rename 'close_price' to 'S'
            df_am = pricing_data_am[['close_price', 'K', 'T', 'r', 'sigma', 'price']].rename(columns={'close_price': 'S'})

    st.subheader("Train Neural Network Model")
    
    # Show training information
    if 'stock_data_nn' in st.session_state and not st.session_state['stock_data_nn'].empty:
        data_size = len(st.session_state['stock_data_nn'])
        st.info(f"**Training Data Available:** {min(data_size, 100 if option_type_nn == 'European' else 50)} samples (limited for fast training)")
        st.info("**Estimated Training Time:** 5-15 seconds")
    elif option_type_nn == "European" and not st.session_state['stock_data_eu'].empty:
        data_size = len(st.session_state['stock_data_eu'])
        st.info(f"**Training Data Available:** {min(data_size, 100)} samples (limited for fast training)")
        st.info("**Estimated Training Time:** 5-15 seconds")
    elif option_type_nn == "American" and not st.session_state['stock_data_am'].empty:
        data_size = len(st.session_state['stock_data_am'])
        st.info(f"**Training Data Available:** {min(data_size, 50)} samples (limited for fast training)")
        st.info("**Estimated Training Time:** 5-15 seconds")
    else:
        st.warning("No stock data available. Please fetch stock data first.")
    
    train_button = st.button("Train Neural Network", key="train_nn")

    if train_button:
        with st.spinner('Training Neural Network...'):
            try:
                if option_type_nn == "European":
                    X_train = df_eu[['S', 'K', 'T', 'r', 'sigma']].values
                    y_train = df_eu['price'].values
                    dataset = OptionPricingDataset(features=X_train, prices=y_train)
                    # Store training data in session state for visualization
                    st.session_state['training_features'] = X_train
                    st.session_state['training_prices'] = y_train
                    st.session_state['training_data_type'] = "European"
                else:
                    X_train = df_am[['S', 'K', 'T', 'r', 'sigma']].values
                    y_train = df_am['price'].values
                    dataset = OptionPricingDataset(features=X_train, prices=y_train)
                    # Store training data in session state for visualization
                    st.session_state['training_features'] = X_train
                    st.session_state['training_prices'] = y_train
                    st.session_state['training_data_type'] = "American"

                # Use smaller batch size and fewer epochs for faster training
                dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

                input_dim = 5  # Features: S, K, T, r, sigma
                model = FOBSMNeuralNetwork(input_dim=input_dim, hidden_dim=32)  # Smaller network
                trainer = OptionPricingTrainer(model=model, learning_rate=0.01)  # Higher learning rate

                epochs = 20  # Much fewer epochs for faster training
                history = {'loss': []}

                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                for epoch in range(1, epochs + 1):
                    metrics = trainer.train_epoch(dataloader, epoch)
                    history['loss'].append(metrics['loss'])
                    
                    # Update progress
                    progress = epoch / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Training... Epoch {epoch}/{epochs} - Loss: {metrics['loss']:.4f}")

                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                st.success("Neural Network trained successfully!")

                # Save the trained model and history to session_state for future predictions
                st.session_state['model_nn'] = model
                st.session_state['history_nn'] = history
            except Exception as e:
                st.error(f"An error occurred during Neural Network training: {e}")

    st.subheader("Predict Option Price with Neural Network")
    predict_button = st.button("Predict Option Price", key="predict_nn")

    if predict_button:
        try:
            if 'model_nn' not in st.session_state or st.session_state['model_nn'] is None:
                st.warning("Please train the Neural Network model first.")
            else:
                model_nn = st.session_state['model_nn']
                
                # Check if training data is available
                if 'training_features' in st.session_state:
                    scaler = StandardScaler()
                    scaler.fit(st.session_state['training_features'])
                    features = np.array([[S_nn, K_nn, T_nn, r_nn, sigma_nn]])
                    predicted_price = predict_option_price(model_nn, features, scaler)
                    st.success(f"**Predicted Option Price (NN):** ${predicted_price:.4f}")
                else:
                    st.warning("No training data available for scaling. Please train the model first.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    st.subheader("Visualize Neural Network Performance")
    if ('model_nn' in st.session_state and st.session_state['model_nn'] is not None and 
        'history_nn' in st.session_state):
        model_nn = st.session_state['model_nn']
        history_nn = st.session_state['history_nn']

        # Check if 'loss' key exists and has data
        if 'loss' in history_nn and len(history_nn['loss']) > 0:
            # Plot training loss
            fig_loss, ax_loss = plt.subplots(figsize=(8, 6))
            sns.lineplot(x=np.arange(1, len(history_nn['loss']) + 1), y=history_nn['loss'], ax=ax_loss, label='Training Loss')
            ax_loss.set_title("Neural Network Training Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss (MSE)")
            ax_loss.legend()
            plt.tight_layout()
            st.pyplot(fig_loss)
        else:
            st.warning("No training loss data available to plot.")

        # Plot Prediction vs Actual - only if training data is available
        try:
            st.write("**Prediction vs Actual Prices:**")
            # Check if training data variables exist in session state
            if ('training_data_type' in st.session_state and 
                'training_features' in st.session_state and 
                'training_prices' in st.session_state):
                
                X_train = st.session_state['training_features']
                y_train = st.session_state['training_prices']

                with torch.no_grad():
                    inputs = torch.tensor(X_train).float()
                    predictions = model_nn(inputs).numpy().flatten()

                fig_pred, ax_pred = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_train, y=predictions, alpha=0.5, ax=ax_pred)
                ax_pred.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')  # Diagonal line
                ax_pred.set_title('Neural Network Predictions vs Actual Prices')
                ax_pred.set_xlabel('Actual Prices')
                ax_pred.set_ylabel('Predicted Prices')
                plt.tight_layout()
                st.pyplot(fig_pred)
            else:
                st.info("Training data not available for visualization. Train the model first.")
        except Exception as e:
            st.warning(f"Could not generate prediction plot: {e}")
    else:
        st.info("Train the Neural Network to visualize performance metrics.")

if __name__ == "__main__":
    main()