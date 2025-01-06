# app.py


from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from DataFetcher import fetch_intraday_stock_data
from pricing_models import (
    black_scholes_call,
    black_scholes_put,
    crank_nicolson_american_option,
    monte_carlo_simulation,
    fobs_m_option_pricing
)
from visualizations import generate_heatmap, plot_training_history, plot_prediction_vs_actual
from neural_networks import (
    FOBSMNeuralNetwork,
    OptionPricingDataset, 
    OptionPricingTrainer,
    predict_option_price,
    calculate_prediction_intervals
)
from utils import initialize_session_state
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Initialize session state variables if they don't exist
initialize_session_state()

# Set Seaborn theme for consistent styling
sns.set_style("whitegrid")


def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Option Pricing Application", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.header("Navigation")
    selected_tab = st.sidebar.radio("Go to", ["Introduction", "European Options", "American Options", "Neural Networks", "Comparison"])
    
    # Display the selected tab
    if selected_tab == "Introduction":
        display_introduction()
    elif selected_tab == "European Options":
        display_european_options()
    elif selected_tab == "American Options":
        display_american_options()
    elif selected_tab == "Neural Networks":
        display_neural_networks()
    elif selected_tab == "Comparison":
        display_comparison()


def display_introduction():
    st.title("Option Pricing Application")
    st.header("Introduction")
    st.markdown("""
    ### **Project Goal**
    
    This application provides a comprehensive analysis of option pricing using the **Black-Scholes Model**, **Neural Networks**, and extends it to accommodate **American options** through the **Crank-Nicolson numerical scheme** and **FOBSM (Finite Optimal Boundary Schemes Method)**. By comparing various pricing methods, users can gain insights into the impact of different models on option valuation and understand the practical applications of these models in the financial market.
    
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
    
    - **FOBSM:**
      - **Purpose:** Further enhances numerical pricing methods by optimally handling boundary conditions and improving computational efficiency.
      - **Complexity:** More sophisticated and may require deeper mathematical understanding and careful implementation.
    
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
    """)
    
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


def display_american_options():
    st.header("American Options Pricing with Advanced Numerical Schemes")
    st.markdown("""
        The **Crank-Nicolson numerical scheme** and **FOBSM (Finite Optimal Boundary Schemes Method)** extend the Black-Scholes Model to accommodate **American-style options**, which can be exercised at any time before expiration. This section allows you to:
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
        st.subheader("Fetch and Prepare Data for Neural Network")
        # Assume the user selects whether to use European or American option data
        option_type_nn = st.selectbox(
            "Select Option Type for Neural Network",
            ("European", "American"),
            help="Choose the option type to train the Neural Network."
        )

        if option_type_nn == "European":
            # Ensure European option data is fetched
            if st.session_state['stock_data_eu'].empty:
                st.warning("Please fetch European option data first.")
                st.stop()
            # Retrieve parameters
            S_nn = st.number_input(
                "Neural Network - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_eu'],
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
                value=st.session_state['volatility_eu'],
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
            # Ensure American option data is fetched
            if st.session_state['stock_data_am'].empty:
                st.warning("Please fetch American option data first.")
                st.stop()
            # Retrieve parameters
            S_nn = st.number_input(
                "Neural Network - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_am'],
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
                value=st.session_state['volatility_am'],
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
            pricing_data_eu = st.session_state['stock_data_eu'].copy()
            # Ensure 'price' column exists for training
            # **Important:** Replace this with actual historical option pricing data
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
            pricing_data_am = st.session_state['stock_data_am'].copy()
            if 'price' not in pricing_data_am.columns:
                # Simulate option prices using Crank-Nicolson for synthetic training data
                pricing_data_am['price'] = pricing_data_am.apply(
                    lambda row: crank_nicolson_american_option(row['close_price'], K_nn, T_nn, r_nn, sigma_nn, option_type_selected_nn.lower(), M=100, N=100),
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
    train_button = st.button("Train Neural Network", key="train_nn")

    if train_button:
        with st.spinner('Training Neural Network...'):
            try:
                if option_type_nn == "European":
                    dataset = OptionPricingDataset(
                        features=df_eu[['S', 'K', 'T', 'r', 'sigma']].values,
                        prices=df_eu['price'].values
                    )
                else:
                    dataset = OptionPricingDataset(
                        features=df_am[['S', 'K', 'T', 'r', 'sigma']].values,
                        prices=df_am['price'].values
                    )

                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                input_dim = 5  # Features: S, K, T, r, sigma
                model = FOBSMNeuralNetwork(input_dim=input_dim)
                trainer = OptionPricingTrainer(model=model, learning_rate=0.001)

                epochs = 100
                history = {'loss': []}

                for epoch in range(1, epochs + 1):
                    metrics = trainer.train_epoch(dataloader, epoch)
                    history['loss'].append(metrics['loss'])
                    if epoch % 10 == 0:
                        st.write(f"Epoch {epoch}/{epochs} - Loss: {metrics['loss']:.4f}")

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
                scaler = StandardScaler()  # Ideally, load the scaler used during training
                # Fit scaler on the training data
                if option_type_nn == "European":
                    scaler.fit(df_eu[['S', 'K', 'T', 'r', 'sigma']].values)
                    features = np.array([[S_nn, K_nn, T_nn, r_nn, sigma_nn]])
                else:
                    scaler.fit(df_am[['S', 'K', 'T', 'r', 'sigma']].values)
                    features = np.array([[S_nn, K_nn, T_nn, r_nn, sigma_nn]])

                predicted_price = predict_option_price(model_nn, features, scaler)
                st.success(f"**Predicted Option Price (NN):** ${predicted_price:.4f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    st.subheader("Visualize Neural Network Performance")
    if 'model_nn' in st.session_state and 'history_nn' in st.session_state:
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

        # Plot Prediction vs Actual
        st.write("**Prediction vs Actual Prices:**")
        if option_type_nn == "European":
            X_train = df_eu[['S', 'K', 'T', 'r', 'sigma']].values
            y_train = df_eu['price'].values
        else:
            X_train = df_am[['S', 'K', 'T', 'r', 'sigma']].values
            y_train = df_am['price'].values

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
        st.info("Train the Neural Network to visualize performance metrics.")

def display_comparison():
    st.header("Accuracy Comparison: European vs American Options")
    st.markdown("""
    This section allows you to compare the pricing accuracy between **European Options** (Black-Scholes Model and Neural Networks) and **American Options** (Crank-Nicolson Scheme and FOBSM) under identical input parameters.
    """)
    
    with st.container():
        st.subheader("Comparison Input Parameters")
        # 1. Add a Search Bar for Ticker Symbols
        ticker_symbol_cmp = st.text_input(
            "Comparison - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data for comparison."
        )

        fetch_data_button_cmp = st.button("Fetch Stock Data for Comparison", key="fetch_cmp")

        if fetch_data_button_cmp:
            # Fetch data and store in session_state
            try:
                with st.spinner('Fetching data...'):
                    stock_data_cmp = fetch_intraday_stock_data(ticker_symbol_cmp)
                if not stock_data_cmp:
                    st.error(f"No intraday data returned for {ticker_symbol_cmp}. Please check the symbol and try again.")
                else:
                    # Extract closing prices
                    closing_prices_cmp = [float(entry['4. close']) for entry in stock_data_cmp.values()]
                    closing_prices_cmp.reverse()  # Chronological order

                    if len(closing_prices_cmp) < 2:
                        st.error("Not enough data points to calculate volatility.")
                    else:
                        # Calculate log returns
                        log_returns_cmp = np.diff(np.log(closing_prices_cmp))
                        volatility_cmp = np.std(log_returns_cmp) * np.sqrt(252) * 100  # Annualized volatility in %

                        # Latest close price
                        latest_close_price_cmp = closing_prices_cmp[-1]

                        # Store in session_state
                        st.session_state['stock_data_cmp'] = pd.DataFrame({
                            'timestamp': list(stock_data_cmp.keys()),
                            'close_price': closing_prices_cmp
                        })
                        st.session_state['volatility_cmp'] = volatility_cmp
                        st.session_state['latest_close_price_cmp'] = latest_close_price_cmp

                        st.success(f"**Latest Close Price for {ticker_symbol_cmp}:** ${latest_close_price_cmp:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_cmp:.2f}%")

                        # Update input fields with fetched data
                        S_cmp = st.number_input(
                            "Comparison - Underlying Asset Price (S)",
                            min_value=0.01,
                            value=latest_close_price_cmp,
                            step=1.0,
                            format="%.2f",
                            help="Current price of the underlying asset."
                        )
                        sigma_cmp = st.number_input(
                            "Comparison - Volatility (σ) (%)",
                            min_value=0.01,
                            value=volatility_cmp,
                            step=0.1,
                            format="%.2f",
                            help="Annual volatility of the underlying asset."
                        ) / 100
                        # Update other inputs as necessary or keep them as manual inputs
                        K_cmp = st.number_input(
                            "Comparison - Strike Price (K)",
                            min_value=0.01,
                            value=100.0,
                            step=1.0,
                            format="%.2f",
                            help="Strike price of the option."
                        )
                        T_cmp = st.number_input(
                            "Comparison - Time to Maturity (T) in years",
                            min_value=0.01,
                            value=1.0,
                            step=0.1,
                            format="%.2f",
                            help="Time remaining until the option's expiration."
                        )
                        r_cmp = st.number_input(
                            "Comparison - Risk-Free Rate (r) (%)",
                            min_value=0.0,
                            value=5.0,
                            step=0.1,
                            format="%.2f",
                            help="Annual risk-free interest rate."
                        ) / 100
                        option_type_cmp = st.selectbox(
                            "Comparison - Option Type",
                            ("Call", "Put"),
                            help="Select the type of option for comparison."
                        )
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
        else:
            # Default input fields if no data is fetched
            S_cmp = st.number_input(
                "Comparison - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state.get('latest_close_price_cmp', 100.0),
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_cmp = st.number_input(
                "Comparison - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_cmp = st.number_input(
                "Comparison - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_cmp = st.number_input(
                "Comparison - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_cmp = st.number_input(
                "Comparison - Volatility (σ) (%)",
                min_value=0.01,
                value=20.0,
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_cmp = st.selectbox(
                "Comparison - Option Type",
                ("Call", "Put"),
                help="Select the type of option for comparison."
            )

    st.subheader("Compare Option Prices")
    compare_button = st.button("Compare European and American Option Prices", key="compare_cmp")
    if compare_button:
        try:
            # Ensure that both European and American data are fetched
            if st.session_state['stock_data_eu'].empty:
                st.error("Please fetch European option data before comparison.")
            if st.session_state['stock_data_am'].empty:
                st.error("Please fetch American option data before comparison.")
            if st.session_state['stock_data_eu'].empty or st.session_state['stock_data_am'].empty:
                st.stop()
            
            # Retrieve parameters for European Option
            S_eu = st.session_state.get('latest_close_price_eu', 100.0)
            K_eu = K_cmp
            T_eu = T_cmp
            r_eu = r_cmp
            sigma_eu = sigma_cmp
            option_type_eu = option_type_cmp.lower()
            
            # Retrieve parameters for American Option
            S_am = st.session_state.get('latest_close_price_am', 100.0)
            K_am = K_cmp
            T_am = T_cmp
            r_am = r_cmp
            sigma_am = sigma_cmp
            option_type_am = option_type_cmp.lower()
            M_am = 100  # Default value; consider making it dynamic if needed
            N_am = 100  # Default value; consider making it dynamic if needed
            
            # Neural Network Parameters (if applicable)
            if 'model_nn' in st.session_state:
                model_nn = st.session_state['model_nn']
                X_nn = np.array([[S_eu, K_eu, T_eu, r_eu, sigma_eu]])
                price_eu_nn = predict_price_nn(model_nn, X_nn)
            else:
                price_eu_nn = None
            
            # Calculate European Option Price using Black-Scholes
            if option_type_eu == 'call':
                price_eu_bs = black_scholes_call(S_eu, K_eu, T_eu, r_eu, sigma_eu)
            else:
                price_eu_bs = black_scholes_put(S_eu, K_eu, T_eu, r_eu, sigma_eu)
            
            # Calculate American Option Price using Crank-Nicolson
            if option_type_am == 'call':
                price_am_cn = crank_nicolson_american_option(
                    S=S_am,
                    K=K_am,
                    T=T_am,
                    r=r_am,
                    sigma=sigma_am,
                    option_type='call',
                    M=int(M_am),
                    N=int(N_am)
                )
            else:
                price_am_cn = crank_nicolson_american_option(
                    S=S_am,
                    K=K_am,
                    T=T_am,
                    r=r_am,
                    sigma=sigma_am,
                    option_type='put',
                    M=int(M_am),
                    N=int(N_am)
                )
            
            # Calculate American Option Price using FOBSM
            if option_type_am == 'call':
                price_am_fobs_m = fobs_m_option_pricing(
                    S=S_am,
                    K=K_am,
                    T=T_am,
                    r=r_am,
                    sigma=sigma_am,
                    option_type='call',
                    M=int(M_am),
                    N=int(N_am)
                )
            else:
                price_am_fobs_m = fobs_m_option_pricing(
                    S=S_am,
                    K=K_am,
                    T=T_am,
                    r=r_am,
                    sigma=sigma_am,
                    option_type='put',
                    M=int(M_am),
                    N=int(N_am)
                )
            
            # Display comparison metrics
            st.markdown("### **Comparison Results**")
            col1_cmp, col2_cmp, col3_cmp, col4_cmp = st.columns(4)
            with col1_cmp:
                st.metric("Black-Scholes Price", f"${price_eu_bs:.4f}")
            with col2_cmp:
                if price_eu_nn is not None:
                    st.metric("Neural Network Price", f"${price_eu_nn:.4f}")
                else:
                    st.metric("Neural Network Price", "N/A")
            with col3_cmp:
                st.metric("Crank-Nicolson Price", f"${price_am_cn:.4f}")
            with col4_cmp:
                st.metric("FOBSM Price", f"${price_am_fobs_m:.4f}")

            # Plot comparison
            st.markdown("### **Price Comparison Visualization**")
            fig_cmp, ax_cmp = plt.subplots(figsize=(10, 6))
            models = ['Black-Scholes (European)', 'Neural Network (European)', 'Crank-Nicolson (American)', 'FOBSM (American)']
            prices = [price_eu_bs, price_eu_nn if price_eu_nn is not None else 0, price_am_cn, price_am_fobs_m]
            colors = ['blue', 'green', 'orange', 'red']
            ax_cmp.bar(models, prices, color=colors)
            ax_cmp.set_title("Option Price Comparison Across Models")
            ax_cmp.set_ylabel("Option Price ($)")
            for i, v in enumerate(prices):
                ax_cmp.text(i, v + max(prices)*0.01, f"${v:.4f}", ha='center', va='bottom', color='black')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_cmp)
   
            # Optionally, add further visualizations or statistical comparisons
        except Exception as e:
            st.error(f"An error occurred during comparison: {e}")


if __name__ == "__main__":
    main()
