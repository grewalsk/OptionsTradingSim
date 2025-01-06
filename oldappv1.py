# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataFetcher import fetch_intraday_stock_data
from OptionTrading.pricing_models import (
    black_scholes_call,
    black_scholes_put,
    crank_nicolson_american_option,
    monte_carlo_simulation,
    generate_heatmap
)

# Initialize session state variables if they don't exist
if 'stock_data_am' not in st.session_state:
    st.session_state['stock_data_am'] = {}
if 'stock_data_eu' not in st.session_state:
    st.session_state['stock_data_eu'] = {}
if 'volatility_eu' not in st.session_state:
    st.session_state['volatility_eu'] = 20.0  # Default value
if 'volatility_am' not in st.session_state:
    st.session_state['volatility_am'] = 20.0  # Default value
if 'latest_close_price_eu' not in st.session_state:
    st.session_state['latest_close_price_eu'] = 100.0  # Default value
if 'latest_close_price_am' not in st.session_state:
    st.session_state['latest_close_price_am'] = 100.0  # Default value

# Set Seaborn theme for consistent styling
sns.set_style("whitegrid")

def main():
    # Utilize Streamlit’s Sidebar for Inputs and Navigation
    st.set_page_config(page_title="Option Pricing Application", layout="wide")
    st.sidebar.header("Navigation")
    selected_tab = st.sidebar.radio("Go to", ["Introduction", "European Options", "American Options", "Comparison"])
    
    st.sidebar.header("Input Parameters")
    
    # Navigation through sidebar radio buttons
    if selected_tab == "Introduction":
        display_introduction()
    elif selected_tab == "European Options":
        display_european_options()
    elif selected_tab == "American Options":
        display_american_options()
    elif selected_tab == "Comparison":
        display_comparison()

def display_introduction():
    st.title("Option Pricing Application")
    st.header("Introduction")
    st.markdown("""
    ### **Project Goal**
    
    This application provides a comprehensive analysis of option pricing using the **Black-Scholes Model** and extends it to accommodate **American options** through the **Crank-Nicolson numerical scheme**. By comparing the pricing of European and American options, users can gain insights into the impact of early exercise features and understand the practical applications of these models in the financial market.
    
    ### **Why This Project is Interesting and Market Applicable**
    
    - **Informed Decision-Making:** Investors and traders can utilize these pricing models to make informed decisions about buying and selling options.
    - **Risk Management:** Understanding option pricing helps in assessing the risk and potential returns associated with different option strategies.
    - **Market Efficiency:** Accurate pricing models contribute to the overall efficiency of financial markets by ensuring options are fairly valued.
    - **Educational Value:** The application serves as an educational tool for those looking to understand the mathematical foundations of option pricing.
    
    ### **Model Limitations**
    
    - **Black-Scholes Model:**
      - **Applicability:** Designed primarily for **European-style options** which can only be exercised at expiration.
      - **Assumptions:** Assumes constant volatility and risk-free rate, log-normal distribution of asset prices, and no dividends.
      - **Limitations:** Does not account for early exercise features inherent to **American options**, especially **puts** and **calls on dividend-paying stocks**.
    
    - **Crank-Nicolson Scheme:**
      - **Purpose:** Extends the Black-Scholes Model to price **American options** by accommodating the possibility of early exercise.
      - **Complexity:** Involves numerical methods and computational resources.
      - **Accuracy:** Provides more accurate pricing for American options but requires careful implementation to ensure stability and convergence.
    """)

def display_european_options():
    st.header("European Options Pricing with Black-Scholes Model")
    st.markdown("""
        The **Black-Scholes Model** provides a mathematical framework for pricing **European-style options**. This section allows you to:
        - **Calculate Option Prices:** Using the Black-Scholes formula.
        - **Conduct Sensitivity Analysis:** Visualize how option prices respond to changes in key parameters through heatmaps.
        - **Assess Uncertainty:** Generate confidence intervals to understand the reliability of option price estimates.
        
    ### **Overview**
    
    The **Black-Scholes Model** employs partial differential equations to determine the fair price of European call and put options. By considering factors such as the underlying asset's price, strike price, time to maturity, risk-free rate, and volatility, the model provides a closed-form solution that is widely used in financial markets.
    
    **Impact:**
    - **Pricing Accuracy:** Offers precise option pricing under the assumption of efficient markets.
    - **Risk Assessment:** Facilitates the calculation of Greeks, which are essential for managing financial risk.
    - **Market Adoption:** Forms the backbone of modern financial derivatives markets and trading strategies.
    """)
    
    with st.container():
        # European Options Input Parameters

        # 1. Add a Search Bar for Ticker Symbols
        ticker_symbol_eu = st.sidebar.text_input(
            "European Option - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data."
        )

        fetch_data_button_eu = st.sidebar.button("Fetch Stock Data for European Option")

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
                        st.session_state['stock_data_eu'] = stock_data_eu
                        st.session_state['volatility_eu'] = volatility_eu
                        st.session_state['latest_close_price_eu'] = latest_close_price_eu

                        st.success(f"**Latest Close Price for {ticker_symbol_eu}:** ${latest_close_price_eu:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_eu:.2f}%")

                        # Update input fields with fetched data
                        S_eu = st.sidebar.number_input(
                            "European Option - Underlying Asset Price (S)",
                            min_value=0.01,
                            value=latest_close_price_eu,
                            step=1.0,
                            format="%.2f",
                            help="Current price of the underlying asset."
                        )
                        sigma_eu = st.sidebar.number_input(
                            "European Option - Volatility (σ) (%)",
                            min_value=0.01,
                            value=volatility_eu,
                            step=0.1,
                            format="%.2f",
                            help="Annual volatility of the underlying asset."
                        ) / 100
                        # Update other inputs as necessary or keep them as manual inputs
                        K_eu = st.sidebar.number_input(
                            "European Option - Strike Price (K)",
                            min_value=0.01,
                            value=100.0,
                            step=1.0,
                            format="%.2f",
                            help="Strike price of the option."
                        )
                        T_eu = st.sidebar.number_input(
                            "European Option - Time to Maturity (T) in years",
                            min_value=0.01,
                            value=1.0,
                            step=0.1,
                            format="%.2f",
                            help="Time remaining until the option's expiration."
                        )
                        r_eu = st.sidebar.number_input(
                            "European Option - Risk-Free Rate (r) (%)",
                            min_value=0.0,
                            value=5.0,
                            step=0.1,
                            format="%.2f",
                            help="Annual risk-free interest rate."
                        ) / 100
                        option_type_eu = st.sidebar.selectbox(
                            "European Option - Option Type",
                            ("Call", "Put"),
                            help="Select the type of European option."
                        )
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
        else:
            # Default input fields if no data is fetched
            S_eu = st.sidebar.number_input(
                "European Option - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_eu'],
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_eu = st.sidebar.number_input(
                "European Option - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_eu = st.sidebar.number_input(
                "European Option - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_eu = st.sidebar.number_input(
                "European Option - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_eu = st.sidebar.number_input(
                "European Option - Volatility (σ) (%)",
                min_value=0.01,
                value=st.session_state['volatility_eu'],
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_eu = st.sidebar.selectbox(
                "European Option - Option Type",
                ("Call", "Put"),
                help="Select the type of European option."
            )

        st.subheader("Calculate Option Price")
        # Calculate Option Price
        if st.button("Calculate European Option Price"):
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
        num_simulations_eu = st.sidebar.number_input(
            "European Option - Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of Monte Carlo simulation runs to estimate confidence intervals."
        )
        confidence_level_eu = st.sidebar.slider(
            "European Option - Confidence Level (%)",
            min_value=50,
            max_value=99,
            value=95,
            step=1,
            help="Probability that the true option price lies within the confidence interval."
        )
        
        if st.button("Run Monte Carlo Simulation for European Option"):
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
        param_labels_eu = {
            'S': 'Underlying Asset Price (S)',
            'T': 'Time to Maturity (T)',
            'r': 'Risk-Free Rate (r)',
            'sigma': 'Volatility (σ)'
        }
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
        
        if st.button("Generate Heatmap for European Option"):
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
    st.header("American Options Pricing with Crank-Nicolson Scheme")
    st.markdown("""
        The **Crank-Nicolson numerical scheme** extends the Black-Scholes Model to accommodate **American-style options**, which can be exercised at any time before expiration. This section allows you to:
        - **Calculate Option Prices:** Using the Crank-Nicolson finite difference method.
        - **Conduct Sensitivity Analysis:** Visualize how option prices respond to changes in key parameters through heatmaps.
        - **Assess Uncertainty:** Generate confidence intervals to understand the reliability of option price estimates.
        
    ### **Overview**
    
    The **Crank-Nicolson Scheme** is a finite difference method used to numerically solve the Black-Scholes Partial Differential Equation (PDE) for **American options**. Unlike European options, American options can be exercised at any time before expiration, introducing a free boundary problem that the Crank-Nicolson method effectively handles.
    
    **Impact:**
    - **Flexibility:** Allows for the pricing of options with early exercise features, making it applicable to a wider range of financial instruments.
    - **Accuracy:** Provides more precise option pricing in scenarios where early exercise is advantageous.
    - **Computational Efficiency:** Balances computational speed with accuracy, making it suitable for real-time trading applications.
    """)
    
    with st.container():
        # American Options Input Parameters

        # 1. Add a Search Bar for Ticker Symbols
        ticker_symbol_am = st.sidebar.text_input(
            "American Option - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data."
        )

        fetch_data_button_am = st.sidebar.button("Fetch Stock Data for American Option")

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
                        st.session_state['stock_data_am'] = stock_data_am
                        st.session_state['volatility_am'] = volatility_am
                        st.session_state['latest_close_price_am'] = latest_close_price_am

                        st.success(f"**Latest Close Price for {ticker_symbol_am}:** ${latest_close_price_am:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_am:.2f}%")

                        # Update input fields with fetched data
                        S_am = st.sidebar.number_input(
                            "American Option - Underlying Asset Price (S)",
                            min_value=0.01,
                            value=latest_close_price_am,
                            step=1.0,
                            format="%.2f",
                            help="Current price of the underlying asset."
                        )
                        sigma_am = st.sidebar.number_input(
                            "American Option - Volatility (σ) (%)",
                            min_value=0.01,
                            value=volatility_am,
                            step=0.1,
                            format="%.2f",
                            help="Annual volatility of the underlying asset."
                        ) / 100
                        # Update other inputs as necessary or keep them as manual inputs
                        K_am = st.sidebar.number_input(
                            "American Option - Strike Price (K)",
                            min_value=0.01,
                            value=100.0,
                            step=1.0,
                            format="%.2f",
                            help="Strike price of the option."
                        )
                        T_am = st.sidebar.number_input(
                            "American Option - Time to Maturity (T) in years",
                            min_value=0.01,
                            value=1.0,
                            step=0.1,
                            format="%.2f",
                            help="Time remaining until the option's expiration."
                        )
                        r_am = st.sidebar.number_input(
                            "American Option - Risk-Free Rate (r) (%)",
                            min_value=0.0,
                            value=5.0,
                            step=0.1,
                            format="%.2f",
                            help="Annual risk-free interest rate."
                        ) / 100
                        option_type_am = st.sidebar.selectbox(
                            "American Option - Option Type",
                            ("Call", "Put"),
                            help="Select the type of American option."
                        )
                        M_am = st.sidebar.number_input(
                            "American Option - Number of Asset Price Steps (M)",
                            min_value=50,
                            max_value=500,
                            value=100,
                            step=10,
                            help="Number of steps in the asset price dimension for the finite difference grid."
                        )
                        N_am = st.sidebar.number_input(
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
            S_am = st.sidebar.number_input(
                "American Option - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_am'],
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_am = st.sidebar.number_input(
                "American Option - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_am = st.sidebar.number_input(
                "American Option - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_am = st.sidebar.number_input(
                "American Option - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_am = st.sidebar.number_input(
                "American Option - Volatility (σ) (%)",
                min_value=0.01,
                value=st.session_state['volatility_am'],
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_am = st.sidebar.selectbox(
                "American Option - Option Type",
                ("Call", "Put"),
                help="Select the type of American option."
            )
            M_am = st.sidebar.number_input(
                "American Option - Number of Asset Price Steps (M)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of steps in the asset price dimension for the finite difference grid."
            )
            N_am = st.sidebar.number_input(
                "American Option - Number of Time Steps (N)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of steps in the time dimension for the finite difference grid."
            )

        st.subheader("Calculate Option Price")
        # Calculate Option Price
        if st.button("Calculate American Option Price"):
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
        num_simulations_am = st.sidebar.number_input(
            "American Option - Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Number of Monte Carlo simulation runs to estimate confidence intervals."
        )
        confidence_level_am = st.sidebar.slider(
            "American Option - Confidence Level (%)",
            min_value=50,
            max_value=99,
            value=95,
            step=1,
            help="Probability that the true option price lies within the confidence interval."
        )
        
        if st.button("Run Monte Carlo Simulation for American Option"):
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
        param_labels_am = {
            'S': 'Underlying Asset Price (S)',
            'T': 'Time to Maturity (T)',
            'r': 'Risk-Free Rate (r)',
            'sigma': 'Volatility (σ)'
        }
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
        
        if st.button("Generate Heatmap for American Option"):
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

def display_comparison():
    st.header("Accuracy Comparison: European vs American Options")
    st.markdown("""
    This section allows you to compare the pricing accuracy between **European Options** (Black-Scholes Model) and **American Options** (Crank-Nicolson Scheme) under identical input parameters.
    """)

    with st.container():
        # Comparison Input Parameters

        # 1. Add a Search Bar for Ticker Symbols
        ticker_symbol_cmp = st.sidebar.text_input(
            "Comparison - Stock Ticker Symbol (e.g., AAPL)",
            value="AAPL",
            help="Enter the stock ticker symbol to fetch real-time data for comparison."
        )

        fetch_data_button_cmp = st.sidebar.button("Fetch Stock Data for Comparison")

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
                        st.session_state['stock_data_cmp'] = stock_data_cmp
                        st.session_state['volatility_cmp'] = volatility_cmp
                        st.session_state['latest_close_price_cmp'] = latest_close_price_cmp

                        st.success(f"**Latest Close Price for {ticker_symbol_cmp}:** ${latest_close_price_cmp:.2f}")
                        st.write(f"**Annualized Historical Volatility:** {volatility_cmp:.2f}%")

                        # Update input fields with fetched data
                        S_cmp = st.sidebar.number_input(
                            "Comparison - Underlying Asset Price (S)",
                            min_value=0.01,
                            value=latest_close_price_cmp,
                            step=1.0,
                            format="%.2f",
                            help="Current price of the underlying asset."
                        )
                        sigma_cmp = st.sidebar.number_input(
                            "Comparison - Volatility (σ) (%)",
                            min_value=0.01,
                            value=volatility_cmp,
                            step=0.1,
                            format="%.2f",
                            help="Annual volatility of the underlying asset."
                        ) / 100
                        # Update other inputs as necessary or keep them as manual inputs
                        K_cmp = st.sidebar.number_input(
                            "Comparison - Strike Price (K)",
                            min_value=0.01,
                            value=100.0,
                            step=1.0,
                            format="%.2f",
                            help="Strike price of the option."
                        )
                        T_cmp = st.sidebar.number_input(
                            "Comparison - Time to Maturity (T) in years",
                            min_value=0.01,
                            value=1.0,
                            step=0.1,
                            format="%.2f",
                            help="Time remaining until the option's expiration."
                        )
                        r_cmp = st.sidebar.number_input(
                            "Comparison - Risk-Free Rate (r) (%)",
                            min_value=0.0,
                            value=5.0,
                            step=0.1,
                            format="%.2f",
                            help="Annual risk-free interest rate."
                        ) / 100
                        option_type_cmp = st.sidebar.selectbox(
                            "Comparison - Option Type",
                            ("Call", "Put"),
                            help="Select the type of option for comparison."
                        )
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
        else:
            # Default input fields if no data is fetched
            S_cmp = st.sidebar.number_input(
                "Comparison - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_cmp'],
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_cmp = st.sidebar.number_input(
                "Comparison - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_cmp = st.sidebar.number_input(
                "Comparison - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_cmp = st.sidebar.number_input(
                "Comparison - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_cmp = st.sidebar.number_input(
                "Comparison - Volatility (σ) (%)",
                min_value=0.01,
                value=20.0,
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_cmp = st.sidebar.selectbox(
                "Comparison - Option Type",
                ("Call", "Put"),
                help="Select the type of option for comparison."
            )

    st.subheader("Compare Option Prices")
    if st.button("Compare European and American Option Prices"):
        try:
            # Ensure that both European and American data are fetched
            if not st.session_state['stock_data_eu']:
                st.error("Please fetch European option data before comparison.")
            if not st.session_state['stock_data_am']:
                st.error("Please fetch American option data before comparison.")
            if not st.session_state['stock_data_eu'] or not st.session_state['stock_data_am']:
                st.stop()
            
            # Retrieve parameters for European Option
            S_eu = st.sidebar.number_input(
                "European Option - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_eu'],
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_eu = st.sidebar.number_input(
                "European Option - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_eu = st.sidebar.number_input(
                "European Option - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_eu = st.sidebar.number_input(
                "European Option - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_eu = st.sidebar.number_input(
                "European Option - Volatility (σ) (%)",
                min_value=0.01,
                value=st.session_state['volatility_eu'],
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_eu = st.sidebar.selectbox(
                "European Option - Option Type",
                ("Call", "Put"),
                help="Select the type of European option."
            )
            
            # Retrieve parameters for American Option
            S_am = st.sidebar.number_input(
                "American Option - Underlying Asset Price (S)",
                min_value=0.01,
                value=st.session_state['latest_close_price_am'],
                step=1.0,
                format="%.2f",
                help="Current price of the underlying asset."
            )
            K_am = st.sidebar.number_input(
                "American Option - Strike Price (K)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                format="%.2f",
                help="Strike price of the option."
            )
            T_am = st.sidebar.number_input(
                "American Option - Time to Maturity (T) in years",
                min_value=0.01,
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Time remaining until the option's expiration."
            )
            r_am = st.sidebar.number_input(
                "American Option - Risk-Free Rate (r) (%)",
                min_value=0.0,
                value=5.0,
                step=0.1,
                format="%.2f",
                help="Annual risk-free interest rate."
            ) / 100
            sigma_am = st.sidebar.number_input(
                "American Option - Volatility (σ) (%)",
                min_value=0.01,
                value=st.session_state['volatility_am'],
                step=0.1,
                format="%.2f",
                help="Annual volatility of the underlying asset."
            ) / 100
            option_type_am = st.sidebar.selectbox(
                "American Option - Option Type",
                ("Call", "Put"),
                help="Select the type of American option."
            )
            M_am = st.sidebar.number_input(
                "American Option - Number of Asset Price Steps (M)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of steps in the asset price dimension for the finite difference grid."
            )
            N_am = st.sidebar.number_input(
                "American Option - Number of Time Steps (N)",
                min_value=50,
                max_value=500,
                value=100,
                step=10,
                help="Number of steps in the time dimension for the finite difference grid."
            )
        
            # Calculate European Option Price
            if option_type_cmp.lower() == 'call':
                price_eu_cmp = black_scholes_call(S_eu, K_eu, T_eu, r_eu, sigma_eu)
            else:
                price_eu_cmp = black_scholes_put(S_eu, K_eu, T_eu, r_eu, sigma_eu)
        
            # Calculate American Option Price
            if option_type_cmp.lower() == 'call':
                price_am_cmp = crank_nicolson_american_option(
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
                price_am_cmp = crank_nicolson_american_option(
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
            col1_cmp, col2_cmp, col3_cmp = st.columns(3)
            with col1_cmp:
                st.metric(f"Black-Scholes {option_type_cmp.capitalize()} Price", f"${price_eu_cmp:.4f}")
            with col2_cmp:
                st.metric(f"Crank-Nicolson {option_type_cmp.capitalize()} Price", f"${price_am_cmp:.4f}")
            with col3_cmp:
                difference = price_am_cmp - price_eu_cmp
                percentage_diff = (difference / price_eu_cmp) * 100 if price_eu_cmp != 0 else 0
                st.metric("Price Difference", f"${difference:.4f} ({percentage_diff:.2f}%)")
    
            # Plot comparison
            st.markdown("### **Price Comparison Visualization**")
            fig_cmp, ax_cmp = plt.subplots(figsize=(6, 4))
            models = ['Black-Scholes (European)', 'Crank-Nicolson (American)']
            prices = [price_eu_cmp, price_am_cmp]
            colors = ['blue', 'orange']
            ax_cmp.bar(models, prices, color=colors)
            ax_cmp.set_title(f"{option_type_cmp.capitalize()} Option Price Comparison")
            ax_cmp.set_ylabel("Option Price ($)")
            for i, v in enumerate(prices):
                ax_cmp.text(i, v + max(prices)*0.01, f"${v:.4f}", ha='center', va='bottom', color='black')
            ax_cmp.legend([f"{option_type_cmp.capitalize()} Price"], loc='upper left')
            plt.tight_layout()
            st.pyplot(fig_cmp)
        except Exception as e:
            st.error(f"An error occurred during comparison: {e}")

if __name__ == "__main__":
    main()
