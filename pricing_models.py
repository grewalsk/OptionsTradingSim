# OptionTrading/pricing_models.py

import numpy as np
from scipy.stats import norm
import pandas as pd
import streamlit as st

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call option.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
    
    Returns:
        float: Call option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European put option.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
    
    Returns:
        float: Put option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def crank_nicolson_american_option(S, K, T, r, sigma, option_type='call', M=100, N=100):
    """
    Prices an American option using the Crank-Nicolson numerical scheme.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
        option_type (str): 'call' or 'put'.
        M (int): Number of asset price steps.
        N (int): Number of time steps.
    
    Returns:
        float: Option price.
    """
    dt = T / N
    dS = (2 * S) / M
    grid_size = (M + 1, N + 1)
    V = np.zeros(grid_size)
    S_values = np.linspace(0, 2 * S, M + 1)
    
    # Boundary conditions
    if option_type == 'call':
        V[:, -1] = np.maximum(S_values - K, 0)
        V[0, :] = 0
        V[-1, :] = 2 * S - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
    else:
        V[:, -1] = np.maximum(K - S_values, 0)
        V[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        V[-1, :] = 0
    
    # Coefficients
    alpha = 0.25 * dt * (sigma ** 2 * (np.arange(M + 1) ** 2) - r * np.arange(M + 1))
    beta = -dt * 0.5 * (sigma ** 2 * (np.arange(M + 1) ** 2) + r)
    gamma = 0.25 * dt * (sigma ** 2 * (np.arange(M + 1) ** 2) + r * np.arange(M + 1))
    
    # Iterating over time steps backwards
    for t in range(N - 1, -1, -1):
        # Setup tridiagonal matrix
        A = np.zeros((M - 1, M - 1))
        B = np.zeros(M - 1)
        for i in range(1, M):
            A[i - 1, i - 1] = 1 - beta[i]
            if i != 1:
                A[i - 1, i - 2] = -alpha[i]
            if i != M - 1:
                A[i - 1, i] = -gamma[i]
            B[i - 1] = alpha[i] * V[i - 1, t + 1] + (1 + beta[i]) * V[i, t + 1] + gamma[i] * V[i + 1, t + 1]
        # Solve the system
        try:
            V_inner = np.linalg.solve(A, B)
            V[1:M, t] = V_inner
            # Early exercise condition
            if option_type == 'call':
                V[:, t] = np.maximum(V[:, t], S_values - K)
            else:
                V[:, t] = np.maximum(V[:, t], K - S_values)
        except np.linalg.LinAlgError:
            st.error("Linear algebra error during Crank-Nicolson iteration.")
            break
    # Interpolate to find the option price for S
    option_price = np.interp(S, S_values, V[:, 0])
    return option_price

def fobs_m_option_pricing(S, K, T, r, sigma, option_type='call', M=100, N=100):
    """
    Prices an American option using the Finite Optimal Boundary Schemes Method (FOBSM).
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
        option_type (str): 'call' or 'put'.
        M (int): Number of asset price steps.
        N (int): Number of time steps.
    
    Returns:
        float: Option price.
    """
    # **Implement the actual FOBSM method here based on your reference paper**
    # The following is a placeholder and should be replaced with the correct implementation
    
    dt = T / N
    dS = (2 * S) / M
    grid_size = (M + 1, N + 1)
    V = np.zeros(grid_size)
    S_values = np.linspace(0, 2 * S, M + 1)
    
    # Boundary conditions
    if option_type == 'call':
        V[:, -1] = np.maximum(S_values - K, 0)
        V[0, :] = 0
        V[-1, :] = 2 * S - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
    else:
        V[:, -1] = np.maximum(K - S_values, 0)
        V[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        V[-1, :] = 0
    
    # Coefficients (Assuming similar to Crank-Nicolson with FOBSM enhancements)
    alpha = 0.25 * dt * (sigma ** 2 * (np.arange(M + 1) ** 2) - r * np.arange(M + 1))
    beta = -dt * 0.5 * (sigma ** 2 * (np.arange(M + 1) ** 2) + r)
    gamma = 0.25 * dt * (sigma ** 2 * (np.arange(M + 1) ** 2) + r * np.arange(M + 1))
    
    # Iterating over time steps backwards with FOBSM enhancements
    for t in range(N - 1, -1, -1):
        # Setup tridiagonal matrix with FOBSM enhancements
        A = np.zeros((M - 1, M - 1))
        B = np.zeros(M - 1)
        for i in range(1, M):
            A[i - 1, i - 1] = 1 - beta[i]
            if i != 1:
                A[i - 1, i - 2] = -alpha[i]
            if i != M - 1:
                A[i - 1, i] = -gamma[i]
            B[i - 1] = alpha[i] * V[i - 1, t + 1] + (1 + beta[i]) * V[i, t + 1] + gamma[i] * V[i + 1, t + 1]
        # Solve the system
        try:
            V_inner = np.linalg.solve(A, B)
            V[1:M, t] = V_inner
            # Early exercise condition with FOBSM adjustments
            if option_type == 'call':
                V[:, t] = np.maximum(V[:, t], S_values - K)
            else:
                V[:, t] = np.maximum(V[:, t], K - S_values)
        except np.linalg.LinAlgError:
            st.error("Linear algebra error during FOBSM iteration.")
            break
    # Interpolate to find the option price for S
    option_price = np.interp(S, S_values, V[:, 0])
    return option_price

def monte_carlo_simulation(S, K, T, r, sigma, option_type='call', num_simulations=1000, confidence_level=0.95):
    """
    Performs Monte Carlo simulation to estimate the option price and confidence interval.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate (annual).
        sigma (float): Volatility of the underlying asset (annual).
        option_type (str): 'call' or 'put'.
        num_simulations (int): Number of simulation paths.
        confidence_level (float): Confidence level for the interval (e.g., 0.95 for 95%).
    
    Returns:
        tuple: (lower_bound, upper_bound, simulated_prices)
    """
    try:
        # Generate random paths using Geometric Brownian Motion
        Z = np.random.standard_normal(num_simulations)
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate option payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount payoffs back to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate statistics
        price_estimate = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)
        lower_bound = price_estimate - z_score * std_error
        upper_bound = price_estimate + z_score * std_error
        
        return lower_bound, upper_bound, discounted_payoffs
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation: {e}")
        return None, None, None
