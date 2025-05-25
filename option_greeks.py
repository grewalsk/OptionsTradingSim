# option_greeks.py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
import time
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple, List, Optional

def calculate_option_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                          option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho) for given parameters.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
        
    Returns:
        dict: Dictionary containing values for all Greeks.
    """
    # Calculate d1 and d2 for Black-Scholes
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate Greeks
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    # Common for both call and put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # Scaled by 0.01 (1% change in vol)
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'call' else -norm.cdf(-d2)) * 0.01  # Scaled by 0.01
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Daily theta
        'vega': vega,
        'rho': rho
    }

def plot_option_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                      option_type: str = 'call', greek_type: str = 'delta', 
                      param_to_vary: str = 'S', range_percent: float = 20, 
                      num_points: int = 50) -> plt.Figure:
    """
    Plot how a specific option Greek changes with respect to a varying parameter.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
        greek_type (str): 'delta', 'gamma', 'theta', 'vega', or 'rho'.
        param_to_vary (str): Parameter to vary ('S', 'T', 'r', 'sigma').
        range_percent (float): Percentage range to vary the parameter.
        num_points (int): Number of points to calculate.
        
    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    start_time = time.time()
    
    params = {
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma
    }
    
    # Create range of values for the parameter to vary
    min_val = params[param_to_vary] * (1 - range_percent / 100)
    max_val = params[param_to_vary] * (1 + range_percent / 100)
    
    # Handle special cases
    if param_to_vary == 'T' and min_val <= 0:
        min_val = 0.01  # Avoid zero or negative time to maturity
    if param_to_vary == 'sigma' and min_val <= 0:
        min_val = 0.01  # Avoid zero or negative volatility
        
    param_values = np.linspace(min_val, max_val, num_points)
    
    # Calculate greeks for each value
    greek_values = []
    for val in param_values:
        temp_params = params.copy()
        temp_params[param_to_vary] = val
        greeks = calculate_option_greeks(temp_params['S'], temp_params['K'], 
                                       temp_params['T'], temp_params['r'], 
                                       temp_params['sigma'], option_type)
        greek_values.append(greeks[greek_type])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_values, greek_values, linewidth=2.5)
    
    # Format x-axis based on parameter type
    if param_to_vary == 'r' or param_to_vary == 'sigma':
        # Display as percentage
        ax.set_xlabel(f"{param_to_vary} (%)")
        tick_labels = [f"{x*100:.1f}%" for x in param_values[::len(param_values)//5]]
        ax.set_xticks(param_values[::len(param_values)//5])
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xlabel(f"{param_to_vary}")
    
    # Add grid, title, etc.
    ax.set_ylabel(f"{greek_type.capitalize()}")
    ax.set_title(f"{greek_type.capitalize()} vs {param_to_vary} for {option_type.capitalize()} Option")
    ax.grid(True, alpha=0.3)
    
    # Add current parameter value as vertical line
    ax.axvline(x=params[param_to_vary], color='red', linestyle='--', 
              alpha=0.7, label=f"Current {param_to_vary} = {params[param_to_vary]}")
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Add annotation with current greek value
    current_greek_idx = np.abs(param_values - params[param_to_vary]).argmin()
    current_greek = greek_values[current_greek_idx]
    ax.scatter([params[param_to_vary]], [current_greek], color='red', s=80, zorder=5)
    ax.annotate(f"{greek_type.capitalize()} = {current_greek:.4f}", 
               xy=(params[param_to_vary], current_greek),
               xytext=(10, 10), textcoords='offset points',
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    ax.legend()
    plt.tight_layout()
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.02, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="left", fontsize=8, color='gray')
    
    return fig

def plot_option_greek_surface(S: float, K: float, T: float, r: float, sigma: float, 
                             option_type: str = 'call', greek_type: str = 'delta',
                             param1: str = 'S', param2: str = 'T',
                             range_percent: float = 20, num_points: int = 20) -> plt.Figure:
    """
    Create a 3D surface plot showing how a Greek varies with two parameters.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
        greek_type (str): 'delta', 'gamma', 'theta', 'vega', or 'rho'.
        param1 (str): First parameter to vary (X-axis).
        param2 (str): Second parameter to vary (Y-axis).
        range_percent (float): Percentage range to vary parameters.
        num_points (int): Number of points in each dimension.
        
    Returns:
        matplotlib.figure.Figure: The generated 3D surface plot.
    """
    start_time = time.time()
    
    params = {
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma
    }
    
    # Check that two different parameters are selected
    if param1 == param2:
        raise ValueError("param1 and param2 must be different parameters")
    
    # Create ranges for both parameters with appropriate minimum values
    min_val1 = max(0.01, params[param1] * (1 - range_percent / 100))
    max_val1 = params[param1] * (1 + range_percent / 100)
    param1_vals = np.linspace(min_val1, max_val1, num_points)
    
    min_val2 = max(0.01, params[param2] * (1 - range_percent / 100))
    max_val2 = params[param2] * (1 + range_percent / 100)
    param2_vals = np.linspace(min_val2, max_val2, num_points)
    
    # Create meshgrid
    X, Y = np.meshgrid(param1_vals, param2_vals)
    Z = np.zeros_like(X)
    
    # Calculate greeks over the grid
    for i in range(num_points):
        for j in range(num_points):
            temp_params = params.copy()
            temp_params[param1] = X[i, j]
            temp_params[param2] = Y[i, j]
            
            greeks = calculate_option_greeks(
                temp_params['S'], temp_params['K'], 
                temp_params['T'], temp_params['r'], 
                temp_params['sigma'], option_type
            )
            Z[i, j] = greeks[greek_type]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add grid and improve visual appearance
    cmap = plt.cm.viridis
    surface = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.8,
                           linewidth=0, antialiased=True, shade=True)
    
    # Format labels appropriately
    param1_label = f"{param1}" if param1 not in ['r', 'sigma'] else f"{param1} (%)"
    param2_label = f"{param2}" if param2 not in ['r', 'sigma'] else f"{param2} (%)"
    
    # Axis labels
    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    ax.set_zlabel(f"{greek_type.capitalize()}")
    
    # Add title
    ax.set_title(f"{greek_type.capitalize()} Surface for {option_type.capitalize()} Option")
    
    # Add color bar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label=f"{greek_type.capitalize()} Value")
    
    # Mark current parameter values with a point
    ax.scatter(
        [params[param1]], [params[param2]], 
        [calculate_option_greeks(S, K, T, r, sigma, option_type)[greek_type]],
        color='red', s=100, marker='o'
    )
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.02, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="left", fontsize=8, color='gray')
    
    return fig

def generate_greeks_table(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str = 'call') -> pd.DataFrame:
    """
    Generate a table of Greeks for various spot prices around the current spot.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
        
    Returns:
        pd.DataFrame: Table of Greeks for different spot prices.
    """
    # Generate spot prices around current spot
    spot_changes = [-10, -5, -2, 0, 2, 5, 10]  # Percent changes
    spots = [S * (1 + pct/100) for pct in spot_changes]
    
    data = {
        'Spot Price': spots,
        'Price Change (%)': spot_changes,
        'Delta': [],
        'Gamma': [],
        'Theta': [],
        'Vega': [],
        'Rho': []
    }
    
    # Calculate Greeks for each spot price
    for spot in spots:
        greeks = calculate_option_greeks(spot, K, T, r, sigma, option_type)
        data['Delta'].append(greeks['delta'])
        data['Gamma'].append(greeks['gamma'])
        data['Theta'].append(greeks['theta'])
        data['Vega'].append(greeks['vega'])
        data['Rho'].append(greeks['rho'])
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Format columns
    df['Spot Price'] = df['Spot Price'].map(lambda x: f"${x:.2f}")
    df['Delta'] = df['Delta'].map(lambda x: f"{x:.4f}")
    df['Gamma'] = df['Gamma'].map(lambda x: f"{x:.4f}")
    df['Theta'] = df['Theta'].map(lambda x: f"${x:.4f}")
    df['Vega'] = df['Vega'].map(lambda x: f"${x:.4f}")
    df['Rho'] = df['Rho'].map(lambda x: f"${x:.4f}")
    
    return df

def plot_greeks_comparison(S: float, K: float, T: float, r: float, sigma: float,
                         param_to_vary: str = 'S', range_percent: float = 20,
                         num_points: int = 50) -> plt.Figure:
    """
    Plot all Greeks together for comparison as the selected parameter varies.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        param_to_vary (str): Parameter to vary ('S', 'T', 'r', 'sigma').
        range_percent (float): Percentage range to vary the parameter.
        num_points (int): Number of points to calculate.
        
    Returns:
        matplotlib.figure.Figure: The generated plot with subplots for each Greek.
    """
    params = {
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma
    }
    
    # Create range of values for the parameter to vary
    min_val = max(0.01, params[param_to_vary] * (1 - range_percent / 100))
    max_val = params[param_to_vary] * (1 + range_percent / 100)
    param_values = np.linspace(min_val, max_val, num_points)
    
    # Option types to compare
    option_types = ['call', 'put']
    greek_types = ['delta', 'gamma', 'theta', 'vega', 'rho']
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(greek_types), 1, figsize=(12, 15), sharex=True)
    
    # Set colors for option types
    colors = {'call': 'blue', 'put': 'red'}
    
    # Calculate and plot Greeks for each option type
    for option_type in option_types:
        greek_values = {greek: [] for greek in greek_types}
        
        for val in param_values:
            temp_params = params.copy()
            temp_params[param_to_vary] = val
            
            greeks = calculate_option_greeks(
                temp_params['S'], temp_params['K'], 
                temp_params['T'], temp_params['r'], 
                temp_params['sigma'], option_type
            )
            
            for greek in greek_types:
                greek_values[greek].append(greeks[greek])
        
        # Plot each Greek in its subplot
        for i, greek in enumerate(greek_types):
            axes[i].plot(param_values, greek_values[greek], 
                       color=colors[option_type], linewidth=2,
                       label=f"{option_type.capitalize()}")
            
            # Add reference line at current parameter value
            if i == 0:  # Only add label to the first occurrence
                axes[i].axvline(x=params[param_to_vary], color='black', linestyle='--', 
                              alpha=0.5, label=f"Current {param_to_vary}")
            else:
                axes[i].axvline(x=params[param_to_vary], color='black', linestyle='--', alpha=0.5)
            
            # Add horizontal line at y=0
            axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.2)
            
            # Add grid and labels
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylabel(greek.capitalize())
            axes[i].set_title(f"{greek.capitalize()} vs {param_to_vary}")
            axes[i].legend()
    
    # Format x-axis based on parameter type
    if param_to_vary in ['r', 'sigma']:
        # Display as percentage
        x_label = f"{param_to_vary} (%)"
        tick_labels = [f"{x*100:.1f}%" for x in param_values[::len(param_values)//5]]
        axes[-1].set_xticks(param_values[::len(param_values)//5])
        axes[-1].set_xticklabels(tick_labels)
    else:
        x_label = param_to_vary
    
    axes[-1].set_xlabel(x_label)
    
    plt.tight_layout()
    return fig