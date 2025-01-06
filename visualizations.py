# OptionTrading/visualizations.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pricing_models import black_scholes_call, black_scholes_put

def generate_heatmap(param1, param2, S, K, T, r, sigma, option_type='call', range_percent=0.5, num_points=50):
    """
    Generates a heatmap showing how the option price varies with two parameters.
    
    Parameters:
        param1 (str): First parameter to vary (X-axis).
        param2 (str): Second parameter to vary (Y-axis).
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
        range_percent (float): Percentage range to vary parameters.
        num_points (int): Number of points in each dimension.
    
    Returns:
        matplotlib.figure.Figure: The generated heatmap.
    """
    params = {
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma
    }
    
    # Validate parameters
    if param1 not in params or param2 not in params:
        raise ValueError("Invalid parameters selected for heatmap.")
    
    # Create ranges
    param1_vals = np.linspace(params[param1] * (1 - range_percent / 100),
                              params[param1] * (1 + range_percent / 100),
                              num_points)
    param2_vals = np.linspace(params[param2] * (1 - range_percent / 100),
                              params[param2] * (1 + range_percent / 100),
                              num_points)
    
    # Initialize grid
    option_prices = np.zeros((num_points, num_points))
    
    # Calculate option prices over the grid
    for i, val1 in enumerate(param1_vals):
        for j, val2 in enumerate(param2_vals):
            temp_params = params.copy()
            temp_params[param1] = val1
            temp_params[param2] = val2
            if option_type == 'call':
                price = black_scholes_call(temp_params['S'], temp_params['K'], temp_params['T'],
                                           temp_params['r'], temp_params['sigma'])
            else:
                price = black_scholes_put(temp_params['S'], temp_params['K'], temp_params['T'],
                                          temp_params['r'], temp_params['sigma'])
            option_prices[i, j] = price
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(option_prices, xticklabels=np.round(param2_vals, 2),
                yticklabels=np.round(param1_vals, 2), cmap="YlGnBu")
    plt.xlabel(param2)
    plt.ylabel(param1)
    plt.title(f"Heatmap of {option_type.capitalize()} Option Prices")
    plt.tight_layout()
    
    return plt

def plot_training_history(history):
    """
    Plots training loss over epochs.
    
    Parameters:
        history (dict): Training history containing loss values.
    
    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=np.arange(1, len(history['loss'])+1), y=history['loss'], label='Training Loss')
    # If validation loss is available, plot it as well
    if 'val_loss' in history and history['val_loss']:
        sns.lineplot(x=np.arange(1, len(history['val_loss'])+1), y=history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    return plt

def plot_prediction_vs_actual(actual, predicted):
    """
    Plots predicted option prices against actual prices.
    
    Parameters:
        actual (np.ndarray): Actual option prices.
        predicted (np.ndarray): Predicted option prices by the Neural Network.
    
    Returns:
        matplotlib.figure.Figure: The generated scatter plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=actual, y=predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')  # Diagonal line
    plt.title('Neural Network Predictions vs Actual Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.tight_layout()
    return plt
