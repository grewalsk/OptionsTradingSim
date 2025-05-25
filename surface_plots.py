# OptionTrading/surface_plots.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from typing import Tuple, Dict, List, Optional, Union
from pricing_models import black_scholes_call, black_scholes_put
from scipy.optimize import brentq
import seaborn as sns

def plot_option_price_surface(S: float, K: float, r: float, option_type: str = 'call',
                             sigma_range: Tuple[float, float] = (0.05, 0.6), 
                             T_range: Tuple[float, float] = (0.05, 2.0),
                             sigma_points: int = 20, T_points: int = 20, alpha: float = 0.8) -> plt.Figure:
    """
    Generate a 3D surface plot of option price vs. volatility vs. time to maturity.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        option_type (str): 'call' or 'put'.
        sigma_range (tuple): (min, max) range for volatility.
        T_range (tuple): (min, max) range for time to maturity.
        sigma_points (int): Number of volatility points.
        T_points (int): Number of time points.
        alpha (float): Transparency value for the 3D surface (0.0 to 1.0).
        
    Returns:
        matplotlib.figure.Figure: The generated 3D surface plot.
    """
    start_time = time.time()
    
    # Create ranges for volatility and time
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], sigma_points)
    T_vals = np.linspace(T_range[0], T_range[1], T_points)
    
    # Create meshgrid
    X, Y = np.meshgrid(sigma_vals, T_vals)
    Z = np.zeros_like(X)
    
    # Calculate option prices over the grid
    for i in range(sigma_points):
        for j in range(T_points):
            sigma = X[j, i]
            T = Y[j, i]
            if option_type.lower() == 'call':
                Z[j, i] = black_scholes_call(S, K, T, r, sigma)
            else:
                Z[j, i] = black_scholes_put(S, K, T, r, sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with enhanced visual appearance
    cmap = plt.cm.viridis
    surface = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=alpha,
                           linewidth=0, antialiased=True, shade=True)
    
    # Add grid and improve visual appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    
    # Set axis labels
    ax.set_xlabel('Volatility (σ)')
    ax.set_ylabel('Time to Maturity (T) in years')
    ax.set_zlabel('Option Price ($)')
    ax.set_title(f'{option_type.capitalize()} Option Price Surface')
    
    # Add color bar
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, label='Option Price ($)')
    
    # Add a reference point for current vol and time
    sigma_mid = (sigma_range[0] + sigma_range[1]) / 2
    T_mid = (T_range[0] + T_range[1]) / 2
    
    # Calculate price at reference point
    ref_price = black_scholes_call(S, K, T_mid, r, sigma_mid) if option_type.lower() == 'call' else black_scholes_put(S, K, T_mid, r, sigma_mid)
    
    # Add reference point
    ax.scatter([sigma_mid], [T_mid], [ref_price], 
              color='red', s=100, marker='o', label='Reference Point')
    
    # Add legend
    ax.legend()
    
    # Compute and display key information
    plt.figtext(0.02, 0.02, 
               f"S=${S:.2f}, K=${K:.2f}, r={r*100:.2f}%\n"
               f"Reference Point: σ={sigma_mid:.2f}, T={T_mid:.2f}yrs, Price=${ref_price:.2f}",
               ha="left", fontsize=9)
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.98, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="right", fontsize=8, color='gray')
    
    return fig

def plot_option_price_contour(S: float, K: float, r: float, option_type: str = 'call',
                             sigma_range: Tuple[float, float] = (0.05, 0.6), 
                             T_range: Tuple[float, float] = (0.05, 2.0),
                             sigma_points: int = 50, T_points: int = 50) -> plt.Figure:
    """
    Generate a contour plot of option price vs. volatility vs. time to maturity.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        option_type (str): 'call' or 'put'.
        sigma_range (tuple): (min, max) range for volatility.
        T_range (tuple): (min, max) range for time to maturity.
        sigma_points (int): Number of volatility points.
        T_points (int): Number of time points.
        
    Returns:
        matplotlib.figure.Figure: The generated contour plot.
    """
    start_time = time.time()
    
    # Create ranges for volatility and time
    sigma_vals = np.linspace(sigma_range[0], sigma_range[1], sigma_points)
    T_vals = np.linspace(T_range[0], T_range[1], T_points)
    
    # Create meshgrid
    X, Y = np.meshgrid(sigma_vals, T_vals)
    Z = np.zeros_like(X)
    
    # Calculate option prices over the grid
    for i in range(sigma_points):
        for j in range(T_points):
            sigma = X[j, i]
            T = Y[j, i]
            if option_type.lower() == 'call':
                Z[j, i] = black_scholes_call(S, K, T, r, sigma)
            else:
                Z[j, i] = black_scholes_put(S, K, T, r, sigma)
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot filled contours with improved visuals
    levels = 20  # Number of contour levels
    contour = ax.contourf(X, Y, Z, levels, cmap='viridis')
    
    # Add contour lines with labels
    contour_lines = ax.contour(X, Y, Z, 10, colors='white', alpha=0.5, linewidths=0.8)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Add grid and improve appearance
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Volatility (σ)')
    ax.set_ylabel('Time to Maturity (T) in years')
    ax.set_title(f'{option_type.capitalize()} Option Price Contour Map')
    
    # Add color bar
    cbar = fig.colorbar(contour, ax=ax, label='Option Price ($)')
    
    # Add reference point
    sigma_mid = (sigma_range[0] + sigma_range[1]) / 2
    T_mid = (T_range[0] + T_range[1]) / 2
    
    # Calculate price at reference point
    ref_price = black_scholes_call(S, K, T_mid, r, sigma_mid) if option_type.lower() == 'call' else black_scholes_put(S, K, T_mid, r, sigma_mid)
    
    # Add reference point
    ax.scatter([sigma_mid], [T_mid], color='red', s=100, marker='o', label=f'Reference: ${ref_price:.2f}')
    ax.legend(loc='best')
    
    # Compute and display key information
    plt.figtext(0.02, 0.02, 
               f"S=${S:.2f}, K=${K:.2f}, r={r*100:.2f}%\n"
               f"Reference: σ={sigma_mid:.2f}, T={T_mid:.2f}yrs, Price=${ref_price:.2f}",
               ha="left", fontsize=9)
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.98, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="right", fontsize=8, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for text at bottom
    return fig

def plot_implied_volatility_surface(S: float, K_array: np.ndarray, T_array: np.ndarray, 
                                   r: float, market_prices: np.ndarray, option_type: str = 'call') -> plt.Figure:
    """
    Generate a 3D surface plot of implied volatility.
    
    Parameters:
        S (float): Current stock price.
        K_array (np.ndarray): Array of strike prices.
        T_array (np.ndarray): Array of times to maturity.
        r (float): Risk-free interest rate.
        market_prices (np.ndarray): 2D array of option prices (shape must match K_array × T_array).
        option_type (str): 'call' or 'put'.
        
    Returns:
        matplotlib.figure.Figure: The generated 3D implied volatility surface.
    """
    start_time = time.time()
    
    # Function to solve for implied volatility
    def implied_vol_objective(sigma, S, K, T, r, market_price, opt_type):
        if opt_type.lower() == 'call':
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_put(S, K, T, r, sigma)
        return model_price - market_price
    
    # Create meshgrid from K and T arrays
    K_mesh, T_mesh = np.meshgrid(K_array, T_array)
    Z = np.zeros_like(K_mesh)
    
    # Calculate implied volatility over the grid
    for i in range(len(T_array)):
        for j in range(len(K_array)):
            K = K_mesh[i, j]
            T = T_mesh[i, j]
            market_price = market_prices[i, j]
            
            try:
                # Solve for implied volatility
                implied_vol = brentq(
                    implied_vol_objective, 0.001, 2.0, 
                    args=(S, K, T, r, market_price, option_type),
                    maxiter=100,
                    rtol=1e-4
                )
                Z[i, j] = implied_vol
            except:
                # If solution fails, set to NaN
                Z[i, j] = np.nan
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with enhanced visual appearance
    # First, mask the NaN values
    masked_Z = np.ma.masked_invalid(Z)
    
    # Plot the surface
    cmap = plt.cm.plasma
    surface = ax.plot_surface(K_mesh, T_mesh, masked_Z, cmap=cmap, edgecolor='none', 
                           alpha=0.8, linewidth=0, antialiased=True, shade=True)
    
    # Improve axis appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Maturity (T) in years')
    ax.set_zlabel('Implied Volatility (σ)')
    ax.set_title(f'Implied Volatility Surface ({option_type.capitalize()} Options)')
    
    # Add color bar
    cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Implied Volatility (σ)')
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.98, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="right", fontsize=8, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for text at bottom
    return fig

def plot_option_price_vs_strike_expiry(S: float, K_array: np.ndarray, T_array: np.ndarray,
                                      r: float, sigma: float, option_type: str = 'call') -> plt.Figure:
    """
    Generate a 3D surface plot of option price vs. strike price vs. time to maturity.
    
    Parameters:
        S (float): Current stock price.
        K_array (np.ndarray): Array of strike prices.
        T_array (np.ndarray): Array of times to maturity.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        option_type (str): 'call' or 'put'.
        
    Returns:
        matplotlib.figure.Figure: The generated 3D surface plot.
    """
    start_time = time.time()
    
    # Create meshgrid from K and T arrays
    K_mesh, T_mesh = np.meshgrid(K_array, T_array)
    Z = np.zeros_like(K_mesh)
    
    # Calculate option prices over the grid
    for i in range(len(T_array)):
        for j in range(len(K_array)):
            K = K_mesh[i, j]
            T = T_mesh[i, j]
            
            if option_type.lower() == 'call':
                Z[i, j] = black_scholes_call(S, K, T, r, sigma)
            else:
                Z[i, j] = black_scholes_put(S, K, T, r, sigma)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface with enhanced visual appearance
    cmap = plt.cm.viridis
    surface = ax.plot_surface(K_mesh, T_mesh, Z, cmap=cmap, edgecolor='none', 
                           alpha=0.8, linewidth=0, antialiased=True, shade=True)
    
    # Improve axis appearance
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)
    
    # Mark the current stock price with a line
    K_idx = np.abs(K_array - S).argmin()
    if 0 <= K_idx < len(K_array):
        x_vals = np.ones(len(T_array)) * K_array[K_idx]
        ax.plot(x_vals, T_array, Z[:, K_idx], 'r-', linewidth=2, label=f'S = ${S:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Time to Maturity (T) in years')
    ax.set_zlabel('Option Price ($)')
    ax.set_title(f'{option_type.capitalize()} Option Price vs Strike vs Maturity')
    
    # Add color bar
    cbar = fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Option Price ($)')
    
    # Display calculation parameters
    plt.figtext(0.02, 0.02, 
               f"S=${S:.2f}, r={r*100:.2f}%, σ={sigma*100:.2f}%",
               ha="left", fontsize=9)
    
    # Add legend
    ax.legend()
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.98, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="right", fontsize=8, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for text at bottom
    return fig

def generate_synthetic_market_prices(S: float, K_array: np.ndarray, T_array: np.ndarray,
                                   r: float, base_sigma: float, smile_factor: float = 0.2,
                                   term_structure_factor: float = 0.1, option_type: str = 'call') -> np.ndarray:
    """
    Generate synthetic market prices with volatility smile/skew and term structure.
    
    Parameters:
        S (float): Current stock price.
        K_array (np.ndarray): Array of strike prices.
        T_array (np.ndarray): Array of times to maturity.
        r (float): Risk-free interest rate.
        base_sigma (float): Base volatility.
        smile_factor (float): Factor to control volatility smile/skew strength.
        term_structure_factor (float): Factor to control term structure effect.
        option_type (str): 'call' or 'put'.
        
    Returns:
        np.ndarray: 2D array of synthetic market prices.
    """
    # Create meshgrid
    K_mesh, T_mesh = np.meshgrid(K_array, T_array)
    prices = np.zeros_like(K_mesh)
    
    # Moneyness (K/S)
    moneyness = K_mesh / S
    
    # Volatility surface with smile/skew and term structure
    # Smile effect: higher volatility for deep ITM and OTM options
    # Term structure: shorter maturities have higher volatility
    implied_vols = base_sigma * (1 + smile_factor * (moneyness - 1)**2 - term_structure_factor * T_mesh)
    
    # Ensure volatility is positive
    implied_vols = np.maximum(implied_vols, 0.01)
    
    # Calculate prices using Black-Scholes
    for i in range(len(T_array)):
        for j in range(len(K_array)):
            if option_type.lower() == 'call':
                prices[i, j] = black_scholes_call(S, K_array[j], T_array[i], r, implied_vols[i, j])
            else:
                prices[i, j] = black_scholes_put(S, K_array[j], T_array[i], r, implied_vols[i, j])
    
    return prices