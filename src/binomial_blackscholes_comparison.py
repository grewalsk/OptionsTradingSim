# binomial_blackscholes_comparison.py

import numpy as np
from scipy.stats import norm

def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    """
    Calculates the price of an option using the Binomial Options Pricing Model.

    Parameters:
    - S (float): Current price of the underlying asset
    - K (float): Strike price of the option
    - T (float): Time to maturity (in years)
    - r (float): Risk-free interest rate (annual)
    - sigma (float): Volatility of the underlying asset (annual)
    - N (int): Number of steps in the binomial tree
    - option_type (str): 'call' or 'put'

    Returns:
    - float: Option price
    """
    # Calculate time step
    dt = T / N
    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # Calculate risk-neutral probabilities
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset prices at maturity
    asset_prices = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    
    # Initialize option values at maturity
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    elif option_type == 'put':
        option_values = np.maximum(K - asset_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Step backwards through the tree
    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:i+2] + (1 - p) * option_values[0:i+1])
    
    return option_values[0]

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call option.

    Parameters:
    - S (float): Current price of the underlying asset
    - K (float): Strike price of the option
    - T (float): Time to maturity (in years)
    - r (float): Risk-free interest rate (annual)
    - sigma (float): Volatility of the underlying asset (annual)

    Returns:
    - float: Black-Scholes call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European put option.

    Parameters:
    - S (float): Current price of the underlying asset
    - K (float): Strike price of the option
    - T (float): Time to maturity (in years)
    - r (float): Risk-free interest rate (annual)
    - sigma (float): Volatility of the underlying asset (annual)

    Returns:
    - float: Black-Scholes put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T ) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

def main():
    # Parameters
    S = 100        # Current asset price
    K = 100        # Strike price
    T = 1          # Time to maturity (1 year)
    r = 0.05       # Risk-free rate (5%)
    sigma = 0.2    # Volatility (20%)
    N = 100        # Number of steps
    option_type = 'call'  # 'call' or 'put'

    # Calculate Binomial Option Price
    binomial_price = binomial_option_pricing(S, K, T, r, sigma, N, option_type)

    # Calculate Black-Scholes Option Price
    if option_type == 'call':
        bs_price = black_scholes_call(S, K, T, r, sigma)
    elif option_type == 'put':
        bs_price = black_scholes_put(S, K, T, r, sigma)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Display Results
    print(f"=== {option_type.capitalize()} Option Pricing Comparison ===")
    print(f"Parameters:")
    print(f"  Underlying Price (S): {S}")
    print(f"  Strike Price (K): {K}")
    print(f"  Time to Maturity (T): {T} year(s)")
    print(f"  Risk-Free Rate (r): {r*100}%")
    print(f"  Volatility (σ): {sigma*100}%")
    print(f"  Number of Steps (N): {N}\n")

    print(f"Binomial Model {option_type.capitalize()} Option Price: {binomial_price:.4f}")
    print(f"Black-Scholes {option_type.capitalize()} Option Price: {bs_price:.4f}")

if __name__ == "__main__":
    main()
