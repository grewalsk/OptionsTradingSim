# model_comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
from typing import Dict, Tuple, List, Optional, Union
from pricing_models import black_scholes_call, black_scholes_put, monte_carlo_simulation, fobs_m_option_pricing
from fobsm_model import fractional_bsm_price
from neural_networks import predict_option_price
from sklearn.preprocessing import StandardScaler

def compare_option_pricing_models(S: float, K: float, T: float, r: float, sigma: float, 
                               alpha: Optional[float] = None, option_type: str = 'call', 
                               num_monte_carlo: int = 10000) -> Dict[str, float]:
    """
    Compare different option pricing models for the same parameters.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        alpha (float): Fractional parameter for FOBSM model.
        option_type (str): 'call' or 'put'.
        num_monte_carlo (int): Number of simulations for Monte Carlo.
        
    Returns:
        dict: Dictionary of model names and corresponding prices.
    """
    start_time = time.time()
    results = {}
    computation_times = {}
    
    # Black-Scholes model
    bs_start = time.time()
    if option_type.lower() == 'call':
        results['Black-Scholes'] = black_scholes_call(S, K, T, r, sigma)
    else:
        results['Black-Scholes'] = black_scholes_put(S, K, T, r, sigma)
    computation_times['Black-Scholes'] = time.time() - bs_start
    
    # FOBSM model
    if alpha is not None:
        fobsm_start = time.time()
        try:
            results['FOBSM'] = fractional_bsm_price(S, K, T, r, sigma, alpha, option_type=option_type.lower())
            computation_times['FOBSM'] = time.time() - fobsm_start
        except Exception as e:
            st.warning(f"Error calculating FOBSM price: {e}")
            results['FOBSM'] = None
            computation_times['FOBSM'] = 0
    
    # Monte Carlo simulation
    mc_start = time.time()
    try:
        mc_lower, mc_upper, _ = monte_carlo_simulation(
            S, K, T, r, sigma, alpha, option_type.lower(), num_monte_carlo, 0.95
        )
        results['Monte Carlo'] = (mc_lower + mc_upper) / 2
        results['MC Lower 95%'] = mc_lower
        results['MC Upper 95%'] = mc_upper
        computation_times['Monte Carlo'] = time.time() - mc_start
    except Exception as e:
        st.warning(f"Error during Monte Carlo simulation: {e}")
        results['Monte Carlo'] = None
        results['MC Lower 95%'] = None
        results['MC Upper 95%'] = None
        computation_times['Monte Carlo'] = 0
    
    # Neural Network (if available)
    if 'model_nn' in st.session_state and st.session_state['model_nn'] is not None:
        nn_start = time.time()
        model_nn = st.session_state['model_nn']
        
        # Prepare features based on whether alpha is included
        if alpha is not None:
            features = np.array([[S, K, T, r, sigma, alpha]])
        else:
            features = np.array([[S, K, T, r, sigma]])
        
        # Predict with NN
        try:
            scaler = StandardScaler()
            scaler.fit(features)  # This is simplified; in practice you'd use the scaler from training
            predicted_price = predict_option_price(model_nn, features, scaler)
            results['Neural Network'] = predicted_price
            computation_times['Neural Network'] = time.time() - nn_start
        except Exception as e:
            st.warning(f"Error predicting with Neural Network: {e}")
            results['Neural Network'] = None
            computation_times['Neural Network'] = 0
    
    # Add overall computation time
    computation_times['Total'] = time.time() - start_time
    
    # Add computation times to results
    for model, t in computation_times.items():
        results[f"{model} Time (s)"] = t
    
    return results

def plot_model_comparison(S: float, K: float, T: float, r: float, sigma: float, 
                        alpha: Optional[float] = None, option_type: str = 'call', 
                        param_to_vary: str = 'S', range_percent: float = 20, 
                        num_points: int = 50) -> plt.Figure:
    """
    Plot model comparison across a range of values for a given parameter.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        alpha (float): Fractional parameter for FOBSM model.
        option_type (str): 'call' or 'put'.
        param_to_vary (str): Parameter to vary ('S', 'K', 'T', 'r', 'sigma', 'alpha').
        range_percent (float): Percentage range to vary the parameter.
        num_points (int): Number of points to calculate.
        
    Returns:
        matplotlib.figure.Figure: The generated comparison plot.
    """
    start_time = time.time()
    
    params = {
        'S': S,
        'K': K,
        'T': T,
        'r': r,
        'sigma': sigma,
        'alpha': alpha if alpha is not None else 0.95  # Default value if alpha is None
    }
    
    # Create range of values for the parameter to vary
    min_val = max(0.01, params[param_to_vary] * (1 - range_percent / 100))
    max_val = params[param_to_vary] * (1 + range_percent / 100)
    
    # Special case for alpha which must be between 0 and 1
    if param_to_vary == 'alpha':
        min_val = max(0.01, min_val)
        max_val = min(1.0, max_val)
    
    param_values = np.linspace(min_val, max_val, num_points)
    
    # Determine which models to include
    use_fobsm = alpha is not None or param_to_vary == 'alpha'
    use_neural_network = 'model_nn' in st.session_state and st.session_state['model_nn'] is not None
    
    # Initialize arrays for model prices
    bs_prices = np.zeros(num_points)
    fobsm_prices = np.zeros(num_points) if use_fobsm else None
    nn_prices = np.zeros(num_points) if use_neural_network else None
    
    # Calculate prices for each parameter value
    for i, val in enumerate(param_values):
        temp_params = params.copy()
        temp_params[param_to_vary] = val
        
        # Black-Scholes prices
        if option_type.lower() == 'call':
            bs_prices[i] = black_scholes_call(temp_params['S'], temp_params['K'], 
                                            temp_params['T'], temp_params['r'], 
                                            temp_params['sigma'])
        else:
            bs_prices[i] = black_scholes_put(temp_params['S'], temp_params['K'], 
                                           temp_params['T'], temp_params['r'], 
                                           temp_params['sigma'])
        
        # FOBSM prices if applicable
        if use_fobsm:
            try:
                fobsm_prices[i] = fractional_bsm_price(
                    temp_params['S'], temp_params['K'], temp_params['T'],
                    temp_params['r'], temp_params['sigma'], temp_params['alpha'],
                    option_type.lower())
            except:
                fobsm_prices[i] = np.nan
        
        # Neural Network prices if available
        if use_neural_network:
            model_nn = st.session_state['model_nn']
            if alpha is not None:
                features = np.array([[temp_params['S'], temp_params['K'], 
                                    temp_params['T'], temp_params['r'], 
                                    temp_params['sigma'], temp_params['alpha']]])
            else:
                features = np.array([[temp_params['S'], temp_params['K'], 
                                    temp_params['T'], temp_params['r'], 
                                    temp_params['sigma']]])
            try:
                scaler = StandardScaler()
                scaler.fit(features)  # This is simplified; you'd use the scaler from training
                nn_prices[i] = predict_option_price(model_nn, features, scaler)
            except:
                nn_prices[i] = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot prices for each model
    ax.plot(param_values, bs_prices, 'b-', linewidth=2.5, label='Black-Scholes')
    
    if use_fobsm and not np.all(np.isnan(fobsm_prices)):
        ax.plot(param_values, fobsm_prices, 'r-', linewidth=2.5, label='FOBSM')
    
    if use_neural_network and not np.all(np.isnan(nn_prices)):
        ax.plot(param_values, nn_prices, 'g--', linewidth=2, label='Neural Network')
    
    # Format x-axis based on parameter type
    if param_to_vary in ['r', 'sigma']:
        # Display as percentage
        x_label = f"{param_to_vary} (%)"
        tick_labels = [f"{x*100:.1f}%" for x in param_values[::len(param_values)//5]]
        ax.set_xticks(param_values[::len(param_values)//5])
        ax.set_xticklabels(tick_labels)
    else:
        x_label = param_to_vary
    
    # Add grid, legend, labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Option Price ($)')
    ax.set_title(f'Model Comparison: Price vs {param_to_vary} for {option_type.capitalize()} Option')
    ax.legend(loc='best')
    
    # Add vertical line at current parameter value
    ax.axvline(x=params[param_to_vary], color='k', linestyle='--', alpha=0.5, label=f'Current {param_to_vary}')
    
    # Compute and display key information about current prices
    current_bs = bs_prices[len(bs_prices)//2]
    current_info = f"Current BS Price: ${current_bs:.2f}"
    
    if use_fobsm:
        current_fobsm = fobsm_prices[len(fobsm_prices)//2]
        if not np.isnan(current_fobsm):
            current_info += f", FOBSM: ${current_fobsm:.2f}"
    
    if use_neural_network:
        current_nn = nn_prices[len(nn_prices)//2]
        if not np.isnan(current_nn):
            current_info += f", NN: ${current_nn:.2f}"
    
    plt.figtext(0.02, 0.02, current_info, ha="left", fontsize=9)
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.98, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="right", fontsize=8, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout for text at bottom
    return fig

def calculate_model_error_metrics(true_prices: np.ndarray, model_predictions: List[np.ndarray], 
                                model_names: List[str]) -> pd.DataFrame:
    """
    Calculate error metrics for different models compared to true prices.
    
    Parameters:
        true_prices (np.array): True option prices.
        model_predictions (list of np.array): Predicted prices from different models.
        model_names (list): Names of models corresponding to predictions.
        
    Returns:
        pd.DataFrame: DataFrame of error metrics for each model.
    """
    metrics = {
        'Model': model_names,
        'MAE': [],
        'RMSE': [],
        'MAPE (%)': [],
        'Max Error': []
    }
    
    for predictions in model_predictions:
        # Filter out NaN values
        valid_indices = ~np.isnan(predictions)
        valid_true = true_prices[valid_indices]
        valid_pred = predictions[valid_indices]
        
        if len(valid_true) == 0:
            # All predictions are NaN
            metrics['MAE'].append(np.nan)
            metrics['RMSE'].append(np.nan)
            metrics['MAPE (%)'].append(np.nan)
            metrics['Max Error'].append(np.nan)
            continue
        
        # Mean Absolute Error
        mae = np.mean(np.abs(valid_pred - valid_true))
        metrics['MAE'].append(mae)
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((valid_pred - valid_true)**2))
        metrics['RMSE'].append(rmse)
        
        # Mean Absolute Percentage Error
        # Avoid division by zero
        nonzero_indices = valid_true != 0
        if np.any(nonzero_indices):
            mape = np.mean(np.abs((valid_true[nonzero_indices] - valid_pred[nonzero_indices]) / 
                               valid_true[nonzero_indices])) * 100
        else:
            mape = np.nan
        metrics['MAPE (%)'].append(mape)
        
        # Maximum Error
        max_error = np.max(np.abs(valid_pred - valid_true))
        metrics['Max Error'].append(max_error)
    
    df = pd.DataFrame(metrics)
    
    # Format the metrics for display
    for col in ['MAE', 'RMSE', 'Max Error']:
        df[col] = df[col].apply(lambda x: f"${x:.4f}" if not pd.isna(x) else "N/A")
    
    df['MAPE (%)'] = df['MAPE (%)'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
    
    return df

def plot_model_accuracy_comparison(S: float, K: float, T: float, r: float, 
                                 sigma: float, alpha: Optional[float] = None,
                                 option_type: str = 'call', 
                                 reference_model: str = 'Monte Carlo',
                                 num_simulations: int = 50000) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Compare model accuracy against a reference model, typically Monte Carlo simulation.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        alpha (float): Fractional parameter for FOBSM model.
        option_type (str): 'call' or 'put'.
        reference_model (str): Model to use as ground truth ('Monte Carlo' or 'FOBSM').
        num_simulations (int): Number of simulations for Monte Carlo reference.
        
    Returns:
        tuple: (matplotlib.figure.Figure, pd.DataFrame) with plot and error metrics.
    """
    start_time = time.time()
    
    # Generate a series of test cases by varying S
    test_S_values = np.linspace(0.7 * S, 1.3 * S, 20)
    
    # Arrays to store results
    bs_prices = np.zeros_like(test_S_values)
    fobsm_prices = np.zeros_like(test_S_values) if alpha is not None else None
    nn_prices = np.zeros_like(test_S_values) if 'model_nn' in st.session_state and st.session_state['model_nn'] is not None else None
    reference_prices = np.zeros_like(test_S_values)
    
    # Calculate prices for all models
    for i, test_S in enumerate(test_S_values):
        # Black-Scholes
        if option_type.lower() == 'call':
            bs_prices[i] = black_scholes_call(test_S, K, T, r, sigma)
        else:
            bs_prices[i] = black_scholes_put(test_S, K, T, r, sigma)
        
        # FOBSM if alpha is provided
        if alpha is not None:
            try:
                fobsm_prices[i] = fractional_bsm_price(test_S, K, T, r, sigma, alpha, option_type.lower())
            except:
                fobsm_prices[i] = np.nan
        
        # Neural Network if available
        if nn_prices is not None:
            model_nn = st.session_state['model_nn']
            if alpha is not None:
                features = np.array([[test_S, K, T, r, sigma, alpha]])
            else:
                features = np.array([[test_S, K, T, r, sigma]])
            
            try:
                scaler = StandardScaler()
                scaler.fit(features)
                nn_prices[i] = predict_option_price(model_nn, features, scaler)
            except:
                nn_prices[i] = np.nan
        
        # Reference model (typically Monte Carlo for high accuracy)
        if reference_model == 'Monte Carlo':
            try:
                mc_lower, mc_upper, _ = monte_carlo_simulation(
                    test_S, K, T, r, sigma, alpha, option_type.lower(), num_simulations, 0.95
                )
                reference_prices[i] = (mc_lower + mc_upper) / 2
            except:
                reference_prices[i] = np.nan
        elif reference_model == 'FOBSM' and alpha is not None:
            try:
                reference_prices[i] = fractional_bsm_price(test_S, K, T, r, sigma, alpha, option_type.lower())
            except:
                reference_prices[i] = np.nan
        else:
            # Default to Black-Scholes if no valid reference model
            if option_type.lower() == 'call':
                reference_prices[i] = black_scholes_call(test_S, K, T, r, sigma)
            else:
                reference_prices[i] = black_scholes_put(test_S, K, T, r, sigma)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot prices
    ax1.plot(test_S_values, reference_prices, 'k-', linewidth=3, label=f'{reference_model} (Reference)')
    ax1.plot(test_S_values, bs_prices, 'b-', linewidth=2, label='Black-Scholes')
    
    if fobsm_prices is not None and not np.all(np.isnan(fobsm_prices)):
        ax1.plot(test_S_values, fobsm_prices, 'r-', linewidth=2, label='FOBSM')
    
    if nn_prices is not None and not np.all(np.isnan(nn_prices)):
        ax1.plot(test_S_values, nn_prices, 'g--', linewidth=2, label='Neural Network')
    
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Option Price ($)')
    ax1.set_title(f'Model Price Comparison for {option_type.capitalize()} Option')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Plot errors (% difference from reference)
    bs_error = 100 * (bs_prices - reference_prices) / reference_prices
    ax2.plot(test_S_values, bs_error, 'b-', linewidth=2, label='Black-Scholes')
    
    if fobsm_prices is not None and not np.all(np.isnan(fobsm_prices)):
        fobsm_error = 100 * (fobsm_prices - reference_prices) / reference_prices
        ax2.plot(test_S_values, fobsm_error, 'r-', linewidth=2, label='FOBSM')
    
    if nn_prices is not None and not np.all(np.isnan(nn_prices)):
        nn_error = 100 * (nn_prices - reference_prices) / reference_prices
        ax2.plot(test_S_values, nn_error, 'g--', linewidth=2, label='Neural Network')
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Error (%)')
    ax2.set_title(f'Percentage Error Relative to {reference_model}')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # Calculate error metrics dataframe
    model_names = ['Black-Scholes']
    model_predictions = [bs_prices]
    
    if fobsm_prices is not None:
        model_names.append('FOBSM')
        model_predictions.append(fobsm_prices)
    
    if nn_prices is not None:
        model_names.append('Neural Network')
        model_predictions.append(nn_prices)
    
    error_df = calculate_model_error_metrics(reference_prices, model_predictions, model_names)
    
    # Add key parameter information
    plt.figtext(0.02, 0.02, 
               f"K=${K:.2f}, T={T:.2f}yrs, r={r*100:.2f}%, σ={sigma*100:.2f}%"
               + (f", α={alpha:.2f}" if alpha is not None else ""),
               ha="left", fontsize=9)
    
    # Calculate and add computation time
    computation_time = time.time() - start_time
    plt.figtext(0.98, 0.02, f"Computation time: {computation_time:.4f}s", 
               ha="right", fontsize=8, color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig, error_df

def compute_model_speed_benchmarks(S: float, K: float, T: float, r: float, sigma: float,
                                 alpha: Optional[float] = None, option_type: str = 'call',
                                 num_runs: int = 20) -> pd.DataFrame:
    """
    Benchmark the speed of different option pricing models.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        alpha (float): Fractional parameter for FOBSM model.
        option_type (str): 'call' or 'put'.
        num_runs (int): Number of runs for each model to average timing.
        
    Returns:
        pd.DataFrame: DataFrame with performance metrics for each model.
    """
    # Initialize result dictionary
    timings = {
        'Model': [],
        'Avg Time (ms)': [],
        'Min Time (ms)': [],
        'Max Time (ms)': [],
        'Std Dev (ms)': [],
        'Relative Speed': []
    }
    
    # Function to time a model
    def time_model(func, *args):
        times = []
        for _ in range(num_runs):
            start = time.time()
            func(*args)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        return {
            'avg': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times)
        }
    
    # Black-Scholes timing
    if option_type.lower() == 'call':
        bs_times = time_model(black_scholes_call, S, K, T, r, sigma)
    else:
        bs_times = time_model(black_scholes_put, S, K, T, r, sigma)
    
    timings['Model'].append('Black-Scholes')
    timings['Avg Time (ms)'].append(bs_times['avg'])
    timings['Min Time (ms)'].append(bs_times['min'])
    timings['Max Time (ms)'].append(bs_times['max'])
    timings['Std Dev (ms)'].append(bs_times['std'])
    timings['Relative Speed'].append(1.0)  # BS is the reference
    
    # FOBSM timing if alpha is provided
    if alpha is not None:
        try:
            fobsm_times = time_model(fractional_bsm_price, S, K, T, r, sigma, alpha, option_type.lower())
            
            timings['Model'].append('FOBSM')
            timings['Avg Time (ms)'].append(fobsm_times['avg'])
            timings['Min Time (ms)'].append(fobsm_times['min'])
            timings['Max Time (ms)'].append(fobsm_times['max'])
            timings['Std Dev (ms)'].append(fobsm_times['std'])
            timings['Relative Speed'].append(bs_times['avg'] / fobsm_times['avg'])
        except Exception as e:
            st.warning(f"Error benchmarking FOBSM: {e}")
    
    # Monte Carlo timing (reduced number of simulations for benchmarking)
    benchmark_sim_count = 1000
    try:
        def mc_benchmark():
            return monte_carlo_simulation(S, K, T, r, sigma, alpha, option_type.lower(), benchmark_sim_count, 0.95)
        
        mc_times = time_model(mc_benchmark)
        
        timings['Model'].append(f'Monte Carlo ({benchmark_sim_count} sims)')
        timings['Avg Time (ms)'].append(mc_times['avg'])
        timings['Min Time (ms)'].append(mc_times['min'])
        timings['Max Time (ms)'].append(mc_times['max'])
        timings['Std Dev (ms)'].append(mc_times['std'])
        timings['Relative Speed'].append(bs_times['avg'] / mc_times['avg'])
    except Exception as e:
        st.warning(f"Error benchmarking Monte Carlo: {e}")
    
    # Neural Network timing if available
    if 'model_nn' in st.session_state and st.session_state['model_nn'] is not None:
        try:
            model_nn = st.session_state['model_nn']
            
            # Prepare features
            if alpha is not None:
                features = np.array([[S, K, T, r, sigma, alpha]])
            else:
                features = np.array([[S, K, T, r, sigma]])
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            def nn_benchmark():
                return predict_option_price(model_nn, features, scaler)
            
            nn_times = time_model(nn_benchmark)
            
            timings['Model'].append('Neural Network')
            timings['Avg Time (ms)'].append(nn_times['avg'])
            timings['Min Time (ms)'].append(nn_times['min'])
            timings['Max Time (ms)'].append(nn_times['max'])
            timings['Std Dev (ms)'].append(nn_times['std'])
            timings['Relative Speed'].append(bs_times['avg'] / nn_times['avg'])
        except Exception as e:
            st.warning(f"Error benchmarking Neural Network: {e}")
    
    # Convert to DataFrame and format
    df = pd.DataFrame(timings)
    
    # Format numeric columns
    for col in ['Avg Time (ms)', 'Min Time (ms)', 'Max Time (ms)', 'Std Dev (ms)']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    df['Relative Speed'] = df['Relative Speed'].apply(lambda x: f"{x:.3f}x" if x < 1 else f"{x:.3f}x")
    
    return df