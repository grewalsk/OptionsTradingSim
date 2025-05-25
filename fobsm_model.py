# fobsm_model.py

import numpy as np
from scipy.special import gamma
import torch
import torch.nn as nn

class FOBSM:
    def __init__(self, S, K, T, r, sigma, alpha=0.7, theta=0.5, transaction_cost=0.001):
        """
        Initialize Fractional Order Black-Scholes-Merton (FOBSM) model.
        
        Parameters:
            S (float): Current stock price
            K (float): Strike price
            T (float): Time to maturity
            r (float): Risk-free rate
            sigma (float): Volatility
            alpha (float): Fractional order for time derivative (0 < alpha < 1)
            theta (float): Fractional order for price derivative (0 < theta < 1)
            transaction_cost (float): Transaction cost factor
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.alpha = alpha
        self.theta = theta
        self.tc = transaction_cost

    def fractional_time_derivative(self, C, t, t_prev, dt):
        """
        Implements equation (10) from the paper:
        ∂ᵅC/∂tᵅ = 1/Γ(1-α) ∫[0 to t](t-τ)^(-α) ∂C/∂τ dτ
        """
        if t == 0:
            return 0
            
        # Discretize the integral
        tau = np.linspace(0, t, int(t/dt))
        weights = (t - tau + 1e-8) ** (-self.alpha)
        dC = np.diff(np.concatenate(([C[t_prev]], [C[t]]))) / dt
        
        # Compute Riemann-Liouville fractional derivative
        return np.sum(weights * dC * dt) / gamma(1 - self.alpha)

    def fractional_space_derivative(self, C, S, dS, order):
        n = len(S)
        derivative = np.zeros(n)
        m = int(np.ceil(order))  # Ensure m is an integer

        w = np.zeros(m+1)
        w[0] = 1
        for j in range(1, m+1):
            w[j] = w[j-1] * (order - j + 1) / j

        for i in range(m, n):
            i_m = int(i-m)  # Convert to integer
            derivative[i] = np.sum(w * C[i_m:i+1][::-1]) / (dS**order)

        return derivative


    def solve_fobsm_pde(self):
        N_t = 100  # Time steps
        N_s = 100  # Space steps
        dt = self.T/N_t
        S_max = 2 * self.S
        dS = S_max / N_s

        # Ensure S is a NumPy array
        S = np.linspace(0, S_max, N_s)
        S = np.array(S)  # Convert to NumPy array
    
        C = np.zeros((N_s, N_t))  # Option values
        C_hist = np.zeros((N_s, N_t))  # Store history

        C[:, -1] = np.maximum(self.K - S, 0)
        C_hist[:, -1] = C[:, -1]

        for j in range(N_t - 2, -1, -1):
            t_now = int(j)  # Convert j to integer

            for i in range(1, N_s - 1):
                S_now = S[i]

                C_s_frac = self.fractional_space_derivative(C[:, j+1], S, dS, self.theta)
                C_s_frac = np.array(C_s_frac)  # Ensure it's a NumPy array
                C_ss_frac = self.fractional_space_derivative(C[:, j+1], S, dS, 2*self.theta)
                C_ss_frac = np.array(C_ss_frac)  # Ensure it's a NumPy array

                transaction_cost = self.tc * C[i, j+1]

                drift = self.r * S_now * C_s_frac[i]
                diffusion = 0.5 * (self.sigma**2) * (S_now**2) * C_ss_frac[i]
                discount = self.r * C[i, j+1]

                C[i, j] = C[i, j+1] + dt * (
                    -self.fractional_time_derivative(C[i, :], t_now, j+1, dt)
                    + drift
                    + diffusion
                    - discount
                    - transaction_cost
                )

            # Apply early exercise condition for American put
                C[i, j] = max(C[i, j], self.K - S_now)

            C_hist[:, j] = C[:, j]

            C[0, j] = self.K
            C[-1, j] = 0

        # Ensure interpolation works correctly
        option_price = np.interp(self.S, S, C[:, 0])

        return option_price


    def calculate_greeks(self):
        """
        Calculate option Greeks using fractional derivatives
        """
        dS = 0.01 * self.S
        dt = 0.01 * self.T
        
        # Base price
        base_price = self.solve_fobsm_pde()
        
        # Delta
        S_up = FOBSM(self.S + dS, self.K, self.T, self.r, self.sigma, 
                     self.alpha, self.theta, self.tc)
        delta = (S_up.solve_fobsm_pde() - base_price) / dS
        
        # Theta
        T_down = FOBSM(self.S, self.K, self.T - dt, self.r, self.sigma,
                       self.alpha, self.theta, self.tc)
        theta = -(T_down.solve_fobsm_pde() - base_price) / dt
        
        # Gamma
        S_down = FOBSM(self.S - dS, self.K, self.T, self.r, self.sigma,
                       self.alpha, self.theta, self.tc)
        gamma = (S_up.solve_fobsm_pde() - 2*base_price + S_down.solve_fobsm_pde()) / (dS**2)
        
        return {
            'price': base_price,
            'delta': delta,
            'theta': theta,
            'gamma': gamma
        }

class FOBSMNeuralNetwork(nn.Module):
    """
    Neural Network architecture for FOBSM option pricing
    """
    def __init__(self, input_dim=7):  # [S, K, T, r, sigma, alpha, theta]
        super(FOBSMNeuralNetwork, self).__init__()
        
        # Deep network with residual connections
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(4)
        ])
        
        self.output_layer = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
            
        return self.output_layer(x)
class FOBSMDataset(torch.utils.data.Dataset):
    def __init__(self, features, prices):
        self.features = torch.FloatTensor(features)
        self.prices = torch.FloatTensor(prices).reshape(-1, 1)
    
    def __len__(self):
        return len(self.prices)
    
    def __getitem__(self, idx):
        return self.features[idx], self.prices[idx]

class FOBSMNeuralTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for features, prices in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, prices)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return {'loss': total_loss / len(dataloader)}
    
def fractional_bsm_price(S, K, T, r, sigma, alpha, theta=0.5, transaction_cost=0.001, option_type="call"):
    """
    Computes the option price using the Fractional Order Black-Scholes-Merton (FOBSM) model.

    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        alpha (float): Fractional order for time derivative (0 < alpha < 1).
        theta (float): Fractional order for space derivative (0 < theta < 1).
        transaction_cost (float): Transaction cost factor.
        option_type (str): "call" for a call option, "put" for a put option.

    Returns:
        float: The computed option price.
    """
    fobsm_model = FOBSM(S, K, T, r, sigma, alpha, theta, transaction_cost)
    option_price = fobsm_model.solve_fobsm_pde()

    if option_type.lower() == "call":
        return max(option_price, S - K)  # Early exercise condition for American call
    else:
        return max(option_price, K - S)  # Early exercise condition for American put