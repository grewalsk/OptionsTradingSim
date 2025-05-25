# Option Trading Application

A comprehensive option pricing application built with Streamlit that implements multiple pricing models including Black-Scholes, Crank-Nicolson, FOBSM, and Neural Networks.

## Features

- **European Options Pricing**: Black-Scholes model with Monte Carlo simulations
- **American Options Pricing**: Crank-Nicolson and FOBSM numerical methods
- **Neural Network Pricing**: Deep learning approach for option valuation
- **3D Visualizations**: Interactive surface plots and heatmaps
- **Real-time Data**: Integration with Alpha Vantage API for live stock data
- **Sensitivity Analysis**: Greeks calculation and parameter sensitivity

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Alpha Vantage API key in Streamlit secrets (optional for demo mode)

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                    # Main Streamlit application
├── pricing_models.py         # Option pricing implementations
├── neural_networks.py        # Neural network models and training
├── visualizations.py         # Plotting and heatmap functions
├── surface_plots.py          # 3D surface plotting utilities
├── DataFetcher.py           # API data fetching functions
├── utils.py                 # Utility functions and session management
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Mathematical Foundations

### 1. Black-Scholes Model

#### Mathematical Theory
The Black-Scholes model assumes that stock prices follow a geometric Brownian motion:

```
dS = μSdt + σSdW
```

Where:
- `S` = stock price
- `μ` = drift rate (expected return)
- `σ` = volatility
- `dW` = Wiener process (random walk)

#### Black-Scholes Partial Differential Equation
The option price V(S,t) satisfies:

```
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```

Where:
- `V(S,t)` = option value as function of stock price S and time t
- `r` = risk-free rate
- `σ` = volatility

#### Analytical Solutions

**European Call Option:**
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)
```

**European Put Option:**
```
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)
```

Where:
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

- `S₀` = current stock price
- `K` = strike price
- `T` = time to expiration
- `N(x)` = cumulative standard normal distribution
- `e` = mathematical constant (≈2.71828)
- `ln` = natural logarithm

#### The Greeks

**Delta (Δ)**: Price sensitivity to underlying asset price
```
Δ_call = N(d₁)
Δ_put = N(d₁) - 1
```

**Gamma (Γ)**: Rate of change of delta
```
Γ = φ(d₁) / (S₀σ√T)
```

**Theta (Θ)**: Time decay
```
Θ_call = -[S₀φ(d₁)σ/(2√T) + rKe^(-rT)N(d₂)]
```

**Vega (ν)**: Sensitivity to volatility
```
ν = S₀φ(d₁)√T
```

**Rho (ρ)**: Sensitivity to interest rate
```
ρ_call = KTe^(-rT)N(d₂)
ρ_put = -KTe^(-rT)N(-d₂)
```

Where `φ(x)` is the standard normal probability density function.

### 2. Crank-Nicolson Finite Difference Method

#### Mathematical Framework
For American options, we solve the Black-Scholes PDE with early exercise constraints using finite differences.

#### Discretization
Transform the PDE into a system of linear equations:
```
-aⱼVⱼ₋₁^(n+1) + (1+bⱼ)Vⱼ^(n+1) - cⱼVⱼ₊₁^(n+1) = aⱼVⱼ₋₁^n + (1-bⱼ)Vⱼ^n + cⱼVⱼ₊₁^n
```

Where:
```
aⱼ = (1/4)σ²j²Δt - (1/4)rjΔt
bⱼ = (1/2)σ²j²Δt + (1/2)rΔt
cⱼ = (1/4)σ²j²Δt + (1/4)rjΔt
```

- `j` = spatial grid index
- `n` = time step index
- `Δt` = time step size

#### Boundary Conditions

**Call Option:**
- `V(0,t) = 0` (worthless if S=0)
- `V(S_max,t) = S_max - Ke^(-r(T-t))` (deep in-the-money)
- `V(S,T) = max(S-K, 0)` (payoff at expiration)

**Put Option:**
- `V(0,t) = Ke^(-r(T-t))` (maximum value if S=0)
- `V(S_max,t) = 0` (worthless for large S)
- `V(S,T) = max(K-S, 0)` (payoff at expiration)

#### Early Exercise Constraint
For American options, at each time step:
```
V(S,t) ≥ max(S-K, 0)  [for calls]
V(S,t) ≥ max(K-S, 0)  [for puts]
```

### 3. FOBSM (Fractional Order Black-Scholes Model)

#### Mathematical Foundation
The Fractional Order Black-Scholes Model (FOBSM) extends the classical Black-Scholes framework by incorporating fractional calculus to model memory effects and long-range dependencies in financial markets. This implementation is inspired by the research presented in [this paper](https://arxiv.org/abs/2310.04464).

#### Fractional Black-Scholes Equation
The FOBSM replaces the standard second derivative with a fractional derivative of order α (0 < α ≤ 1):

```
∂V/∂t + (1/2)σ²S²(∂^α V/∂S^α) + rS(∂V/∂S) - rV = 0
```

Where:
- `V(S,t)` = option value as function of stock price S and time t
- `∂^α V/∂S^α` = Caputo fractional derivative of order α
- `α` = fractional order parameter (0 < α ≤ 1)
- All other parameters are the same as in classical Black-Scholes

#### Caputo Fractional Derivative
The Caputo fractional derivative of order α for a function f(x) is defined as:

```
D^α f(x) = (1/Γ(n-α)) ∫₀^x f^(n)(ξ)/(x-ξ)^(α-n+1) dξ
```

Where:
- `Γ(x)` = Gamma function: Γ(x) = ∫₀^∞ t^(x-1)e^(-t) dt
- `n = ⌈α⌉` = ceiling of α (smallest integer ≥ α)
- `f^(n)(ξ)` = nth derivative of f at point ξ

For the specific case in FOBSM (0 < α ≤ 1), this simplifies to:
```
∂^α V/∂S^α = (1/Γ(1-α)) ∫₀^S (∂V/∂ξ)/((S-ξ)^α) dξ
```

#### Numerical Implementation
The fractional derivative is approximated using the Grünwald-Letnikov formula:
```
∂^α V/∂S^α ≈ h^(-α) Σₖ₌₀^n (-1)^k (α choose k) V(S-kh)
```

Where:
- `h` = grid spacing
- `n` = number of grid points
- `(α choose k)` = generalized binomial coefficient:

```
(α choose k) = Γ(α+1)/(Γ(k+1)Γ(α-k+1))
```

#### FOBSM vs Classical Black-Scholes

| Parameter | Classical BS | FOBSM |
|-----------|-------------|--------|
| Derivative Order | α = 1 (integer) | 0 < α ≤ 1 (fractional) |
| Market Memory | No memory | Memory effects |
| Price Evolution | Markovian | Non-Markovian |
| Mathematical Tool | Standard calculus | Fractional calculus |

#### Physical and Financial Interpretation
- **α = 1**: Reduces to classical Black-Scholes (no memory)
- **α < 1**: Incorporates memory effects in market behavior
- **Lower α values**: More pronounced memory effects and long-range dependencies
- **Market implications**: Captures phenomena like volatility clustering, long-term correlations, and non-Gaussian price distributions

#### Advantages of FOBSM
1. **Memory Effects**: Models how past price movements influence current dynamics
2. **Market Anomalies**: Better captures real market behavior than classical models
3. **Flexibility**: Parameter α allows tuning for different market conditions
4. **Mathematical Rigor**: Based on well-established fractional calculus theory

*Note: This FOBSM implementation draws inspiration from the fractional calculus approach to option pricing as described in the referenced research paper.*

### 4. Neural Network Approach

#### Architecture
The neural network implements a feedforward architecture:

```
Input Layer (5 neurons): [S, K, T, r, σ]
Hidden Layer 1 (64 neurons): ReLU activation
Hidden Layer 2 (32 neurons): ReLU activation  
Output Layer (1 neuron): Option price
```

#### Mathematical Formulation
For a neural network with layers l = 1, 2, ..., L:

```
z^[l] = W^[l]a^[l-1] + b^[l]
a^[l] = g^[l](z^[l])
```

Where:
- `W^[l]` = weight matrix for layer l
- `b^[l]` = bias vector for layer l
- `g^[l]` = activation function for layer l
- `a^[l]` = activations for layer l

#### Loss Function
Mean Squared Error between predicted and theoretical prices:
```
L = (1/m) Σᵢ₌₁^m (ŷᵢ - yᵢ)²
```

Where:
- `m` = number of training samples
- `ŷᵢ` = predicted option price
- `yᵢ` = theoretical option price (from Black-Scholes)

#### Backpropagation
Gradients computed using chain rule:
```
∂L/∂W^[l] = (1/m) ∂L/∂a^[L] × ∂a^[L]/∂z^[L] × ... × ∂z^[l]/∂W^[l]
```

#### Optimization
Adam optimizer with learning rate scheduling:
```
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²
θₜ₊₁ = θₜ - α(m̂ₜ/(√v̂ₜ + ε))
```

Where:
- `gₜ` = gradient at time t
- `β₁, β₂` = exponential decay rates
- `α` = learning rate
- `ε` = small constant for numerical stability

### 5. Monte Carlo Simulation

#### Geometric Brownian Motion Simulation
Stock price evolution simulated as:
```
S(t+Δt) = S(t)exp((r - σ²/2)Δt + σ√Δt × Z)
```

Where `Z ~ N(0,1)` is a standard normal random variable.

#### Antithetic Variance Reduction
For each random draw Z, also simulate with -Z to reduce variance:
```
S₁ = S₀exp((r - σ²/2)T + σ√T × Z)
S₂ = S₀exp((r - σ²/2)T + σ√T × (-Z))
```

#### Confidence Intervals
Using Central Limit Theorem, 95% confidence interval:
```
CI = μ̂ ± 1.96 × (σ̂/√n)
```

Where:
- `μ̂` = sample mean of option payoffs
- `σ̂` = sample standard deviation
- `n` = number of simulations

## Models Implemented

### Summary Table

| Model | Type | Advantages | Use Case |
|-------|------|------------|----------|
| Black-Scholes | Analytical | Fast, exact solution | European options |
| Crank-Nicolson | Numerical | Handles early exercise | American options |
| FOBSM | Fractional | Memory effects | Enhanced modeling |
| Neural Network | ML | Flexible, adaptive | Pattern recognition |

## API Requirements

For real-time data fetching, you'll need an Alpha Vantage API key:
1. Get a free key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Add it to `.streamlit/secrets.toml`:
   ```toml
   [alphavantage]
   api_key = "your_api_key_here"
   ```

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Disclaimer

This application is for educational and research purposes only. It should not be used for actual trading decisions without proper validation and risk management.