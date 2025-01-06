# neural_networks.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler

class OptionPricingDataset(Dataset):
    """
    Custom PyTorch Dataset for option pricing data.
    Handles data preprocessing and scaling automatically.
    """
    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize dataset with features and target prices.
        
        Parameters:
            features: Array of shape (n_samples, n_features) containing option parameters
                     [S, K, T, r, sigma]
            prices: Array of shape (n_samples,) containing option prices
            scaler: Optional pre-fit StandardScaler for feature normalization
        """
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = torch.FloatTensor(self.scaler.fit_transform(features))
        else:
            self.scaler = scaler
            self.features = torch.FloatTensor(self.scaler.transform(features))
            
        self.prices = torch.FloatTensor(prices).reshape(-1, 1)
        
    def __len__(self) -> int:
        return len(self.prices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.prices[idx]

class FOBSMNeuralNetwork(nn.Module):
    """
    Neural Network architecture designed for option pricing.
    Uses residual connections and batch normalization for improved training stability.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 128):
        """
        Initialize the network architecture.
        
        Parameters:
            input_dim: Number of input features (default 5 for [S, K, T, r, sigma])
            hidden_dim: Size of hidden layers
        """
        super(FOBSMNeuralNetwork, self).__init__()
        
        # Input layer with batch normalization
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights for better convergence."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        x = self.input_layer(x)
        
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
            
        return self.output_layer(x)

class OptionPricingTrainer:
    """
    Handles the training process for the option pricing neural network.
    Includes price and Greeks calculation in the loss function.
    """
    def __init__(self, model: FOBSMNeuralNetwork, learning_rate: float = 0.001):
        """
        Initialize trainer with model and optimization parameters.
        
        Parameters:
            model: Instance of FOBSMNeuralNetwork
            learning_rate: Learning rate for Adam optimizer
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.price_criterion = nn.MSELoss()
        
    def calculate_greeks(self, S: torch.Tensor, prices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate first and second order derivatives for Greeks."""
        S.requires_grad_(True)
        pred_prices = self.model(S)
        
        # Calculate Delta (first derivative)
        delta = torch.autograd.grad(
            pred_prices, S, grad_outputs=torch.ones_like(pred_prices),
            create_graph=True, retain_graph=True
        )[0]
        
        # Calculate Gamma (second derivative)
        gamma = torch.autograd.grad(
            delta, S, grad_outputs=torch.ones_like(delta),
            retain_graph=True
        )[0]
        
        return delta, gamma
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        self.model.train()
        total_loss = 0.0
        
        for batch_features, batch_prices in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_features)
            price_loss = self.price_criterion(predictions, batch_prices)
            
            # Calculate Greeks loss
            delta, gamma = self.calculate_greeks(batch_features, batch_prices)
            greeks_loss = (delta.abs().mean() + gamma.abs().mean()) * 0.1
            
            # Combined loss
            loss = price_loss + greeks_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        self.scheduler.step(avg_loss)
        
        return {
            'loss': avg_loss,
            'price_loss': price_loss.item(),
            'greeks_loss': greeks_loss.item()
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Perform validation and return metrics."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_features, batch_prices in dataloader:
                predictions = self.model(batch_features)
                loss = self.price_criterion(predictions, batch_prices)
                total_loss += loss.item()
                
        return {'val_loss': total_loss / len(dataloader)}

def predict_option_price(model: FOBSMNeuralNetwork, features: np.ndarray,
                        scaler: StandardScaler) -> float:
    """
    Predict option price using trained model.
    
    Parameters:
        model: Trained FOBSMNeuralNetwork instance
        features: Array of shape (5,) containing option parameters [S, K, T, r, sigma]
        scaler: Scaler used during training
    
    Returns:
        float: Predicted option price
    """
    model.eval()
    with torch.no_grad():
        scaled_features = scaler.transform(features.reshape(1, -1))
        x = torch.FloatTensor(scaled_features)
        prediction = model(x)
        return prediction.item()

def calculate_prediction_intervals(model: FOBSMNeuralNetwork, features: np.ndarray,
                                scaler: StandardScaler, n_samples: int = 1000) -> Dict[str, float]:
    """
    Calculate prediction intervals using dropout-based uncertainty estimation.
    
    Parameters:
        model: Trained FOBSMNeuralNetwork instance
        features: Input features for prediction
        scaler: Scaler used during training
        n_samples: Number of Monte Carlo samples
    
    Returns:
        Dictionary containing mean prediction and confidence intervals
    """
    model.train()  # Enable dropout for uncertainty estimation
    predictions = []
    
    scaled_features = scaler.transform(features.reshape(1, -1))
    x = torch.FloatTensor(scaled_features)
    
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(x).item()
            predictions.append(pred)
            
    predictions = np.array(predictions)
    
    return {
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'lower_95': float(np.percentile(predictions, 2.5)),
        'upper_95': float(np.percentile(predictions, 97.5))
    }
