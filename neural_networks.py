# OptionTrading/neural_networks.py

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
    Simplified Neural Network for fast option pricing.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        """
        Initialize a simple, fast network architecture.
        
        Parameters:
            input_dim: Number of input features (default 5 for [S, K, T, r, sigma])
            hidden_dim: Size of hidden layers (reduced for speed)
        """
        super(FOBSMNeuralNetwork, self).__init__()
        
        # Simple 3-layer network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple forward pass."""
        return self.network(x)

class OptionPricingTrainer:
    """
    Simple, fast trainer for option pricing neural network.
    """
    def __init__(self, model: FOBSMNeuralNetwork, learning_rate: float = 0.01):
        """
        Initialize trainer with model and optimization parameters.
        
        Parameters:
            model: Instance of FOBSMNeuralNetwork
            learning_rate: Learning rate for Adam optimizer (higher for faster training)
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        self.model.train()
        total_loss = 0.0
        
        for batch_features, batch_prices in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_features)
            loss = self.criterion(predictions, batch_prices)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        return {'loss': avg_loss}
    
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
