#import numpy as np
import cupy as cp
from numpy.typing import ArrayLike

eps = 1e-6

def cross_entropy_loss_regularized(y_pred: ArrayLike, y_true: ArrayLike, padding_token_idx = 0, alpha = 0.2) -> cp.ndarray:
    """
    Compute cross entropy loss between true 1-hot encoded vector and softmax output of a predictor.
    """
    
    # Clip predictions to prevent log(0)
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    
    # Compute cross-entropy loss
    loss = -cp.sum(y_true * cp.log(y_pred), axis=1)
    cross_entropy = cp.mean(loss)
    
    # Compute padding token penalty
    padding_probs = y_pred[:, padding_token_idx]  # Extract probabilities for the padding token
    padding_penalty = cp.mean(padding_probs)      # Average probability of padding token
    
    # Combine cross-entropy loss and regularization term
    total_loss = cross_entropy + alpha * padding_penalty
    
    return total_loss
