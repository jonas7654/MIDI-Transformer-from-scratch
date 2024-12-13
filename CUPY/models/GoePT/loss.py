#import numpy as np
import cupy as cp
from numpy.typing import ArrayLike

eps = 1e-6

def cross_entropy_loss(y_pred: ArrayLike, y_true: ArrayLike) -> cp.ndarray:
    """
    Compute cross entropy loss between true 1-hot encoded vector and softmax output of a predictor.
    """
    
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    # Compute cross entropy loss
    loss = -cp.sum(y_true * cp.log(y_pred), axis = 1)
    
    
    
    return cp.mean(loss)
