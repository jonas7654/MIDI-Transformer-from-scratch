#import numpy as np
import cupy as cp
from numpy.typing import ArrayLike

eps = 1e-6

def cross_entropy_loss(y_pred: ArrayLike, y_true: ArrayLike) -> cp.ndarray:
    """
    Compute cross entropy loss between true 1-hot encoded vector and softmax output of a predictor.
    """
    
    # Clip predictions to prevent log(0)
    y_pred = cp.clip(y_pred, eps, 1 - eps)

    # Compute cross-entropy loss
    loss = -cp.sum(y_true * cp.log(y_pred), axis=1)

    # Create a mask for non-padding tokens 
    mask = (cp.argmax(y_true, axis=1) != 0).astype(cp.float32)

    # Apply mask to the loss
    loss = loss * mask

    # Normalize the loss by the number of valid tokens
    return cp.sum(loss) / cp.sum(mask)
