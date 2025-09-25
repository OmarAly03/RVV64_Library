import numpy as np

# ==== Python Scalar ====
def relu_py_scalar(x):
    """
    Python scalar implementation of ReLU activation.
    
    Args:
        x (np.ndarray): Input tensor
    Returns:
        np.ndarray: Output tensor with ReLU applied element-wise
    """
    x_flat = x.flatten()
    y_flat = np.zeros_like(x_flat)
    
    for i in range(len(x_flat)):
        y_flat[i] = max(0.0, x_flat[i])
    
    return y_flat.reshape(x.shape)