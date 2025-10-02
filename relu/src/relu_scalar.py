import numpy as np

def relu_py_scalar(x):
    """Simple ReLU for 1D arrays using Python loops"""
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = max(0.0, x[i])
    return result