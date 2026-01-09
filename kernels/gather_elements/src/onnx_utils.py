import numpy as np

def max_abs_error(ref, result):
    """Calculate maximum absolute error between reference and result."""
    return np.max(np.abs(ref - result))

def snr_db(ref, result):
    """Calculate Signal-to-Noise Ratio in dB."""
    noise = ref - result
    signal_power = np.sum(ref ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
