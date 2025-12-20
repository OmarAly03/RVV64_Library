import numpy as np

def max_abs_error(ref, test):
    """Calculate maximum absolute error between reference and test arrays."""
    return np.max(np.abs(ref - test))

def snr_db(ref, test):
    """Calculate Signal-to-Noise Ratio in dB."""
    noise = ref - test
    signal_power = np.sum(ref ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = signal_power / noise_power
    return 10 * np.log10(snr) if snr > 0 else float('-inf')
