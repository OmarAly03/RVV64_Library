import numpy as np

def snr_db(ref, test):
    """
    Calculate Signal-to-Noise Ratio in dB
    """
    noise_power = np.mean((ref - test) ** 2)
    signal_power = np.mean(ref ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def max_abs_error(ref, test):
    """
    Calculate maximum absolute error between reference and test arrays
    """
    return np.max(np.abs(ref - test))
