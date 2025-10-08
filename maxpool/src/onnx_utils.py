import numpy as np

def snr_db(ref, test):
    noise = ref - test
    signal_power = np.sum(ref ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

def max_abs_error(ref, test):
    return np.max(np.abs(ref - test))
