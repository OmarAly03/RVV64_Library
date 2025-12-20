import numpy as np

def max_abs_error(f1_score):
    return 1.0 - f1_score

def snr_db(max_abs_error):
    return -20 * np.log10(max_abs_error) if max_abs_error > 0 else float('inf')
