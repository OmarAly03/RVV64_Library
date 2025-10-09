import numpy as np

def python_maxpool(X, K, S):
    N, C, H, W = X.shape
    OH = (H - K) // S + 1
    OW = (W - K) // S + 1
    Y = np.zeros((N, C, OH, OW), dtype=np.float32)

    for n in range(N):
        for c in range(C):
            for oh in range(OH):
                for ow in range(OW):
                    h_start, w_start = oh * S, ow * S
                    h_end, w_end = h_start + K, w_start + K
                    window = X[n, c, h_start:h_end, w_start:w_end]
                    Y[n, c, oh, ow] = np.max(window)
    return Y
