import numpy as np

# ==== Python Scalar (Triple nested loop) ====
def matmul_py_scalar(A, B):
    M, K = A.shape
    K, N = B.shape
    C = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C