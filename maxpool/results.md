

## MaxPool Kernel Verification Results

MaxPool Verification: $X(1 \times 3 \times 32 \times 32) \rightarrow Y(1 \times 3 \times 15 \times 15)$

| Implementation | Max Abs Error | SNR (dB) |
| :--- | :--- | :--- |
| C++ Scalar | **0** | **inf** |
| C++ Vectorized (e32m1) | **0** | **inf** |
| C++ Vectorized (e32m2) | **0** | **inf** |
| C++ Vectorized (e32m4) | **0** | **inf** |
| C++ Vectorized (e32m8) | **0** | **inf** |
