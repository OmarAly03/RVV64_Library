```bash
MaxPool Verification: X(1x3x512x512) -> Y(1x3x255x255)
Implementation              Max Abs Error       SNR (dB)            Indices Match?
----------------------------------------------------------------------------------------
C++ Scalar                  0                   inf                 CORRECT
C++ Vectorized (e32m1)      0                   inf                 CORRECT
C++ Vectorized (e32m2)      0                   inf                 CORRECT
C++ Vectorized (e32m4)      0                   inf                 CORRECT
C++ Vectorized (e32m8)      0                   inf                 CORRECT
```



