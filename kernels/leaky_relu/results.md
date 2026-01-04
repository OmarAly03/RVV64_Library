```bash
Running LeakyReLU on 1024 elements...
Input range: [-2.0, +2.0]
Alpha (negative slope): 0.01

LeakyReLU activation on 1024 elements

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      0                   inf                 
C Tiled Scalar                0                   inf                 
C Vectorized (e32m1)          0                   inf                 
C Vectorized (e32m2)          0                   inf                 
C Vectorized (e32m4)          0                   inf                 
C Vectorized (e32m8)          0                   inf                 
C Tiled Vectorized (e32m1)    0                   inf                 
C Tiled Vectorized (e32m2)    0                   inf                 
C Tiled Vectorized (e32m4)    0                   inf                 
C Tiled Vectorized (e32m8)    0                   inf                 
```