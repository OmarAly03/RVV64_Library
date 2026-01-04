```bash
Running BatchNorm on 1x64x32x32 tensor...
C++ kernels completed.

BatchNormalization on 1x64x32x32 tensor

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