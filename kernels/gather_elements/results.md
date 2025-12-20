```bash
» make test SIZE=16

Testing axis=0...
Running GatherElements on 16x16 data, axis=0

GatherElements on 16x16 data, axis=0

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

Testing axis=1...
Running GatherElements on 16x16 data, axis=1

GatherElements on 16x16 data, axis=1

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
