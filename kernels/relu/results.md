```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/relu$ make run SIZE=65536
Running ReLU on 65536 elements...
Input range: [-2.0, +2.0]

ReLU activation on 65536 elements

Implementation                Max Abs Error       SNR (dB)            
------------------------------------------------------------
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
