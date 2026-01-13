```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/dense$ make run_dense IN_SIZE=128 OUT_SIZE=128
Running Dense (GEMM) with B=1, IN=128, OUT=128...
Input range: [-1.0, +1.0]

Dense (GEMM) validation with B=1, IN=128, OUT=128

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      2.86102e-06         132.687             
C Vectorized (e32m1)          4.29153e-06         132.889             
C Vectorized (e32m2)          4.29153e-06         132.889             
C Vectorized (e32m4)          4.29153e-06         132.889             
C Vectorized (e32m8)          4.29153e-06         132.889             
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/dense$ 
```