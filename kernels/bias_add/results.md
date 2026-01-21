```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/bias_add$ make run_bias_add 
Running BiasAdd with B=1, C=16, H=14, W=14...

BiasAdd validation with B=1, C=16, H=14, W=14

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      0                   inf                 
C Vectorized (e32m1)          0                   inf                 
C Vectorized (e32m2)          0                   inf                 
C Vectorized (e32m4)          0                   inf                 
C Vectorized (e32m8)          0                   inf                 
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/bias_add$ 
```