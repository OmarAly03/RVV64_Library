```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/tensor_add$ make run_tensor_add 
Running TensorAdd on 16 elements...

TensorAdd validation on 16 elements

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      0                   inf                 
C Vectorized (e32m1)          0                   inf                 
C Vectorized (e32m2)          0                   inf                 
C Vectorized (e32m4)          0                   inf                 
C Vectorized (e32m8)          0                   inf                 
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/tensor_add$ 
```