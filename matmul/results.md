```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/matmul$ make run SIZE=512

Matrix multiplication: A(512x512) @ B(512x512) -> C(512x512)
Total operations: 268,435,456 FLOPs

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
Python Scalar            3.24249e-05         127.278             
NumPy dot                1.71661e-05         131.108             
C Scalar                 3.24249e-05         127.278             
C Vectorized (e32m1)     3.24249e-05         127.465             
C Vectorized (e32m2)     3.24249e-05         127.465             
C Vectorized (e32m4)     3.24249e-05         127.465             
C Vectorized (e32m8)     3.24249e-05         127.465
```