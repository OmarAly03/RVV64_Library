```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/matmul$ make run SIZE=128

Matrix multiplication: A(128x128) @ B(128x128) -> C(128x128)
Total operations: 4,194,304 FLOPs

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
Python Scalar            3.09944e-06         137.121             
NumPy dot                0                   inf                 
C Scalar                 3.09944e-06         137.121             
C Vectorized (e32m1)     0                   inf                 
C Vectorized (e32m2)     0                   inf                 
C Vectorized (e32m4)     0                   inf                 
C Vectorized (e32m8)     0                   inf 

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