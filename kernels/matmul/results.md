```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/matmul$ make run SIZE="64 64 64"
Matrix dimensions: 64x64 (K=64)
Tile size: 8
Total operations: 0.524288 million FLOPs

Matrix multiplication: A(64x64) @ B(64x64) -> C(64x64)
Total operations: 524,288 FLOPs

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 1.90735e-06         138.974             
C e32m1                  0                   inf                 
C e32m2                  0                   inf                 
C e32m4                  0                   inf                 
C e32m8                  0                   inf                 
C e32m1 unroll           0                   inf                 
C e32m2 unroll           0                   inf                 
C e32m4 unroll           0                   inf                 
C e32m8 unroll           0                   inf                 
C Tiled Scalar           2.86102e-06         135.8               
C Tiled e32m1            0                   inf                 
C Tiled e32m2            0                   inf                 
C Tiled e32m4            0                   inf                 
C Tiled e32m8            0                   inf                 
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/matmul$
```