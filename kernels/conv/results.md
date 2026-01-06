```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/conv$ make run
Conv2D: Input NCHW=1x3x8x8, Kernel OIHW=3x3x3x3, Stride=(1,1), Pad=(1,1)
Output NCHW=1x3x8x8

Conv2D: N=1 Cin=3 Cout=3 HxW=8x8 k=3x3 stride=(1,1) pad=(1,1)
Total operations: 10,368 FLOPs

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 9.53674e-07         138.992             
C Vectorized (e32m1)     7.15256e-07         139.9               
C Vectorized (e32m2)     7.15256e-07         139.9               
C Vectorized (e32m4)     7.15256e-07         139.9               
C Vectorized (e32m8)     7.15256e-07         139.9               
C IM2COL + GEMM (m8)     7.15256e-07         139.9   
```

```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/conv$ make run 
Conv2D: Input NCHW=1x128x26x26, Kernel OIHW=256x128x3x3, Stride=(1,1), Pad=(1,1)
Output NCHW=1x256x26x26

Conv2D: N=1 Cin=128 Cout=256 HxW=26x26 k=3x3 stride=(1,1) pad=(1,1)
Total operations: 398,721,024 FLOPs

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 6.48499e-05         123.937             
C Vectorized (e32m1)     3.8147e-05          127.627             
C Vectorized (e32m2)     3.8147e-05          127.627             
C Vectorized (e32m4)     3.8147e-05          127.627             
C Vectorized (e32m8)     3.8147e-05          127.627             
C IM2COL + GEMM (m8)     6.29425e-05         123.99  
```