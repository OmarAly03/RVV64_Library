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

```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/conv$ make run_3x3 
Conv2D 3x3 Test: Input HxW=128x128, Kernel=3x3, Stride=(1,1), Pad=(1,1), use_padding=1
Output: 128x128

Input range: [-0.0499998, 0.049997]
Kernel range: [-0.0252116, 0.0483024]

Running implementations...
==================================================

1. M1 (non-batched)... Done
2. M2 (non-batched)... Done
3. M4 (non-batched)... Done
4. M8 (non-batched)... Done
5. M2 (batched, batch_rows=4)... Done
6. M4 (batched, batch_rows=4)... Done
7. M8 (batched, batch_rows=4)... Done

All implementations completed.

3x3 Conv2D: Input HxW=128x128, Kernel=3x3, Pad=(1,1), use_padding=1
Output: 128x128
Total operations: 294,912 FLOPs

Implementation           Max Abs Error     SNR (dB)       Status    
----------------------------------------------------------------------
ONNX (Reference)         0                 inf            PASS      
M1 (non-batched)         0                 inf            PASS      
M2 (non-batched)         0                 inf            PASS      
M4 (non-batched)         0                 inf            PASS      
M8 (non-batched)         0                 inf            PASS      
M2 (batched)             0                 inf            PASS      
M4 (batched)             0                 inf            PASS      
M8 (batched)             0                 inf            PASS      

Value Ranges:
ONNX Ref: [-0.007526, 0.007060]
M1:       [-0.007526, 0.007060]
M8 batch: [-0.007526, 0.007060]
```