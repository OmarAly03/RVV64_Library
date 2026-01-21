```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/maxpool$ make run 

MaxPool Verification
Input Shape:  (16, 1, 4, 4)
Kernel: 2x2, Stride: 1x1, Padding: 0x0
Output Shape: (16, 1, 3, 3)

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar Reference            0                   inf                 
C RVV e32m1                   0                   inf                 
C RVV e32m2                   0                   inf                 
C RVV e32m4                   0                   inf                 
C RVV e32m8                   0                   inf                 
C RVV tiled_m1                0                   inf                 
C RVV tiled_m2                0                   inf                 
C RVV tiled_m4                0                   inf                 
C RVV tiled_m8                0                   inf                 
----------------------------------------------------------------------
```