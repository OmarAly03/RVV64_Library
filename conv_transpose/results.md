```bash
Transposed Convolution: 512x512 input, 5x5 kernel
Stride: 3, Padding: 2, Channels: 2->4
Output: 1534x1534

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 7.15256e-07         143.13              
C Vectorized (e32m1)     7.15256e-07         143.473             

