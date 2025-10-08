
Transposed Convolution: 4x4 input, 3x3 kernel
Stride: 2, Padding: 1, Channels: 1->1
Output: 7x7

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 5.96046e-08         152.463             
C Vectorized (e32m1)     5.96046e-08         149.386             
