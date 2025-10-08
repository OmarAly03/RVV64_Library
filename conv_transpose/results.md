
Transposed Convolution: 16x16 input, 7x7 kernel
Stride: 1, Padding: 3, Channels: 1->1
Output: 16x16

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 1.43051e-06         136.177             
C Vectorized (e32m1)     1.43051e-06         135.746             
