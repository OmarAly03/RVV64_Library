Transposed Convolution: 64x64 input, 3x3 kernel (fixed)
Stride: 1, No Padding, Channels: 3->64
Output: 66x66

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 1.90735e-06         138.499             
C Vectorized (e32m2)     2.38419e-06         138.79         
