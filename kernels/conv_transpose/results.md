
Transposed Convolution: 4x4 input, 3x3 kernel (fixed)
Stride: 1, No Padding, Channels: 1->1
Output: 6x6

Implementation           Max Abs Error       SNR (dB)            
-----------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
C Scalar                 1.19209e-07         143.238             
RVV 3x3 (m1)             6.70552e-08         146.66              
RVV 3x3 (m2)             6.70552e-08         146.66              
RVV 3x3 (m4)             6.70552e-08         146.66              
RVV 3x3 (m8)             6.70552e-08         146.66              
RVV General (m1)         1.19209e-07         143.85              
RVV General (m2)         1.19209e-07         143.85              
RVV General (m4)         1.19209e-07         143.85              
RVV General (m8)         1.19209e-07         143.85              
