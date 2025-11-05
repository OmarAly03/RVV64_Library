```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/relu$ make run SIZE=10000000
Running ReLU on 10000000 elements...
Input range: [-2.0, +2.0]

ReLU activation on 10000000 elements

Implementation           Max Abs Error       SNR (dB)            
------------------------------------------------------------
ONNX Golden Ref          0                   inf                 
Python Scalar            0                   inf                 
NumPy dot                0                   inf                 
C Scalar                 0                   inf                 
C Vectorized (e32m1)     0                   inf                 
C Vectorized (e32m2)     0                   inf                 
C Vectorized (e32m4)     0                   inf                 
C Vectorized (e32m8)     0                   inf
```
