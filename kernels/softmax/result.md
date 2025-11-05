```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/softmax$ make run_softmax
Running Softmax on 4 channels, 128 inner size (Total 512 elements)...
Input range: [-2.0, +2.0]

Softmax validation on shape (4, 128) (Total 512 elements)

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      5.96046e-08         145.884             
C Vectorized                  5.96046e-08         146.306  
```