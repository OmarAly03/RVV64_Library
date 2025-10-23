```bash
/content/RVV64_Library/scatter_elements# make run SIZE=16 AXIS=0
Running ScatterElements on 16x16 data, axis=0

ScatterElements on 16x16 data, axis=0

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      0                   inf                 
C Tiled Scalar                0                   inf                 
C Vectorized (e32m1)          0                   inf                 
C Vectorized (e32m2)          0                   inf                 
C Vectorized (e32m4)          0                   inf                 
C Vectorized (e32m8)          0                   inf                 
C Tiled Vectorized (e32m1)    0                   inf                 
C Tiled Vectorized (e32m2)    0                   inf                 
C Tiled Vectorized (e32m4)    0                   inf                 
C Tiled Vectorized (e32m8)    0                   inf                 
/content/RVV64_Library/scatter_elements# make run SIZE=16 AXIS=1
Running ScatterElements on 16x16 data, axis=1

ScatterElements on 16x16 data, axis=1

Implementation                Max Abs Error       SNR (dB)            
----------------------------------------------------------------------
ONNX Golden Ref               0                   inf                 
C Scalar                      0                   inf                 
C Tiled Scalar                0                   inf                 
C Vectorized (e32m1)          0                   inf                 
C Vectorized (e32m2)          0                   inf                 
C Vectorized (e32m4)          0                   inf                 
C Vectorized (e32m8)          0                   inf                 
C Tiled Vectorized (e32m1)    0                   inf                 
C Tiled Vectorized (e32m2)    0                   inf                 
C Tiled Vectorized (e32m4)    0                   inf                 
C Tiled Vectorized (e32m8)    0                   inf                    
```