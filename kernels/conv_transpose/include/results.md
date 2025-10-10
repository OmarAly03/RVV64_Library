```bash
==== Beginning Conv Transpose Benchmarking ====
Input: 1x2x8x8, Kernel: 2x2x3x3, Output: 1x2x15x15
Total elements - Input: 128, Kernel: 36, Output: 450
 
input/kernel initialization time: 2760 
conv_transpose time scalar: 62276 
conv_transpose time m1: 77452 
conv_transpose time m2: 85900 
conv_transpose time m4: 104204 
conv_transpose time m8: 149260 
================================================= 
```