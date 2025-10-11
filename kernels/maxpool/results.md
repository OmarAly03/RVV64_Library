```bash
omar@omar-Legion-5:~/Repos/RVV64_Library/kernels/maxpool$ make run
make -s -f ../../core/vicuna/sim/Makefile PROG_PATHS=progs.txt CORE=cv32e40x
- V e r i l a t i o n   R e p o r t: Verilator 5.026 2024-06-15 rev v5.026
- Verilator: Built from 1.901 MB sources in 71 modules, into 7.985 MB in 30 C++ files needing 0.006 MB
- Verilator: Walltime 1.267 s (elab=0.138, cvt=0.863, bld=0.000); cpu 0.000 s on 1 threads; alloced 95.195 MB
==== Beginning MaxPool Benchmarking ====
Input: [1,4,16,16], Kernel: 3x3, Stride: 2
Output: [1,4,7,7] (size: 196)
Total memory: 11936 bytes
=========================================

Initializing input data...
Input initialization time: 15367 cycles

Testing scalar implementation...
MaxPool scalar time: 37740 cycles
Testing vector e32m1 implementation...
MaxPool e32m1 time: 37280 cycles
M1 results: CORRECT (196/196 correct)
Testing vector e32m2 implementation...
MaxPool e32m2 time: 61996 cycles
M2 results: CORRECT (196/196 correct)
Testing vector e32m4 implementation...
MaxPool e32m4 time: 70564 cycles
M4 results: CORRECT (196/196 correct)
Testing vector e32m8 implementation...
MaxPool e32m8 time: 88484 cycles
M8 results: CORRECT (196/196 correct)

Sample outputs (first 8):
Index | Scalar Val/Idx | M1 Val/Idx | M2 Val/Idx | M4 Val/Idx | M8 Val/Idx
------|----------------|------------|------------|------------|------------
   0  |   -66 /  34   |  -66 /  34 |  -66 /  34 |  -66 /  34 |  -66 /  34
   1  |   -64 /  36   |  -64 /  36 |  -64 /  36 |  -64 /  36 |  -64 /  36
   2  |   -62 /  38   |  -62 /  38 |  -62 /  38 |  -62 /  38 |  -62 /  38
   3  |   -60 /  40   |  -60 /  40 |  -60 /  40 |  -60 /  40 |  -60 /  40
   4  |   -58 /  42   |  -58 /  42 |  -58 /  42 |  -58 /  42 |  -58 /  42
   5  |   -56 /  44   |  -56 /  44 |  -56 /  44 |  -56 /  44 |  -56 /  44
   6  |   -54 /  46   |  -54 /  46 |  -54 /  46 |  -54 /  46 |  -54 /  46
   7  |   -34 /  66   |  -34 /  66 |  -34 /  66 |  -34 /  66 |  -34 /  66

==== Performance Summary ====
Input size: 1024 elements (4096 bytes)
Output size: 196 elements (784 bytes)
Kernel: 3x3, Stride: 2
Scalar cycles: 37740
Vector M1 cycles: 37280 
```