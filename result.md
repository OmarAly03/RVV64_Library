conv: 224 x 224
input channels | output channels | speedup
1 | 1 | 1.31x






===================================
CONVOLUTION PERFORMANCE TEST
===================================

===== CONVOLUTION PERFORMANCE TEST =====
Input: 224x224x1, Filters: 1, Filter Size: 3x3, Stride: 1, Padding: 1
Output: 224x224x1
Sequential Execution time (3 iterations): 0.423 seconds
Sequential Instructions (3 iterations): 975052290
Vectorized Execution time (3 iterations): 0.323 seconds
Vectorized Instructions (3 iterations): 744522710
Speedup (Time): 1.31x
Speedup (Instructions): 1.31x
Correctness: PASSED

===================================
SUMMARY:
Convolution Test: PASSED
cpu 0 insns: 103193912
total insns: 103193912

===================================
CONVOLUTION PERFORMANCE TEST
===================================

===== CONVOLUTION PERFORMANCE TEST =====
Input: 224x224x16, Filters: 16, Filter Size: 3x3, Stride: 1, Padding: 1
Output: 224x224x16
Sequential Execution time (3 iterations): 17.822 seconds
Sequential Instructions (3 iterations): 41062847832
Vectorized Execution time (3 iterations): 6.934 seconds
Vectorized Instructions (3 iterations): 15976419630
Speedup (Time): 2.57x
Speedup (Instructions): 2.57x
Correctness: PASSED

===================================
SUMMARY:
Convolution Test: PASSED
cpu 0 insns: 2288777025
total insns: 2288777025

===================================
CONVOLUTION PERFORMANCE TEST
===================================

===== CONVOLUTION PERFORMANCE TEST =====
Input: 224x224x32, Filters: 32, Filter Size: 3x3, Stride: 1, Padding: 1
Output: 224x224x32
Sequential Execution time (3 iterations): 61.181 seconds
Sequential Instructions (3 iterations): 140962839664
Vectorized Execution time (3 iterations): 19.150 seconds
Vectorized Instructions (3 iterations): 44122342734
Speedup (Time): 3.19x
Speedup (Instructions): 3.19x
Correctness: PASSED

===================================
SUMMARY:
Convolution Test: PASSED
cpu 0 insns: 6641990763
total insns: 6641990763

===================================
CONVOLUTION PERFORMANCE TEST
===================================

===== CONVOLUTION PERFORMANCE TEST =====
Input: 224x224x64, Filters: 64, Filter Size: 3x3, Stride: 1, Padding: 1
Output: 224x224x64
Sequential Execution time (3 iterations): 224.852 seconds
Sequential Instructions (3 iterations): 518069907198
Vectorized Execution time (3 iterations): 63.705 seconds
Vectorized Instructions (3 iterations): 146780277592
Speedup (Time): 3.53x
Speedup (Instructions): 3.53x
Correctness: PASSED

===================================
SUMMARY:
Convolution Test: PASSED
cpu 0 insns: 22159597610
total insns: 22159597610

===================================
CONVOLUTION PERFORMANCE TEST
===================================

===== CONVOLUTION PERFORMANCE TEST =====
Input: 224x224x128, Filters: 128, Filter Size: 3x3, Stride: 1, Padding: 1
Output: 224x224x128
Sequential Execution time (3 iterations): 890.848 seconds
Sequential Instructions (3 iterations): 2052601046052
Vectorized Execution time (3 iterations): 256.234 seconds
Vectorized Instructions (3 iterations): 590397184162
Speedup (Time): 3.48x
Speedup (Instructions): 3.48x
Correctness: PASSED

===================================
SUMMARY:
Convolution Test: PASSED
cpu 0 insns: 79826132681
total insns: 79826132681

===================================
CONVOLUTION PERFORMANCE TEST
===================================

===== CONVOLUTION PERFORMANCE TEST =====
Input: 224x224x256, Filters: 256, Filter Size: 3x3, Stride: 1, Padding: 1
Output: 224x224x256
Sequential Execution time (3 iterations): 3485.439 seconds
Sequential Instructions (3 iterations): 8030687808962
Vectorized Execution time (3 iterations): 837.428 seconds
Vectorized Instructions (3 iterations): 1929480674906
Speedup (Time): 4.16x
Speedup (Instructions): 4.16x
Correctness: PASSED

===================================
SUMMARY:
Convolution Test: PASSED
cpu 0 insns: 301704207603
total insns: 301704207603

****************************************************************************************
===================================
MATRIX MULTIPLICATION PERFORMANCE TEST
===================================

===== MATRIX MULTIPLICATION PERFORMANCE TEST =====
Matrix A: 128x128, Matrix B: 128x128, Matrix C: 128x128
Sequential Execution time (multiply, 1 iterations): 0.027 seconds
Sequential Instructions (multiply, 1 iterations): 62426686
Vectorized Execution time (transpose+multiply, 1 iterations): 0.012 seconds
Vectorized Instructions (multiply only, 1 iterations): 26429862
Speedup (Time): 2.32x
Speedup (Instructions, multiply only): 2.36x
Correctness: PASSED

===================================
SUMMARY:
Matmul Test: PASSED
cpu 0 insns: 4071606
total insns: 4071606

===================================
MATRIX MULTIPLICATION PERFORMANCE TEST
===================================

===== MATRIX MULTIPLICATION PERFORMANCE TEST =====
Matrix A: 256x256, Matrix B: 256x256, Matrix C: 256x256
Sequential Execution time (multiply, 1 iterations): 0.220 seconds
Sequential Instructions (multiply, 1 iterations): 506982328
Vectorized Execution time (transpose+multiply, 1 iterations): 0.097 seconds
Vectorized Instructions (multiply only, 1 iterations): 222403424
Speedup (Time): 2.26x
Speedup (Instructions, multiply only): 2.28x
Correctness: PASSED

===================================
SUMMARY:
Matmul Test: PASSED
cpu 0 insns: 21178633
total insns: 21178633

===================================
MATRIX MULTIPLICATION PERFORMANCE TEST
===================================

===== MATRIX MULTIPLICATION PERFORMANCE TEST =====
Matrix A: 512x512, Matrix B: 512x512, Matrix C: 512x512
Sequential Execution time (multiply, 1 iterations): 1.754 seconds
Sequential Instructions (multiply, 1 iterations): 4041439460
Vectorized Execution time (transpose+multiply, 1 iterations): 0.710 seconds
Vectorized Instructions (multiply only, 1 iterations): 1630584770
Speedup (Time): 2.47x
Speedup (Instructions, multiply only): 2.48x
Correctness: PASSED

===================================
SUMMARY:
Matmul Test: PASSED
cpu 0 insns: 128222473
total insns: 128222473

===== MATRIX MULTIPLICATION PERFORMANCE TEST =====
Matrix A: 1024x1024, Matrix B: 1024x1024, Matrix C: 1024x1024
Sequential Execution time (multiply, 1 iterations): 13.595 seconds
Sequential Instructions (multiply, 1 iterations): 31324635054
Vectorized Execution time (transpose+multiply, 1 iterations): 5.410 seconds
Vectorized Instructions (multiply only, 1 iterations): 12439887340
Speedup (Time): 2.51x
Speedup (Instructions, multiply only): 2.52x
Correctness: PASSED

===================================
SUMMARY:
Matmul Test: PASSED
cpu 0 insns: 864671444
total insns: 864671444

===================================
MATRIX MULTIPLICATION PERFORMANCE TEST
===================================

===== MATRIX MULTIPLICATION PERFORMANCE TEST =====
Matrix A: 2048x2048, Matrix B: 2048x2048, Matrix C: 2048x2048
Sequential Execution time (multiply, 1 iterations): 121.007 seconds
Sequential Instructions (multiply, 1 iterations): 278814557066
Vectorized Execution time (transpose+multiply, 1 iterations): 46.182 seconds
Vectorized Instructions (multiply only, 1 iterations): 106300946638
Speedup (Time): 2.62x
Speedup (Instructions, multiply only): 2.62x
Correctness: PASSED

===================================
SUMMARY:
Matmul Test: PASSED
cpu 0 insns: 6276700224
total insns: 6276700224

===================================
MATRIX MULTIPLICATION PERFORMANCE TEST
===================================

===== MATRIX MULTIPLICATION PERFORMANCE TEST =====
Matrix A: 4096x4096, Matrix B: 4096x4096, Matrix C: 4096x4096
Sequential Execution time (multiply, 1 iterations): 1085.291 seconds
Sequential Instructions (multiply, 1 iterations): 2500677803534
Vectorized Execution time (transpose+multiply, 1 iterations): 371.018 seconds
Vectorized Instructions (multiply only, 1 iterations): 854405642604
Speedup (Time): 2.93x
Speedup (Instructions, multiply only): 2.93x
Correctness: PASSED

===================================
SUMMARY:
Matmul Test: PASSED
cpu 0 insns: 47654756993
total insns: 47654756993

******************************************************************************************

===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 256x256
Sequential Execution time (1 iterations): 0.002 seconds
Sequential Instructions (1 iterations): 4595642
Vectorized Execution time (1 iterations): 0.001 seconds
Vectorized Instructions (1 iterations): 1666860
Speedup (Time): 2.61x
Speedup (Instructions): 2.76x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 5249815
total insns: 5249815
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 256x256
Sequential Execution time (1 iterations): 0.002 seconds
Sequential Instructions (1 iterations): 4644430
Vectorized Execution time (1 iterations): 0.001 seconds
Vectorized Instructions (1 iterations): 1673138
Speedup (Time): 2.71x
Speedup (Instructions): 2.78x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 5249818
total insns: 5249818
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 1024x1024
Sequential Execution time (1 iterations): 0.035 seconds
Sequential Instructions (1 iterations): 79895258
Vectorized Execution time (1 iterations): 0.014 seconds
Vectorized Instructions (1 iterations): 32437016
Speedup (Time): 2.46x
Speedup (Instructions): 2.46x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 81384573
total insns: 81384573
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run

===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 2048x2048
Sequential Execution time (1 iterations): 0.143 seconds
Sequential Instructions (1 iterations): 328577282
Vectorized Execution time (1 iterations): 0.061 seconds
Vectorized Instructions (1 iterations): 139616232
Speedup (Time): 2.35x
Speedup (Instructions): 2.35x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 324999261
total insns: 324999261

omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
qemu: unknown option 'icount'
make: *** [Makefile:34: mattran_run] Error 1
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 4096x4096
Sequential Execution time (1 iterations): 0.569 seconds
Sequential Instructions (1 iterations): 1310068220
Vectorized Execution time (1 iterations): 0.232 seconds
Vectorized Instructions (1 iterations): 533483174
Speedup (Time): 2.46x
Speedup (Instructions): 2.46x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 1299410413
total insns: 1299410413
# @qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 8192x8192
Sequential Execution time (1 iterations): 2.206 seconds
Sequential Instructions (1 iterations): 5083804186
Vectorized Execution time (1 iterations): 0.910 seconds
Vectorized Instructions (1 iterations): 2096817034
Speedup (Time): 2.42x
Speedup (Instructions): 2.42x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 5197020354
total insns: 5197020354
# @qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 8192x8192
Sequential Execution time (1 iterations): 2.222 seconds
Sequential Instructions (1 iterations): 5120186924
Vectorized Execution time (1 iterations): 0.920 seconds
Vectorized Instructions (1 iterations): 2118652630
Speedup (Time): 2.42x
Speedup (Instructions): 2.42x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 5197020436
total insns: 5197020436
# @qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 4096x4096
Sequential Execution time (1 iterations): 0.558 seconds
Sequential Instructions (1 iterations): 1285742908
Vectorized Execution time (1 iterations): 0.254 seconds
Vectorized Instructions (1 iterations): 585722980
Speedup (Time): 2.19x
Speedup (Instructions): 2.20x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 1302556057
total insns: 1302556057
# @qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 4096x4096
Sequential Execution time (1 iterations): 0.544 seconds
Sequential Instructions (1 iterations): 1253853320
Vectorized Execution time (1 iterations): 0.254 seconds
Vectorized Instructions (1 iterations): 586188506
Speedup (Time): 2.14x
Speedup (Instructions): 2.14x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 1302556047
total insns: 1302556047
# @qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 4096x4096
Sequential Execution time (1 iterations): 0.542 seconds
Sequential Instructions (1 iterations): 1249362240
Vectorized Execution time (1 iterations): 0.230 seconds
Vectorized Instructions (1 iterations): 528987942
Speedup (Time): 2.36x
Speedup (Instructions): 2.36x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 1299410302
total insns: 1299410302
# @qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran
omar@omar-Legion-5:~/Repos/RVV64_Library$ make mattran_run
===================================
MATRIX TRANSPOSE PERFORMANCE TEST
===================================

===== MATRIX TRANSPOSE PERFORMANCE TEST =====
Input Matrix: 16384x16384
Sequential Execution time (1 iterations): 10.092 seconds
Sequential Instructions (1 iterations): 23256851434
Vectorized Execution time (1 iterations): 4.135 seconds
Vectorized Instructions (1 iterations): 9527675880
Speedup (Time): 2.44x
Speedup (Instructions): 2.44x
Correctness: PASSED

===================================
SUMMARY:
Transpose Test: PASSED
cpu 0 insns: 20787355757
total insns: 20787355757

===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x64
Sequential Execution time (10 iterations): 1.285 seconds
Sequential Instructions (10 iterations): 2960531710
Vectorized Execution time (10 iterations): 0.614 seconds
Vectorized Instructions (10 iterations): 1428664046
Speedup (Time): 2.09x
Speedup (Instructions): 2.07x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 345381639
total insns: 345381639
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x1
Sequential Execution time (10 iterations): 0.019 seconds
Sequential Instructions (10 iterations): 43636816
Vectorized Execution time (10 iterations): 0.009 seconds
Vectorized Instructions (10 iterations): 20754844
Speedup (Time): 2.09x
Speedup (Instructions): 2.10x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 5565130
total insns: 5565130
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x8
Sequential Execution time (10 iterations): 0.155 seconds
Sequential Instructions (10 iterations): 356911608
Vectorized Execution time (10 iterations): 0.080 seconds
Vectorized Instructions (10 iterations): 183261614
Speedup (Time): 1.95x
Speedup (Instructions): 1.95x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 43322443
total insns: 43322443
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x8
Sequential Execution time (10 iterations): 0.152 seconds
Sequential Instructions (10 iterations): 350127648
Vectorized Execution time (10 iterations): 0.079 seconds
Vectorized Instructions (10 iterations): 181223256
Speedup (Time): 1.93x
Speedup (Instructions): 1.93x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 43322371
total insns: 43322371
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x8
Sequential Execution time (1 iterations): 0.016 seconds
Sequential Instructions (1 iterations): 36444112
Vectorized Execution time (1 iterations): 0.008 seconds
Vectorized Instructions (1 iterations): 18231450
Speedup (Time): 1.99x
Speedup (Instructions): 2.00x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 33387559
total insns: 33387559
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x1
Sequential Execution time (1 iterations): 0.002 seconds
Sequential Instructions (1 iterations): 5022786
Vectorized Execution time (1 iterations): 0.001 seconds
Vectorized Instructions (1 iterations): 2504176
Speedup (Time): 1.92x
Speedup (Instructions): 2.01x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 4323340
total insns: 4323340
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x8
Sequential Execution time (1 iterations): 0.016 seconds
Sequential Instructions (1 iterations): 36288352
Vectorized Execution time (1 iterations): 0.008 seconds
Vectorized Instructions (1 iterations): 18463982
Speedup (Time): 1.97x
Speedup (Instructions): 1.97x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 33387549
total insns: 33387549
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
4===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x32
Sequential Execution time (1 iterations): 0.062 seconds
Sequential Instructions (1 iterations): 143692098
Vectorized Execution time (1 iterations): 0.033 seconds
Vectorized Instructions (1 iterations): 74816856
Speedup (Time): 1.92x
Speedup (Instructions): 1.92x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 133036995
total insns: 133036995
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x64
Sequential Execution time (1 iterations): 0.131 seconds
Sequential Instructions (1 iterations): 301311982
Vectorized Execution time (1 iterations): 0.067 seconds
Vectorized Instructions (1 iterations): 155147192
Speedup (Time): 1.94x
Speedup (Instructions): 1.94x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 265902912
total insns: 265902912
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 224x224x128
Sequential Execution time (1 iterations): 0.248 seconds
Sequential Instructions (1 iterations): 571078008
Vectorized Execution time (1 iterations): 0.127 seconds
Vectorized Instructions (1 iterations): 293242612
Speedup (Time): 1.95x
Speedup (Instructions): 1.95x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 531634961
total insns: 531634961
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 448x448x1
Sequential Execution time (1 iterations): 0.009 seconds
Sequential Instructions (1 iterations): 19918948
Vectorized Execution time (1 iterations): 0.004 seconds
Vectorized Instructions (1 iterations): 9935710
Speedup (Time): 1.96x
Speedup (Instructions): 2.00x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 16779316
total insns: 16779316
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu
omar@omar-Legion-5:~/Repos/RVV64_Library$ make relu_run
===================================
RELU PERFORMANCE TEST
===================================

===== RELU PERFORMANCE TEST =====
Input: 448x448x128
Sequential Execution time (1 iterations): 1.021 seconds
Sequential Instructions (1 iterations): 2351782216
Vectorized Execution time (1 iterations): 0.520 seconds
Vectorized Instructions (1 iterations): 1199177338
Speedup (Time): 1.96x
Speedup (Instructions): 1.96x
Correctness: PASSED

===================================
SUMMARY:
ReLU Test: PASSED
cpu 0 insns: 2126027315
total insns: 2126027315