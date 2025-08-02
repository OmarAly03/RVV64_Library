GCC = riscv64-unknown-linux-gnu-gcc
FLAGS = -march=rv64gcv -mabi=lp64d -static

lr3: ./src/linear_reg.c main.c
	@$(GCC) $(FLAGS) -o main ./src/linear_reg.c main.c
	@qemu-riscv64 -cpu rv64,v=true ./main

mat_vec: ./examples/mat_vec_mult.c ./src/mat_mul.c
	@$(GCC) $(FLAGS) -o mat_vec ./examples/mat_vec_mult.c ./src/mat_mul.c
	@qemu-riscv64 -cpu rv64,v=true ./mat_vec

mat_transpose: ./examples/mat_transpose.c ./src/mat_mul.c
	@$(GCC) $(FLAGS) -o mat_transpose ./examples/mat_transpose.c ./src/mat_mul.c
	@qemu-riscv64 -cpu rv64,v=true ./mat_transpose

mat_mat: ./examples/mat_mat_mult.c ./src/mat_mul.c
	@$(GCC) $(FLAGS) -o mat_mat ./examples/mat_mat_mult.c ./src/mat_mul.c
	@qemu-riscv64 -cpu rv64,v=true ./mat_mat

conv_relu: ./examples/conv_relu_test.c ./src/conv_relu.c
	@$(GCC) $(FLAGS) -o conv_relu ./examples/conv_relu_test.c ./src/conv_relu.c
	@qemu-riscv64 -cpu rv64,v=true ./conv_relu

conv: ./perf_examples/conv_perftest.c 
	@$(GCC) $(FLAGS) -O2 -o conv ./perf_examples/conv_perftest.c 

conv_run: ./conv
	@qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./conv 

mattran: ./perf_examples/mattran_perftest.c
	@$(GCC) $(FLAGS) -O2 -o mattran ./perf_examples/mattran_perftest.c

mattran_run: ./mattran
	@taskset -c 0-3 qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./mattran 

matmul: ./perf_examples/matmul_perftest.c
	@$(GCC) $(FLAGS) -O2 -o matmul ./perf_examples/matmul_perftest.c

matmul_run: ./mattran
	@taskset -c 0-3 qemu-riscv64 -cpu rv64,v=true,vlen=1024 -plugin file=./libinsn.so -d plugin ./matmul

relu: ./perf_examples/relu_perftest.c 
	@$(GCC) $(FLAGS) -O2 -o relu ./perf_examples/relu_perftest.c
	
relu_run:
	@qemu-riscv64 -cpu rv64,v=true -plugin file=./libinsn.so -d plugin ./relu

softmax: ./perf_examples/softmax_perftest.c 
	@$(GCC) $(FLAGS) -O2 -o softmax ./perf_examples/softmax_perftest.c -lm
	
softmax_run: ./softmax
	@qemu-riscv64 -cpu rv64,v=true,vlen=1024 -plugin file=./libinsn.so -d plugin ./softmax

clean:
	@rm -f lr main mat_transpose mat_mat conv_relu matmul mattran relu conv softmax