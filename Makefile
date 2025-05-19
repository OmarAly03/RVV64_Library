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

clean:
	@rm -f lr main mat_transpose mat_mat conv_relu