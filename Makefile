lr2: linear_reg.c main.c
	@riscv64-unknown-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -static -o main linear_reg.c main.c
	@qemu-riscv64 -cpu rv64,v=true ./main
clean:
	@rm -f lr main