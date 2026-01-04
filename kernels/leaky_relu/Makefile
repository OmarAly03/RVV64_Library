CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# Source files
SRCS = run_leaky_relu.cpp src/rvv_leaky_relu.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_leaky_relu

SIZE ?= 16

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^
	@python3 src/onnx_leaky_relu.py

run:
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE)
	@python3 main.py $(SIZE)

clean:
	rm -f output_files/*

.PHONY: run clean