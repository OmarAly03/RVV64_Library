CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# --- TensorAdd ---
SRCS_TENSOR_ADD = run_tensor_add.cpp src/rvv_tensor_add.cpp src/utils.cpp
TARGET_TENSOR_ADD = ./output_files/run_tensor_add


# --- Kernel Sizes ---
SIZE ?= 16
B_SIZE ?= 1
IN_SIZE ?= 16
OUT_SIZE ?= 16
C_SIZE ?= 16
H_SIZE ?= 14
W_SIZE ?= 14


all: $(TARGET_TENSOR_ADD)

$(TARGET_TENSOR_ADD): $(SRCS_TENSOR_ADD)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^
	@python3 src/onnx_tensor_add.py

run_tensor_add: $(TARGET_TENSOR_ADD)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET_TENSOR_ADD) $(SIZE)
	@python3 main.py $(SIZE)

clean:
	rm -f output_files/*

.PHONY: all run_tensor_add clean