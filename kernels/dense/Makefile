CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# --- Dense (GEMM) ---
SRCS_DENSE = run_dense.cpp src/rvv_dense.cpp src/utils.cpp
TARGET_DENSE = ./output_files/run_dense


# --- Kernel Sizes ---
SIZE ?= 16
B_SIZE ?= 1
IN_SIZE ?= 16
OUT_SIZE ?= 16


all: $(TARGET_RELU) $(TARGET_SOFTMAX) $(TARGET_DENSE)

$(TARGET_DENSE): $(SRCS_DENSE)
	@$(CC) $(FLAGS) $(INCLUDES) -lm -o $@ $^
	@python3 src/onnx_dense.py

run_dense: $(TARGET_DENSE)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET_DENSE) $(B_SIZE) $(IN_SIZE) $(OUT_SIZE)
	@python3 main.py $(B_SIZE) $(IN_SIZE) $(OUT_SIZE)

clean:
	rm -f output_files/*

.PHONY: all run_dense clean