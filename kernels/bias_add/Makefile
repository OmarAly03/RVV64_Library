CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# --- BiasAdd ---
SRCS_BIAS_ADD = run_bias_add.cpp src/rvv_bias_add.cpp src/utils.cpp
TARGET_BIAS_ADD = ./output_files/run_bias_add

# --- Kernel Sizes ---
SIZE ?= 16
B_SIZE ?= 1
IN_SIZE ?= 16
OUT_SIZE ?= 16
C_SIZE ?= 16
H_SIZE ?= 14
W_SIZE ?= 14

all: $(TARGET_BIAS_ADD)

$(TARGET_BIAS_ADD): $(SRCS_BIAS_ADD)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^
	@python3 src/onnx_bias_add.py

run_bias_add: $(TARGET_BIAS_ADD)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET_BIAS_ADD) $(B_SIZE) $(C_SIZE) $(H_SIZE) $(W_SIZE)
	@python3 main.py $(B_SIZE) $(C_SIZE) $(H_SIZE) $(W_SIZE)

clean:
	rm -f output_files/*

.PHONY: all run_bias_add clean