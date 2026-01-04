CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# Source files
SRCS = run_batch_norm.cpp src/rvv_batch_norm.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_batch_norm

# Default tensor size (batch=2, channels=3, height=4, width=4)
N ?= 2
C ?= 3
H ?= 4
W ?= 4

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run: $(TARGET)
	@python3 src/onnx_batch_norm.py $(N) $(C) $(H) $(W)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(N) $(C) $(H) $(W)
	@python3 main.py $(N) $(C) $(H) $(W)

clean:
	rm -f output_files/*

.PHONY: run clean