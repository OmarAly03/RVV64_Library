CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static -O3

# Include directories
INCLUDES = -Iinclude -I../../lib

# Source files (Update these to your MaxPool filenames)
# Assuming your kernels are in src/maxpool.cpp
SRCS = run_maxpool.cpp src/rvv_maxpool.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_maxpool

# Default Dimensions for 'make run'
N ?= 16
C ?= 1
H ?= 4
W ?= 4

$(TARGET): $(SRCS)
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run: $(TARGET)
	@qemu-riscv64 -cpu rv64,v=true,vlen=128 $(TARGET) $(N) $(C) $(H) $(W)
	@python3 main.py $(N) $(C) $(H) $(W)

clean:
	rm -f output_files/*.bin output_files/run_maxpool

.PHONY: run clean