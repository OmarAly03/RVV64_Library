# Compiler and Flags
CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -O0 -g -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# Source files for LeNet-5
SRCS = main.cpp src/lenet5.cpp src/kernels.cpp

# Output binary
TARGET = output_files/main

DIGIT ?= 0

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling LeNet-5 for RVV..."
	# Compile all sources and link with the C math library (-lm)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $(SRCS) -lm

run: $(TARGET)
	@echo "Running LeNet-5 on QEMU..."
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(DIGIT)

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)

.PHONY: all run clean
