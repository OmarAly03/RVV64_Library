CC = riscv64-linux-gnu-g++
FLAGS = -march=rv64gcv -static -O3

# Include directories
INCLUDES = -Iinclude

# Source files
SRCS = run_maxpool.cpp src/rvv_maxpool.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_maxpool

# Default size for H and W, can be overridden from the command line
SIZE ?= 32 32

all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling RISC-V MaxPool executable..."
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run: all
	@mkdir -p ./output_files
	@echo "--- 1. Generating ONNX model ---"
	@python3 src/onnx_maxpool.py
	@echo "\n--- 2. Generating input data (X.bin) ---"
	@python3 generate_data.py $(SIZE)
	@echo "\n--- 3. Running C++ kernels via QEMU ---"
	@qemu-riscv64-static -cpu rv64,v=true,vlen=128,vext_spec=v1.0 $(TARGET) $(SIZE)
	@echo "\n--- 4. Running final Python verification ---"
	@python3 main.py $(SIZE)

clean:
	@echo "Cleaning up..."
	@rm -rf output_files

.PHONY: all run clean
