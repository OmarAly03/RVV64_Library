CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude

# Source files
SRCS = run_nms.cpp src/rvv_nms.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_nms

SIZE ?= 16

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run: $(TARGET)
	@python3 src/onnx_nms.py $(SIZE)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE)
	@python3 main.py $(SIZE)

clean:
	rm -f $(TARGET) output_files/nms.onnx output_files/input.bin output_files/nms_*.bin

.PHONY: run clean