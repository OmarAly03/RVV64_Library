CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static
HOST_CC = g++
HOST_FLAGS = -O2

# Include directories
INCLUDES = -Iinclude

# Source files
SRCS = run_conv_transpose.cpp src/rvv_conv_transpose.cpp src/conv_transpose_scalar.cpp src/utils.cpp
HOST_SRCS = run_conv_transpose.cpp src/conv_transpose_scalar.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_conv_transpose

# Default parameters: input_size kernel_size stride padding in_channels out_channels
INPUT_SIZE ?= 4
KERNEL_SIZE ?= 3
STRIDE ?= 2
PADDING ?= 1
IN_CHANNELS ?= 1
OUT_CHANNELS ?= 1

$(TARGET): $(SRCS)
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

host:
	@mkdir -p output_files
	@$(HOST_CC) $(HOST_FLAGS) $(INCLUDES) -o $(TARGET) $(HOST_SRCS)

onnx:
	@python3 create_onnx_model.py $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS)

frun: $(TARGET) onnx
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS)
	@python3 main.py $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS)

host-run: host onnx
	@./output_files/run_conv_transpose $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS)
	@python3 main.py $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS) > results.md

# Build with RVV support for vectorized version
rvv: onnx
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -DRVV_AVAILABLE -o $(TARGET) $(SRCS)

# Run both scalar and vectorized versions
run: rvv
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS)
	@python3 main.py $(INPUT_SIZE) $(KERNEL_SIZE) $(STRIDE) $(PADDING) $(IN_CHANNELS) $(OUT_CHANNELS) > results.md

clean:
	rm -f $(TARGET)
	rm -rf output_files

.PHONY: run clean host rvv full-run
