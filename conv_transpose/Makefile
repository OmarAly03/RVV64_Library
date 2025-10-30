CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static
HOST_CC = g++
HOST_FLAGS = -O2

# Include directories
INCLUDES = -Iinclude -I../lib

# Source files
SRCS = run_conv_transpose.cpp src/rvv_conv_transpose.cpp src/conv_transpose_scalar.cpp src/utils.cpp
HOST_SRCS = run_conv_transpose.cpp src/conv_transpose_scalar.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_conv_transpose

# Default parameters: input_size in_channels out_channels [padding]
# Note: kernel_size=3, stride=2 are fixed, padding is configurable
# Usage: make run 8 3 64     (for input_size=8, in_channels=3, out_channels=64, padding=0)
# Usage: make run 8 3 64 1   (for input_size=8, in_channels=3, out_channels=64, padding=1)
INPUT_SIZE := $(or $(word 2,$(MAKECMDGOALS)),4)
IN_CHANNELS := $(or $(word 3,$(MAKECMDGOALS)),1)
OUT_CHANNELS := $(or $(word 4,$(MAKECMDGOALS)),1)
PADDING := $(or $(word 5,$(MAKECMDGOALS)),0)

# Dummy targets to allow numeric arguments
ifneq ($(word 2,$(MAKECMDGOALS)),)
$(word 2,$(MAKECMDGOALS)):
	@:
endif
ifneq ($(word 3,$(MAKECMDGOALS)),)
$(word 3,$(MAKECMDGOALS)):
	@:
endif
ifneq ($(word 4,$(MAKECMDGOALS)),)
$(word 4,$(MAKECMDGOALS)):
	@:
endif
ifneq ($(word 5,$(MAKECMDGOALS)),)
$(word 5,$(MAKECMDGOALS)):
	@:
endif

$(TARGET): $(SRCS)
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

host:
	@mkdir -p output_files
	@$(HOST_CC) $(HOST_FLAGS) $(INCLUDES) -o $(TARGET) $(HOST_SRCS)

onnx:
	@python3 create_onnx_model.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING)

frun: $(TARGET) onnx
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING)
	@python3 main.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING)

host-run: host onnx
	@./output_files/run_conv_transpose $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING)
	@python3 main.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING) > results.md

# Build with RVV support for vectorized version
rvv: onnx
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -DRVV_AVAILABLE -o $(TARGET) $(SRCS)

# Run both scalar and vectorized versions
run: rvv
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING)
	@python3 main.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(PADDING) > results.md

clean:
	rm -f $(TARGET)
	rm -rf output_files

.PHONY: run clean host rvv full-run
