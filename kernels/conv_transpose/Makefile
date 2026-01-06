CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static
HOST_CC = g++
HOST_FLAGS = -O2

# Include directories
INCLUDES = -Iinclude -I../../lib

# Source files
SRCS = run_conv_transpose.cpp src/rvv_conv2d_transpose.cpp src/utils.cpp
HOST_SRCS = run_conv_transpose.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_conv_transpose

# Default parameters: input_size in_channels out_channels [stride]
# Note: kernel_size=3 is fixed, stride can be 1 or 2, no padding support
# Usage: make run 8 3 64     (for input_size=8, in_channels=3, out_channels=64, stride=1)
# Usage: make run 8 3 64 2   (for input_size=8, in_channels=3, out_channels=64, stride=2)
INPUT_SIZE := $(or $(word 2,$(MAKECMDGOALS)),4)
IN_CHANNELS := $(or $(word 3,$(MAKECMDGOALS)),1)
OUT_CHANNELS := $(or $(word 4,$(MAKECMDGOALS)),1)
STRIDE := $(or $(word 5,$(MAKECMDGOALS)),1)

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
	@python3 src/onnx_conv2d_transpose.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE)

frun: $(TARGET) onnx
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE)
	@python3 main.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE)

host-run: host onnx
	@./output_files/run_conv_transpose $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE)
	@python3 main.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE) > results.md

# Build with RVV support for vectorized version
rvv: onnx
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -DRVV_AVAILABLE -o $(TARGET) $(SRCS)

# Run both scalar and vectorized versions
run: rvv
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE)
	@python3 main.py $(INPUT_SIZE) $(IN_CHANNELS) $(OUT_CHANNELS) $(STRIDE) > results.md

clean:
	rm -f $(TARGET)
	rm -rf output_files

.PHONY: run clean host rvv full-run
