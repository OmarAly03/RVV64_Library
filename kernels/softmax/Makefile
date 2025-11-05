CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../lib

# --- Softmax ---
# Note: Added -lm for the C math library (exp, fmax)
SRCS_SOFTMAX = run_softmax.cpp src/rvv_softmax.cpp src/utils.cpp
TARGET_SOFTMAX = ./output_files/run_softmax

# New default parameters
CHANNELS ?= 4
INNER_SIZE ?= 128

all: $(TARGET_SOFTMAX)


$(TARGET_SOFTMAX): $(SRCS_SOFTMAX)
	@$(CC) $(FLAGS) $(INCLUDES) -lm -o $@ $^
	@python3 src/onnx_softmax.py


run_softmax: $(TARGET_SOFTMAX)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET_SOFTMAX) $(CHANNELS) $(INNER_SIZE)
	@python3 main.py $(CHANNELS) $(INNER_SIZE)

clean:
	rm -f output_files/*

.PHONY: all run_softmax clean