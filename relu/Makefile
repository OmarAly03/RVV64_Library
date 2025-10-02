CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude

# Source files
SRCS = run_relu.cpp src/rvv_relu.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_relu

SIZE ?= 16

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run:
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE)
	@python3 main.py $(SIZE)

clean:
	rm -f $(TARGET)

.PHONY: run clean