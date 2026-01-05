CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../../lib

# Source files
SRCS = run_conv2d.cpp src/rvv_conv2d.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_conv2d

# Optional runtime args (N Cin Cout H W kH kW sH sW pH pW)
SIZE ?=

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run:
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE)
	@python3 main.py $(SIZE)

clean:
	rm -f $(TARGET)

.PHONY: run clean