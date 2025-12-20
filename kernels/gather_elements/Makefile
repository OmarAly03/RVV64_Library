CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../lib

# Source files
SRCS = run_gather.cpp src/gather_elements_impl.cpp

# Output binary
TARGET = ./output_files/run_gather

SIZE ?= 8
AXIS ?= 0

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run: $(TARGET)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE) $(AXIS)
	@python3 main.py $(SIZE) $(AXIS)

test: $(TARGET)
	@echo "Testing axis=0..."
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE) 0
	@python3 main.py $(SIZE) 0
	@echo ""
	@echo "Testing axis=1..."
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE) 1
	@python3 main.py $(SIZE) 1

clean:
	rm -f output_files/*

.PHONY: run test clean
