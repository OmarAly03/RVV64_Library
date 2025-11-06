CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static

# Include directories
INCLUDES = -Iinclude -I../lib

# Source files
SRCS = run_gather.cpp src/gather_impl.cpp

# Output binary
TARGET = ./output_files/run_gather

SIZE ?= 8
AXIS ?= 0

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run: $(TARGET)
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE) $(AXIS)
	@python3 main.py $(SIZE) $(AXIS)

clean:
	rm -f output_files/*

.PHONY: run clean
