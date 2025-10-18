CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -O0 -g -static

# Include directories
INCLUDES = -Iinclude -I../lib

# Source files
SRCS = run_matmul.cpp src/rvv_matmul.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_matmul

OUTPUT_FILES = ./output_files/*.bin

SIZE ?= 4 4 4
TILE ?= 8

$(TARGET): $(SRCS)
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

run:
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(SIZE) $(TILE)
	@python3 main.py $(SIZE) $(TILE)

clean:
	rm -f $(TARGET) $(OUTPUT_FILES)

.PHONY: run clean