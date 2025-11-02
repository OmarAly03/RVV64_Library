CXX = /opt/riscv/bin/riscv64-unknown-linux-gnu-g++
CXXFLAGS = -std=c++17 -march=rv64gcv -I./include -I../lib
TARGET = run_conv2d
SOURCES = src/conv2d_rvv.cpp src/conv2d_scalar.cpp
SYSROOT := $(shell $(CXX) -print-sysroot)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean

# Run under qemu-user with the cross sysroot so the dynamic loader is found
.PHONY: run
run: $(TARGET)
	@echo "Using SYSROOT=$(SYSROOT)"
	qemu-riscv64 -L $(SYSROOT) ./$(TARGET)

# Optional: build a static binary (requires static libs in your toolchain)
.PHONY: static
static:
	$(CXX) $(CXXFLAGS) -static $(SOURCES) -o $(TARGET)_static
	@echo "Built $(TARGET)_static (static). Run with: ./$(TARGET)_static under qemu-riscv64"