CXX = /opt/riscv/bin/riscv64-unknown-linux-gnu-g++
CXXFLAGS = -march=rv64gcv -I./include
TARGET = conv2d_test
SOURCE = src/conv2d_rvv.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean