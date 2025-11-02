CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -static -O1 -g

# Include directories
INCLUDES = -Iinclude -I../lib

# Source files
SRCS = run_nms.cpp src/rvv_nms.cpp src/utils.cpp

# Output binary
TARGET = ./output_files/run_nms

# Default test parameters: batches classes spatial_dim max_boxes iou_threshold score_threshold
BATCHES ?= 1
CLASSES ?= 2
SPATIAL ?= 100
MAX_BOXES ?= 50
IOU_THRESH ?= 0.5
SCORE_THRESH ?= 0.1

.PHONY: run clean onnx

$(TARGET): $(SRCS)
	@mkdir -p output_files
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $^

onnx:
	@python3 create_onnx_model.py $(MAX_BOXES) $(IOU_THRESH) $(SCORE_THRESH)

run: $(TARGET) onnx onnx
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) $(BATCHES) $(CLASSES) $(SPATIAL) $(MAX_BOXES) $(IOU_THRESH) $(SCORE_THRESH)
	@python3 main.py $(BATCHES) $(CLASSES) $(SPATIAL) $(MAX_BOXES) $(IOU_THRESH) $(SCORE_THRESH)

clean:
	rm -f output_files/*

# Example usage:
# make BATCHES=2 CLASSES=3 SPATIAL=200 MAX_BOXES=100 IOU_THRESH=0.7 SCORE_THRESH=0.2 run
