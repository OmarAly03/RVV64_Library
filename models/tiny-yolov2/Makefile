# TARGET = main
IMG ?= cat

IMG_DIR = ./images/
BIN_IMG_DIR = ./image_binaries/

# Compiler and Flags
CC = riscv64-unknown-linux-gnu-g++
FLAGS = -march=rv64gcv -O1 -g -static

# Include directories
INCLUDES = -Iinclude 

# Source files for YOLO (removed config.cpp)
SRCS = main.cpp src/model.cpp src/kernels.cpp

# Output binary
TARGET = output_files/main

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	@echo "Compiling YOLO for RVV..."
	@$(CC) $(FLAGS) $(INCLUDES) -o $@ $(SRCS) -lm 

run:
	@echo "Running YOLO inference on RISC-V..."
	@qemu-riscv64 -cpu rv64,v=true $(TARGET) image_binaries/$(IMG).bin model_parameters/
	@echo "-------------------------------------------------------------------"
	@echo "Visualizing results..."
	@python3 visualize_results.py images/$(IMG).jpg ./output_files/detection_results.txt -o ./output_files/output_detected.jpg
	@echo "Complete! Check output_detected.jpg for the visualization."

extract_parameters: src/extract_weights.py
	@echo "Extracting Parameters..."
	@python3 src/extract_weights.py
	
extract_images: src/extract_image.py
	python3 src/extract_image.py $(IMG_DIR) $(BIN_IMG_DIR)

clean:
	@echo "Cleaning build files..."
	find . \( -name "*.S" -o -name "*.o" -o -name "*.spike" \) -type f -delete
	rm -f $(TARGET)
	rm -rf ./output_files/*
	@echo "Done!"

clean_parameters:
	@echo "Cleaning parameters files..."
	rm -rf ./model_parameters/*.bin ./model_parameters/*.txt
	@echo "Done!"

clean_img_bin:
	@echo "Cleaning images binaries..."
	rm -rf ./image_binaries/*
	@echo "Done!"