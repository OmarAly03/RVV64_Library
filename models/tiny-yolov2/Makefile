TARGET = main
IMG_PATH ?= images/mark1.jpg

OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
$(TARGET): main.cpp src/kernels.cpp src/model.cpp
	g++ main.cpp src/kernels.cpp src/model.cpp -Iinclude -o $(TARGET) -std=c++17 -O3 $(OPENCV_FLAGS)

run:
	./main images/mark1.jpg model_parameters/

extract_parameters: src/extract_weights.py
	python3 src/extract_weights.py

extract_image: src/extract_image.py
	python3 src/extract_image.py $(IMG_PATH)

clean:
	rm -f main

# For image extraction: 
#	make extract_image IMG_PATH=images/<>