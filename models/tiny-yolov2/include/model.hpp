#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <memory>

// --- Global Constants ---
const int NET_H = 416;
const int NET_W = 416;
const int GRID_H = 13;
const int GRID_W = 13;
const int NUM_ANCHORS = 5;
const int NUM_CLASSES = 20;

// --- Model Constants ---
const std::vector<std::string> LABELS = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

const std::vector<std::vector<float>> ANCHORS = {
    {1.08, 1.19}, {3.42, 4.41}, {6.63, 11.38},
    {9.42, 5.11}, {16.62, 10.52}
};

// --- Post-processing Thresholds ---
const float OBJECT_THRESHOLD = 0.4;
const float NMS_THRESHOLD = 0.3;

// --- Helper Structs ---
struct BoundingBox {
    float x, y, w, h;
    float score;
    int class_id;
};

// --- Utility Function (for loading weights) ---
// A simple helper to load a flat binary file into a vector
inline std::vector<float> load_weights_from_bin(const std::string& filepath, size_t expected_elements) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open weight file " << filepath << std::endl;
        exit(1);
    }
    
    std::vector<float> data(expected_elements);
    file.read(reinterpret_cast<char*>(data.data()), expected_elements * sizeof(float));
    
    if (file.gcount() != expected_elements * sizeof(float)) {
        std::cerr << "Error: Read " << file.gcount() << " bytes, but expected " << expected_elements * sizeof(float) << " from " << filepath << std::endl;
        exit(1);
    }
    
    file.close();
    return data;
}

// Overload for loading the input image (which is 1*3*416*416)
inline std::vector<float> load_input_image(const std::string& filepath) {
    const size_t elements = 1 * 3 * NET_H * NET_W;
    return load_weights_from_bin(filepath, elements);
}