// main.cpp
#include "yolo_model.hpp"
#include <chrono>
#include <iostream>
#include <fstream>

// Simple function to save detection results to a text file
void save_detection_results(const std::vector<BoundingBox>& boxes, const std::string& output_path) {
    std::ofstream outfile(output_path);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open output file " << output_path << std::endl;
        return;
    }
    
    outfile << "Detection Results:\n";
    outfile << "==================\n\n";
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        outfile << "Detection " << (i + 1) << ":\n";
        outfile << "  Class: " << LABELS[box.class_id] << " (ID: " << box.class_id << ")\n";
        outfile << "  Confidence: " << box.score << "\n";
        outfile << "  Center: (" << box.x << ", " << box.y << ")\n";
        outfile << "  Size: " << box.w << " x " << box.h << "\n";
        
        // Convert to corner coordinates
        float x_min = box.x - box.w / 2.0f;
        float y_min = box.y - box.h / 2.0f;
        float x_max = box.x + box.w / 2.0f;
        float y_max = box.y + box.h / 2.0f;
        outfile << "  Bounding Box: (" << x_min << ", " << y_min << ") to (" << x_max << ", " << y_max << ")\n\n";
    }
    
    outfile.close();
    std::cout << "Detection results saved to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_bin_file> <weights_directory>" << std::endl;
        return -1;
    }

    std::string input_bin_path = argv[1];
    std::string weights_dir = argv[2];
    
    // 1. Load the pre-processed input tensor
    std::cout << "Loading input tensor from " << input_bin_path << "..." << std::endl;
    std::vector<float> input_tensor = load_input_image(input_bin_path);
    if (input_tensor.size() != 1 * 3 * NET_H * NET_W) {
        std::cerr << "Error: Input tensor has wrong size. Expected: " 
                  << (1 * 3 * NET_H * NET_W) << ", Got: " << input_tensor.size() << std::endl;
        return -1;
    }

    // 2. Load all model weights
    std::cout << "Loading model weights from " << weights_dir << "..." << std::endl;
    ModelWeights weights;
    load_all_weights(weights, weights_dir);

    // 3. Run inference
    std::cout << "Running inference..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<BoundingBox> final_boxes = yolo_model_inference(weights, input_tensor);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Inference (incl. post-processing) took: " << duration.count() << " ms" << std::endl;

    // 4. Save detection results
    std::cout << "Found " << final_boxes.size() << " final detections!" << std::endl;
    
    if (final_boxes.empty()) {
        std::cout << "No objects detected." << std::endl;
    } else {
        std::cout << "\nDetected objects:" << std::endl;
        for (size_t i = 0; i < final_boxes.size(); ++i) {
            const auto& box = final_boxes[i];
            std::cout << "  " << (i + 1) << ". " << LABELS[box.class_id] 
                      << " (confidence: " << box.score << ")" << std::endl;
        }
    }
    
    // Generate output filename based on input
    std::string output_path = "./output_files/detection_results.txt";
    save_detection_results(final_boxes, output_path);
    
    return 0;
}