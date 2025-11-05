#include <iostream>
#include <vector>
#include <stdexcept> // For std::exception
#include "include/lenet5.hpp" // <-- Include your new class header

// =======================================================
// MAIN INFERENCE APPLICATION
// =======================================================

// Helper function to load image data into a vector
std::vector<float> load_image_to_vector(const std::string& path) {
    std::vector<float> data(IN_SIZE);
    // Assumes the function prototype is:
    // void load_preprocessed_image(std::vector<float>& tensor, const char* filename);
    load_preprocessed_image(data, path.c_str());
    return data;
}

int main() {
    try {
        // 1. Create the model.
        LeNet5 model("./model_parameters");

        // 2. Load the input image data
        std::vector<float> image_data = load_image_to_vector("./image_binaries/0.bin");
        
        // 3. Run prediction
        int prediction = model.predict(image_data);
        
        // 4. Print the final result
        std::cout << "\nPrediction: " << prediction << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}