#include <iostream>
#include <vector>
#include <stdexcept> // For std::exception
#include "include/lenet5.hpp" 

// =======================================================
// MAIN INFERENCE APPLICATION
// =======================================================

// Helper function to load image data into a vector
std::vector<float> load_image_to_vector(const std::string& path) {
    std::vector<float> data(IN_SIZE);
    load_preprocessed_image(data, path.c_str());
    return data;
}

int main(int argc, char* argv[]) {
    try {
        // Check command line arguments
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <digit>" << std::endl;
            std::cerr << "Example: " << argv[0] << " 7" << std::endl;
            return 1;
        }

        // Parse the digit argument
        std::string digit_str = argv[1];
        
        // Validate it's a single digit (0-9)
        if (digit_str.length() != 1 || !std::isdigit(digit_str[0])) {
            std::cerr << "Error: Please provide a single digit (0-9)" << std::endl;
            return 1;
        }

        // 1. Create the model
        LeNet5 model("./model_parameters");

        // 2. Construct the image path using the digit
        std::string image_path = "./image_binaries/" + digit_str + ".bin";
        
        std::vector<float> image_data = load_image_to_vector(image_path);
        
        // 3. Run prediction
        int prediction = model.predict(image_data);
        
        // 4. Print the final result
        std::cout << "\nPrediction: " << prediction << std::endl;
        std::cout << "Expected: " << digit_str << std::endl;
        std::cout << "Result: " << (prediction == std::stoi(digit_str) ? "CORRECT" : "INCORRECT") << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}