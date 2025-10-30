#ifndef LENET5_HPP
#define LENET5_HPP

#include <vector>
#include <string>
#include "defs.hpp"

class LeNet5 {
private:
    // --- Model Parameters (Weights & Biases) ---
    std::vector<float> c1_w, c1_b;
    std::vector<float> c2_1_w, c2_1_b;
    std::vector<float> c2_2_w, c2_2_b;
    std::vector<float> c3_w, c3_b;
    std::vector<float> f4_w, f4_b;
    std::vector<float> f5_w, f5_b;

    // --- Intermediate Tensors (Activations) ---
    std::vector<float> input_tensor;
    std::vector<float> c1_out_nobias, c1_out, relu1_out, pool1_out;
    std::vector<int64_t> indices_pool1;
    std::vector<float> c2_1_out_nobias, c2_1_out, relu2_1_out, pool2_1_out;
    std::vector<int64_t> indices_pool2_1;
    std::vector<float> c2_2_out_nobias, c2_2_out, relu2_2_out, pool2_2_out;
    std::vector<int64_t> indices_pool2_2;
    std::vector<float> add_out;
    std::vector<float> c3_out_nobias, c3_out, relu3_out;
    std::vector<float> f4_out, relu4_out, f5_out;
    std::vector<float> final_output;

public:
    /**
     * @brief Constructor: Loads all weights and allocates memory for tensors.
     * @param model_path Path to the directory containing weight/bias .bin files.
     */
    LeNet5(const std::string& model_path);

    /**
     * @brief Runs the complete inference pipeline on a single input image.
     * @param image_data A vector<float> containing the raw image data.
     * @return The predicted class index (0-9).
     */
    int predict(const std::vector<float>& image_data);
};

#endif // LENET5_HPP