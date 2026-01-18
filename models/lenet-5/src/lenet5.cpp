#include "../include/lenet5.hpp"
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <cstring>   // For std::memcpy
#include <algorithm> // For std::max_element
#include <numeric>   // For std::distance

#include "../include/config.hpp"

// --- Constructor Implementation ---
LeNet5::LeNet5(const std::string& model_path) {
    std::cout << "Loading weights..." << std::endl;
    c1_w = load_weights(model_path + "/c1.c1.c1.weight.bin");
    c1_b = load_weights(model_path + "/c1.c1.c1.bias.bin");
    c2_1_w = load_weights(model_path + "/c2_1.c2.c2.weight.bin");
    c2_1_b = load_weights(model_path + "/c2_1.c2.c2.bias.bin");
    c2_2_w = load_weights(model_path + "/c2_2.c2.c2.weight.bin");
    c2_2_b = load_weights(model_path + "/c2_2.c2.c2.bias.bin");
    c3_w = load_weights(model_path + "/c3.c3.c3.weight.bin");
    c3_b = load_weights(model_path + "/c3.c3.c3.bias.bin");
    f4_w = load_weights(model_path + "/f4.f4.f4.weight.bin");
    f4_b = load_weights(model_path + "/f4.f4.f4.bias.bin");
    f5_w = load_weights(model_path + "/f5.f5.f5.weight.bin");
    f5_b = load_weights(model_path + "/f5.f5.f5.bias.bin");
    std::cout << "All 12 weights/biases loaded." << std::endl;

    // --- Allocate Memory for Tensors ---
    input_tensor.resize(IN_SIZE);
    
    c1_out_nobias.resize(C1_OUT_SIZE);
    c1_out.resize(C1_OUT_SIZE);
    relu1_out.resize(C1_OUT_SIZE);
    pool1_out.resize(POOL1_OUT_SIZE);
    indices_pool1.resize(POOL1_OUT_SIZE);

    c2_1_out_nobias.resize(C2_OUT_SIZE);
    c2_1_out.resize(C2_OUT_SIZE);
    relu2_1_out.resize(C2_OUT_SIZE);
    pool2_1_out.resize(POOL2_OUT_SIZE);
    indices_pool2_1.resize(POOL2_OUT_SIZE);

    c2_2_out_nobias.resize(C2_OUT_SIZE);
    c2_2_out.resize(C2_OUT_SIZE);
    relu2_2_out.resize(C2_OUT_SIZE);
    pool2_2_out.resize(POOL2_OUT_SIZE);
    indices_pool2_2.resize(POOL2_OUT_SIZE);

    add_out.resize(ADD_OUT_SIZE);

    c3_out_nobias.resize(C3_OUT_SIZE);
    c3_out.resize(C3_OUT_SIZE);
    relu3_out.resize(C3_OUT_SIZE);

    f4_out.resize(BATCH_SIZE * F4_OUT);
    relu4_out.resize(BATCH_SIZE * F4_OUT);
    f5_out.resize(BATCH_SIZE * F5_OUT);

    final_output.resize(BATCH_SIZE * F5_OUT);
}

int LeNet5::predict(const std::vector<float>& image_data) {
    if (image_data.size() != IN_SIZE) {
        throw std::runtime_error("Input image data has incorrect size.");
    }
    
    std::memcpy(input_tensor.data(), image_data.data(), IN_SIZE * sizeof(float));

    // --- Layer 1: C1 ---
    // Fixed: Input -> Output -> Weights
    conv2d(input_tensor.data(), c1_out_nobias.data(), c1_w.data(),
           BATCH_SIZE, C1_IN_C, IN_H, IN_W, C1_OUT_C, C1_K, C1_K, 1, 1, 0, 0);
    
    // Fixed: Removed BATCH_SIZE, combined H*W for channel_size
    bias_add(c1_out_nobias.data(), c1_b.data(), c1_out.data(), 
             C1_OUT_C, C1_OUT_H * C1_OUT_W);
    
    relu(c1_out.data(), relu1_out.data(), C1_OUT_SIZE);
    
    // Fixed: Passed separate K and S for H/W
    maxpool(relu1_out.data(), pool1_out.data(), 
            BATCH_SIZE, C1_OUT_C, C1_OUT_H, C1_OUT_W, 
            POOL1_K, POOL1_K, POOL1_S, POOL1_S, 0, 0);

    // --- Branch 1: C2_1 ---
    conv2d(pool1_out.data(), c2_1_out_nobias.data(), c2_1_w.data(),
           BATCH_SIZE, C2_IN_C, POOL1_OUT_H, POOL1_OUT_W, C2_OUT_C, C2_K, C2_K, 1, 1, 0, 0);
    
    bias_add(c2_1_out_nobias.data(), c2_1_b.data(), c2_1_out.data(),
             C2_OUT_C, C2_OUT_H * C2_OUT_W);
    
    relu(c2_1_out.data(), relu2_1_out.data(), C2_OUT_SIZE);

    maxpool(relu2_1_out.data(), pool2_1_out.data(), 
            BATCH_SIZE, C2_OUT_C, C2_OUT_H, C2_OUT_W, 
            POOL2_K, POOL2_K, POOL2_S, POOL2_S, 0, 0);

    // --- Branch 2: C2_2 ---
    conv2d(pool1_out.data(), c2_2_out_nobias.data(), c2_2_w.data(),
           BATCH_SIZE, C2_IN_C, POOL1_OUT_H, POOL1_OUT_W, C2_OUT_C, C2_K, C2_K, 1, 1, 0, 0);
    
    bias_add(c2_2_out_nobias.data(), c2_2_b.data(), c2_2_out.data(),
             C2_OUT_C, C2_OUT_H * C2_OUT_W);
    
    relu(c2_2_out.data(), relu2_2_out.data(), C2_OUT_SIZE);

    maxpool(relu2_2_out.data(), pool2_2_out.data(), 
            BATCH_SIZE, C2_OUT_C, C2_OUT_H, C2_OUT_W, 
            POOL2_K, POOL2_K, POOL2_S, POOL2_S, 0, 0);

    // --- Combine and Output ---
    tensor_add(pool2_1_out.data(), pool2_2_out.data(), add_out.data(), ADD_OUT_SIZE);

    conv2d(add_out.data(), c3_out_nobias.data(), c3_w.data(),
           BATCH_SIZE, C3_IN_C, POOL2_OUT_H, POOL2_OUT_W, C3_OUT_C, C3_K, C3_K, 1, 1, 0, 0);
    
    bias_add(c3_out_nobias.data(), c3_b.data(), c3_out.data(),
             C3_OUT_C, C3_OUT_H * C3_OUT_W);
    
    relu(c3_out.data(), relu3_out.data(), C3_OUT_SIZE);

    dense(relu3_out.data(), f4_w.data(), f4_b.data(), f4_out.data(), F4_IN, F4_OUT);
    relu(f4_out.data(), relu4_out.data(), f4_out.size());
    
    dense(relu4_out.data(), f5_w.data(), f5_b.data(), f5_out.data(), F5_IN, F5_OUT);

    softmax(f5_out.data(), final_output.data(), F5_OUT);

    auto max_it = std::max_element(final_output.begin(), final_output.end());
    return std::distance(final_output.begin(), max_it);
}