#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>   // For std::iota
#include <algorithm> // For std::max_element, std::distance
#include <cstddef>   // For size_t
#include <stdexcept> // For std::runtime_error
#include <cmath>     // For exp

#include <cassert>   // For assert()
#include <cstring>   // For std::memset
#include <cfloat>    // For FLT_MAX

#include "./include/defs.hpp"

// =======================================================
// MAIN INFERENCE APPLICATION
// =======================================================
int main() {
    try {
        // --- 2. Load All Weights From Disk ---
        std::cout << "Loading weights..." << std::endl;
        auto c1_w = load_weights("./model_parameters/c1.c1.c1.weight.bin");
        auto c1_b = load_weights("./model_parameters/c1.c1.c1.bias.bin");
        auto c2_1_w = load_weights("./model_parameters/c2_1.c2.c2.weight.bin");
        auto c2_1_b = load_weights("./model_parameters/c2_1.c2.c2.bias.bin");
        auto c2_2_w = load_weights("./model_parameters/c2_2.c2.c2.weight.bin");
        auto c2_2_b = load_weights("./model_parameters/c2_2.c2.c2.bias.bin");
        auto c3_w = load_weights("./model_parameters/c3.c3.c3.weight.bin");
        auto c3_b = load_weights("./model_parameters/c3.c3.c3.bias.bin");
        auto f4_w = load_weights("./model_parameters/f4.f4.f4.weight.bin");
        auto f4_b = load_weights("./model_parameters/f4.f4.f4.bias.bin");
        auto f5_w = load_weights("./model_parameters/f5.f5.f5.weight.bin");
        auto f5_b = load_weights("./model_parameters/f5.f5.f5.bias.bin");
        std::cout << "All 12 weights/biases loaded." << std::endl;

        // --- 3. Allocate Memory for Tensors ---
        std::vector<float> input_tensor(IN_SIZE);
        
        // C1 path
        std::vector<float> c1_out_nobias(C1_OUT_SIZE);
        std::vector<float> c1_out(C1_OUT_SIZE);
        std::vector<float> relu1_out(C1_OUT_SIZE);
        std::vector<float> pool1_out(POOL1_OUT_SIZE);
        std::vector<int64_t> indices_pool1(POOL1_OUT_SIZE);

        // Branch 1 (c2_1)
        std::vector<float> c2_1_out_nobias(C2_OUT_SIZE);
        std::vector<float> c2_1_out(C2_OUT_SIZE);
        std::vector<float> relu2_1_out(C2_OUT_SIZE);
        std::vector<float> pool2_1_out(POOL2_OUT_SIZE);
        std::vector<int64_t> indices_pool2_1(POOL2_OUT_SIZE);

        // Branch 2 (c2_2)
        std::vector<float> c2_2_out_nobias(C2_OUT_SIZE);
        std::vector<float> c2_2_out(C2_OUT_SIZE);
        std::vector<float> relu2_2_out(C2_OUT_SIZE);
        std::vector<float> pool2_2_out(POOL2_OUT_SIZE);
        std::vector<int64_t> indices_pool2_2(POOL2_OUT_SIZE);

        // Add node
        std::vector<float> add_out(ADD_OUT_SIZE);

        // C3 path
        std::vector<float> c3_out_nobias(C3_OUT_SIZE);
        std::vector<float> c3_out(C3_OUT_SIZE);
        std::vector<float> relu3_out(C3_OUT_SIZE);

        // FC path
        std::vector<float> f4_out(BATCH_SIZE * F4_OUT);
        std::vector<float> relu4_out(BATCH_SIZE * F4_OUT);
        std::vector<float> f5_out(BATCH_SIZE * F5_OUT);

        // Final output
        std::vector<float> final_output(BATCH_SIZE * F5_OUT);

        // --- 4. Load Input Image ---
        load_preprocessed_image(input_tensor, "./image_binaries/6.bin");

        // --- 5. Run Inference Pipeline ---
        std::cout << "Running inference..." << std::endl;

        // Layer 1: C1
        conv2d_scalar(input_tensor.data(), c1_w.data(), c1_out_nobias.data(),
                     BATCH_SIZE, C1_IN_C, C1_OUT_C, IN_H, IN_W, C1_K, C1_K, 1, 1, 0, 0);
		bias_add_scalar(c1_out_nobias.data(), c1_b.data(), c1_out.data(),
				BATCH_SIZE, C1_OUT_C, C1_OUT_H, C1_OUT_W);
		relu_scalar(c1_out.data(), relu1_out.data(), C1_OUT_SIZE);
        maxpool_scalar_tile(relu1_out.data(), pool1_out.data(), indices_pool1.data(),
                           BATCH_SIZE, C1_OUT_C, C1_OUT_H, C1_OUT_W, POOL1_K, POOL1_S, 
                           false, POOL1_OUT_H, POOL1_OUT_W, 0, 0, POOL1_OUT_H, POOL1_OUT_W);

        // --- PARALLEL BLOCK START ---
        // Branch 1: C2_1
        conv2d_scalar(pool1_out.data(), c2_1_w.data(), c2_1_out_nobias.data(),
                     BATCH_SIZE, C2_IN_C, C2_OUT_C, POOL1_OUT_H, POOL1_OUT_W, C2_K, C2_K, 1, 1, 0, 0);
		bias_add_scalar(c2_1_out_nobias.data(), c2_1_b.data(), c2_1_out.data(),
				BATCH_SIZE, C2_OUT_C, C2_OUT_H, C2_OUT_W);

		relu_scalar(c2_1_out.data(), relu2_1_out.data(), C2_OUT_SIZE);
        maxpool_scalar_tile(relu2_1_out.data(), pool2_1_out.data(), indices_pool2_1.data(),
                           BATCH_SIZE, C2_OUT_C, C2_OUT_H, C2_OUT_W, POOL2_K, POOL2_S, 
                           false, POOL2_OUT_H, POOL2_OUT_W, 0, 0, POOL2_OUT_H, POOL2_OUT_W);

        // Branch 2: C2_2
        conv2d_scalar(pool1_out.data(), c2_2_w.data(), c2_2_out_nobias.data(),
                     BATCH_SIZE, C2_IN_C, C2_OUT_C, POOL1_OUT_H, POOL1_OUT_W, C2_K, C2_K, 1, 1, 0, 0);
		bias_add_scalar(c2_2_out_nobias.data(), c2_2_b.data(), c2_2_out.data(),
					BATCH_SIZE, C2_OUT_C, C2_OUT_H, C2_OUT_W);
		relu_scalar(c2_2_out.data(), relu2_2_out.data(), C2_OUT_SIZE);
        maxpool_scalar_tile(relu2_2_out.data(), pool2_2_out.data(), indices_pool2_2.data(),
                           BATCH_SIZE, C2_OUT_C, C2_OUT_H, C2_OUT_W, POOL2_K, POOL2_S, 
                           false, POOL2_OUT_H, POOL2_OUT_W, 0, 0, POOL2_OUT_H, POOL2_OUT_W);
        // --- PARALLEL BLOCK END ---

        // Add node
        tensor_add_scalar(pool2_1_out.data(), pool2_2_out.data(), add_out.data(), ADD_OUT_SIZE);

        // Layer 3: C3 (The 5x5 Conv that acts as a Linear layer)
        conv2d_scalar(add_out.data(), c3_w.data(), c3_out_nobias.data(),
                     BATCH_SIZE, C3_IN_C, C3_OUT_C, POOL2_OUT_H, POOL2_OUT_W, C3_K, C3_K, 1, 1, 0, 0);
		bias_add_scalar(c3_out_nobias.data(), c3_b.data(), c3_out.data(),
				BATCH_SIZE, C3_OUT_C, C3_OUT_H, C3_OUT_W);
		relu_scalar(c3_out.data(), relu3_out.data(), C3_OUT_SIZE);

        // Layer 4: F4 (Flatten/Reshape is implicit, c3_out_size is 120)
		dense_scalar(relu3_out.data(), f4_w.data(), f4_b.data(),
		f4_out.data(), F4_IN, F4_OUT);

		relu_scalar(f4_out.data(), relu4_out.data(), f4_out.size());
        
        // Layer 5: F5
		dense_scalar(relu4_out.data(), f5_w.data(), f5_b.data(),
		f5_out.data(), F5_IN, F5_OUT);

        // Final Layer: LogSoftmax
		softmax_scalar(f5_out.data(), final_output.data(), F5_OUT);

        // --- 6. Find and Print Prediction ---
        // Argmax on the final probabilities
        auto max_it = std::max_element(final_output.begin(), final_output.end());
        int prediction = std::distance(final_output.begin(), max_it);

        std::cout << "\n--- Inference Complete ---" << std::endl;
        std::cout << "Final Output Log-Probabilities:" << std::endl;
        for (int i = 0; i < F5_OUT; ++i) {
            std::cout << "Class " << i << ": " << final_output[i] << std::endl;
        }
        std::cout << "\nPrediction: " << prediction << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}