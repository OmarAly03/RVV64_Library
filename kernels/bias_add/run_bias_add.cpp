#include <iostream>
#include <cstdlib>
#include "./include/defs.h"

using namespace std;

// Helper to initialize matrix with random floats in [-1.0, 1.0]
void init_matrix(float* matrix, size_t count) {
    for (size_t i = 0; i < count; i++) {
        matrix[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t B = 1;  // batch_size
    size_t C = 16; // channels
    size_t H = 14; // height
    size_t W = 14; // width

    if (argc == 5) {
        B = static_cast<size_t>(atoi(argv[1]));
        C = static_cast<size_t>(atoi(argv[2]));
        H = static_cast<size_t>(atoi(argv[3]));
        W = static_cast<size_t>(atoi(argv[4]));
        if (B == 0 || C == 0 || H == 0 || W == 0) {
            cerr << "Invalid arguments. Using defaults." << endl;
            B = 1; C = 16; H = 14; W = 14;
        }
    } else {
        cerr << "Usage: " << argv[0] << " <B> <C> <H> <W>" << endl;
        cerr << "Using default sizes: B=" << B << ", C=" << C << ", H=" << H << ", W=" << W << endl;
    }

    size_t input_size = B * C * H * W;
    size_t output_size = B * C * H * W;
    size_t bias_size = C;

    // --- MEMORY ALLOCATION ---
    float* input = new float[input_size];
    float* bias = new float[bias_size];
    float* output = new float[output_size];

    if (!input || !bias || !output) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE INPUTS ---
    srand(0); // fixed seed
    init_matrix(input, input_size);
    init_matrix(bias, bias_size);

    cout << "Running BiasAdd with B=" << B << ", C=" << C << ", H=" << H << ", W=" << W << "..." << endl;
    
    // --- WRITE INPUTS TO FILE ---
    write_matrix_binary("./output_files/input.bin", input, input_size);
    write_matrix_binary("./output_files/bias.bin", bias, bias_size);

    // --- RUN KERNELS ---

    /***** Scalar BiasAdd *****/
    bias_add_scalar(input, bias, output, B, C, H, W);
    write_matrix_binary("./output_files/bias_add_scalar.bin", output, output_size);

    /***** BiasAdd Vectorized e32mx *****/
    bias_add_e32m1(input, bias, output, B, C, H, W);
    write_matrix_binary("./output_files/bias_add_e32m1.bin", output, output_size);

    bias_add_e32m2(input, bias, output, B, C, H, W);
    write_matrix_binary("./output_files/bias_add_e32m2.bin", output, output_size);

    bias_add_e32m4(input, bias, output, B, C, H, W);
    write_matrix_binary("./output_files/bias_add_e32m4.bin", output, output_size);

    bias_add_e32m8(input, bias, output, B, C, H, W);
    write_matrix_binary("./output_files/bias_add_e32m8.bin", output, output_size);

    // --- CLEANUP ---
    delete[] input;
    delete[] bias;
    delete[] output;

    return 0;
}