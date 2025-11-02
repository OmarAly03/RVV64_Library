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
    size_t IN = 16; // in_features
    size_t OUT = 16; // out_features

    if (argc == 4) {
        B = static_cast<size_t>(atoi(argv[1]));
        IN = static_cast<size_t>(atoi(argv[2]));
        OUT = static_cast<size_t>(atoi(argv[3]));
        if (B == 0 || IN == 0 || OUT == 0) {
            cerr << "Invalid arguments. Using defaults." << endl;
            B = 1; IN = 16; OUT = 16;
        }
    } else {
        cerr << "Usage: " << argv[0] << " <batch_size> <in_features> <out_features>" << endl;
        cerr << "Using default sizes: B=" << B << ", IN=" << IN << ", OUT=" << OUT << endl;
    }

    // --- MEMORY ALLOCATION ---
    float* input = new float[B * IN];
    float* weights = new float[OUT * IN];
    float* bias = new float[OUT];
    float* output = new float[B * OUT];

    if (!input || !weights || !bias || !output) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE INPUTS ---
    srand(0); // fixed seed
    init_matrix(input, B * IN);
    init_matrix(weights, OUT * IN);
    init_matrix(bias, OUT);

    cout << "Running Dense (GEMM) with B=" << B << ", IN=" << IN << ", OUT=" << OUT << "..." << endl;
    cout << "Input range: [-1.0, +1.0]" << endl;

    // --- WRITE INPUTS TO FILE ---
    write_matrix_binary("./output_files/input.bin", input, B * IN);
    write_matrix_binary("./output_files/weights.bin", weights, OUT * IN);
    write_matrix_binary("./output_files/bias.bin", bias, OUT);

    // --- RUN KERNELS ---

    /***** Scalar Dense *****/
    dense_scalar(input, weights, bias, output, B, IN, OUT);
    write_matrix_binary("./output_files/dense_scalar.bin", output, B * OUT);

    /***** Dense Vectorized e32mx *****/
    dense_e32m1(input, weights, bias, output, B, IN, OUT);
    write_matrix_binary("./output_files/dense_e32m1.bin", output, B * OUT);

    dense_e32m2(input, weights, bias, output, B, IN, OUT);
    write_matrix_binary("./output_files/dense_e32m2.bin", output, B * OUT);

    dense_e32m4(input, weights, bias, output, B, IN, OUT);
    write_matrix_binary("./output_files/dense_e32m4.bin", output, B * OUT);

    dense_e32m8(input, weights, bias, output, B, IN, OUT);
    write_matrix_binary("./output_files/dense_e32m8.bin", output, B * OUT);

    // --- CLEANUP ---
    delete[] input;
    delete[] weights;
    delete[] bias;
    delete[] output;

    return 0;
}