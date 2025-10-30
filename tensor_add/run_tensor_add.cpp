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
    size_t N = 16; 
    if (argc == 2) {
        int size = atoi(argv[1]);
        if (size > 0) {
            N = static_cast<size_t>(size);
        } else {
            cerr << "Invalid argument. Using default size " << N << endl;
        }
    } else {
        cerr << "Usage: " << argv[0] << " <size>. Using default size " << N << endl;
    }

    // --- MEMORY ALLOCATION ---
    float* input_a = new float[N];
    float* input_b = new float[N];
    float* output = new float[N];

    if (!input_a || !input_b || !output) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE INPUTS ---
    srand(0); // fixed seed
    init_matrix(input_a, N);
    init_matrix(input_b, N);

    cout << "Running TensorAdd on " << N << " elements..." << endl;
    
    // --- WRITE INPUTS TO FILE ---
    write_matrix_binary("./output_files/input_a.bin", input_a, N);
    write_matrix_binary("./output_files/input_b.bin", input_b, N);

    // --- RUN KERNELS ---

    /***** Scalar TensorAdd *****/
    tensor_add_scalar(input_a, input_b, output, N);
    write_matrix_binary("./output_files/tensor_add_scalar.bin", output, N);

    /***** TensorAdd Vectorized e32mx *****/
    tensor_add_e32m1(input_a, input_b, output, N);
    write_matrix_binary("./output_files/tensor_add_e32m1.bin", output, N);

    tensor_add_e32m2(input_a, input_b, output, N);
    write_matrix_binary("./output_files/tensor_add_e32m2.bin", output, N);

    tensor_add_e32m4(input_a, input_b, output, N);
    write_matrix_binary("./output_files/tensor_add_e32m4.bin", output, N);

    tensor_add_e32m8(input_a, input_b, output, N);
    write_matrix_binary("./output_files/tensor_add_e32m8.bin", output, N);

    // --- CLEANUP ---
    delete[] input_a;
    delete[] input_b;
    delete[] output;

    return 0;
}