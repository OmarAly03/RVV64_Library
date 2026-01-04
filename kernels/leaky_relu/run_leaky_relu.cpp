#include <iostream>
#include <cstdlib>
#include "./include/defs.h"

using namespace std;

#define TILE_SIZE 32

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t N = 16; // Default size
    if (argc == 2) {
        int n = atoi(argv[1]);
        if (n > 0) {
            N = static_cast<size_t>(n);
        } else {
            cerr << "Invalid size. Using default size " << N << endl;
        }
    } else {
        cerr << "Usage: " << argv[0] << " <size>. Using default size " << N << endl;
    }

    // --- MEMORY ALLOCATION ---
    float* in = new float[N];
    float* out = new float[N];

    if (!in || !out) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE INPUT FOR LEAKY RELU ---
    srand(0); // fixed seed for reproducibility
    for (size_t i = 0; i < N; i++) {
        // in range [-2.0, +2.0] to test both positive and negative values
        in[i] = (static_cast<float>(rand()) / RAND_MAX) * 4.0f - 2.0f;
    }

    cout << "Running LeakyReLU on " << N << " elements..." << endl;
    cout << "Input range: [-2.0, +2.0]" << endl;
    cout << "Alpha (negative slope): 0.01" << endl;

    // Write input data
    write_matrix_binary("./output_files/input.bin", in, N);

    float alpha = 0.01f; // LeakyReLU negative slope parameter

    /***** Scalar LeakyReLU *****/
    leaky_relu_scalar(in, out, N, alpha);
    write_matrix_binary("./output_files/leaky_relu_scalar.bin", out, N);

    /***** LeakyReLU Vectorized e32mx *****/
    leaky_relu_e32m1(in, out, N, alpha);
    write_matrix_binary("./output_files/leaky_relu_e32m1.bin", out, N);

    leaky_relu_e32m2(in, out, N, alpha);
    write_matrix_binary("./output_files/leaky_relu_e32m2.bin", out, N);

    leaky_relu_e32m4(in, out, N, alpha);
    write_matrix_binary("./output_files/leaky_relu_e32m4.bin", out, N);

    leaky_relu_e32m8(in, out, N, alpha);
    write_matrix_binary("./output_files/leaky_relu_e32m8.bin", out, N);

	leaky_relu_tiled_scalar(in, out, N, alpha, TILE_SIZE);
    write_matrix_binary("./output_files/leaky_relu_tiled_scalar.bin", out, N);

    /***** LeakyReLU Tiled Vectorized e32mx *****/
    leaky_relu_tiled_e32m1(in, out, N, alpha, TILE_SIZE);
    write_matrix_binary("./output_files/leaky_relu_tiled_e32m1.bin", out, N);

    leaky_relu_tiled_e32m2(in, out, N, alpha, TILE_SIZE);
    write_matrix_binary("./output_files/leaky_relu_tiled_e32m2.bin", out, N);

    leaky_relu_tiled_e32m4(in, out, N, alpha, TILE_SIZE);
    write_matrix_binary("./output_files/leaky_relu_tiled_e32m4.bin", out, N);

    leaky_relu_tiled_e32m8(in, out, N, alpha, TILE_SIZE);
    write_matrix_binary("./output_files/leaky_relu_tiled_e32m8.bin", out, N);

    // --- CLEANUP ---
    delete[] in;
    delete[] out;

    return 0;
}