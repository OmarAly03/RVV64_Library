#include <iostream>
#include <cstdlib>
#include <cmath> // For fabs
#include "./include/defs.h"

using namespace std;

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
    float* in = new float[N];
    float* out = new float[N];

    if (!in || !out) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE INPUT FOR SOFTMAX ---
    srand(0); // fixed seed for reproducibility
    for (size_t i = 0; i < N; i++) {
        // in range [-2.0, +2.0]
        in[i] = (static_cast<float>(rand()) / RAND_MAX) * 4.0f - 2.0f;
    }

    cout << "Running Softmax on " << N << " elements..." << endl;
    cout << "Input range: [-2.0, +2.0]" << endl;

    // Write input data
    write_matrix_binary("./output_files/input.bin", in, N);

    /***** Scalar Softmax *****/
    softmax_scalar(in, out, N);
    write_matrix_binary("./output_files/softmax_scalar.bin", out, N);

    /***** Softmax Vectorized e32mx *****/
    softmax_e32m1(in, out, N);
    write_matrix_binary("./output_files/softmax_e32m1.bin", out, N);

    softmax_e32m2(in, out, N);
    write_matrix_binary("./output_files/softmax_e32m2.bin", out, N);

    softmax_e32m4(in, out, N);
    write_matrix_binary("./output_files/softmax_e32m4.bin", out, N);

    softmax_e32m8(in, out, N);
    write_matrix_binary("./output_files/softmax_e32m8.bin", out, N);

    // --- CLEANUP ---
    delete[] in;
    delete[] out;

    return 0;
}