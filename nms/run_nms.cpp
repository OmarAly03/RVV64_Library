#include <iostream>
#include <cstdlib>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t N = 16; // default size for NMS input
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

    // --- INITIALIZE INPUT FOR NMS ---
    srand(0); // fixed seed for reproducibility
    for (size_t i = 0; i < N; i++) {
        // in range [0.0, 1.0]
        in[i] = (static_cast<float>(rand()) / RAND_MAX);
    }

    cout << "Running NMS on " << N << " elements..." << endl;

    // Write input data
    write_matrix_binary("./output_files/input.bin", in, N);

    // --- RUN NMS IMPLEMENTATIONS ---
    
    // Scalar NMS
    nms_scalar(in, out, N);
    write_matrix_binary("./output_files/nms_scalar.bin", out, N);

    // NMS e32m1
    nms_e32m1(in, out, N);
    write_matrix_binary("./output_files/nms_e32m1.bin", out, N);

    // NMS e32m2
    nms_e32m2(in, out, N);
    write_matrix_binary("./output_files/nms_e32m2.bin", out, N);

    // NMS e32m4
    nms_e32m4(in, out, N);
    write_matrix_binary("./output_files/nms_e32m4.bin", out, N);

    // NMS e32m8
    nms_e32m8(in, out, N);
    write_matrix_binary("./output_files/nms_e32m8.bin", out, N);

    // --- CLEANUP ---
    delete[] in;
    delete[] out;

    return 0;
}