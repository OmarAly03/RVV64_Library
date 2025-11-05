#include <iostream>
#include <cstdlib>
#include <cmath> // For fabs
#include <cstdint>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    uint64_t CHANNELS = 4; 
    uint64_t INNER_SIZE = 128;

    if (argc == 3) {
        int channels_arg = atoi(argv[1]);
        int inner_size_arg = atoi(argv[2]);
        if (channels_arg > 0 && inner_size_arg > 0) {
            CHANNELS = static_cast<uint64_t>(channels_arg);
            INNER_SIZE = static_cast<uint64_t>(inner_size_arg);
        } else {
            cerr << "Invalid arguments. Using default size " 
                 << CHANNELS << "x" << INNER_SIZE << endl;
        }
    } else {
        cerr << "Usage: " << argv[0] << " <channels> <innerSize>. Using default size "
             << CHANNELS << "x" << INNER_SIZE << endl;
    }

    size_t N = CHANNELS * INNER_SIZE;

    // --- MEMORY ALLOCATION ---
    float* in = new float[N];
    float* out = new float[N];
    // Scalar version requires a temporary buffer
    float* buf = new float[INNER_SIZE];

    if (!in || !out || !buf) {
        cerr << "Memory allocation failed!" << endl;
        if(in) delete[] in;
        if(out) delete[] out;
        if(buf) delete[] buf;
        return 1;
    }

    // --- INITIALIZE INPUT FOR SOFTMAX ---
    srand(0); // fixed seed for reproducibility
    for (size_t i = 0; i < N; i++) {
        // in range [-2.0, +2.0]
        in[i] = (static_cast<float>(rand()) / RAND_MAX) * 4.0f - 2.0f;
    }

    cout << "Running Softmax on " << CHANNELS << " channels, " 
         << INNER_SIZE << " inner size (Total " << N << " elements)..." << endl;
    cout << "Input range: [-2.0, +2.0]" << endl;

    // Write input data
    write_matrix_binary("./output_files/input.bin", in, N);

    /***** Scalar Softmax *****/
    softmax(in, out, buf, CHANNELS, INNER_SIZE);
    write_matrix_binary("./output_files/softmax_scalar.bin", out, N);

    /***** Vectorized Softmax *****/
    softmax_vec(in, out, CHANNELS, INNER_SIZE);
    write_matrix_binary("./output_files/softmax_vector.bin", out, N);

    // --- CLEANUP ---
    delete[] in;
    delete[] out;
    delete[] buf;

    return 0;
}