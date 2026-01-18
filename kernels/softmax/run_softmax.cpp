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
            CHANNELS   = static_cast<uint64_t>(channels_arg);
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
    float* in  = new float[N];
    float* out = new float[N];
    float* tmp_in  = new float[CHANNELS]; // column buffer
    float* tmp_out = new float[CHANNELS];

    if (!in || !out || !tmp_in || !tmp_out) {
        cerr << "Memory allocation failed!" << endl;
        if (in)      delete[] in;
        if (out)     delete[] out;
        if (tmp_in)  delete[] tmp_in;
        if (tmp_out) delete[] tmp_out;
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

    // Write input data as flat [CHANNELS * INNER_SIZE]
    write_matrix_binary("./output_files/input.bin", in, N);

    for (uint64_t inner = 0; inner < INNER_SIZE; ++inner) {
        // Gather column 'inner' into tmp_in
        for (uint64_t c = 0; c < CHANNELS; ++c) {
            tmp_in[c] = in[c * INNER_SIZE + inner];
        }

        // Apply 1D softmax across channels
        softmax(tmp_in, tmp_out, CHANNELS);

        // Scatter back to out
        for (uint64_t c = 0; c < CHANNELS; ++c) {
            out[c * INNER_SIZE + inner] = tmp_out[c];
        }
    }

    write_matrix_binary("./output_files/softmax_out.bin", out, N);

    // --- CLEANUP ---
    delete[] in;
    delete[] out;
    delete[] tmp_in;
    delete[] tmp_out;

    return 0;
}