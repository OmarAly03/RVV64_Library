#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    // --- SET PARAMS ---
    int N = 16, C = 1, H = 4, W = 4; 
    int KH = 2, KW = 2;
    int SH = 2, SW = 2;
    int PH = 0, PW = 0;

    if (argc == 5) {
        N = atoi(argv[1]);
        C = atoi(argv[2]);
        H = atoi(argv[3]);
        W = atoi(argv[4]);
    } else {
        cerr << "Usage: " << argv[0] << " <batch> <channels> <height> <width>. Using default 16x1x4x4" << endl;
    }

    // Standard output dimension calculation
    int OH = (H + 2 * PH - KH) / SH + 1;
    int OW = (W + 2 * PW - KW) / SW + 1;

    size_t input_size = (size_t)N * C * H * W;
    size_t output_size = (size_t)N * C * OH * OW;

    // --- MEMORY ALLOCATION ---
    float* in = new float[input_size];
    float* out_scalar = new float[output_size];
    float* out_rvv = new float[output_size];

    // --- INITIALIZE INPUT ---
    srand(42); 
    for (size_t i = 0; i < input_size; i++) {
        in[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f;
    }

    // Fix: Write the complete input array, not just N elements
    write_matrix_binary("./output_files/input.bin", in, input_size);
    
    // --- EXECUTE SCALAR ---
    maxpool_scalar(in, out_scalar, N, C, H, W, KH, KW, SH, SW, PH, PW);
    write_matrix_binary("./output_files/maxpool_scalar.bin", out_scalar, output_size);

    // --- EXECUTE RVV ---

    maxpool_e32m1(in, out_rvv, N, C, H, W, KH, KW, SH, SW, PH, PW);
    write_matrix_binary("./output_files/maxpool_e32m1.bin", out_rvv, output_size);

    maxpool_e32m2(in, out_rvv, N, C, H, W, KH, KW, SH, SW, PH, PW);
    write_matrix_binary("./output_files/maxpool_e32m2.bin", out_rvv, output_size);

    maxpool_e32m4(in, out_rvv, N, C, H, W, KH, KW, SH, SW, PH, PW);
    write_matrix_binary("./output_files/maxpool_e32m4.bin", out_rvv, output_size);

    maxpool_e32m8(in, out_rvv, N, C, H, W, KH, KW, SH, SW, PH, PW);
    write_matrix_binary("./output_files/maxpool_e32m8.bin", out_rvv, output_size);

    // --- CLEANUP ---
    delete[] in;
    delete[] out_scalar;
    delete[] out_rvv;

    return 0;
}