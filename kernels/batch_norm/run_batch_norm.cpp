#include <iostream>
#include <cstdlib>
#include <cstring>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t N = 2, C = 3, H = 4, W = 4; 
    if (argc == 5) {
        int n = atoi(argv[1]);
        int c = atoi(argv[2]);
        int h = atoi(argv[3]);
        int w = atoi(argv[4]);
        if (n > 0 && c > 0 && h > 0 && w > 0) {
            N = static_cast<size_t>(n);
            C = static_cast<size_t>(c);
            H = static_cast<size_t>(h);
            W = static_cast<size_t>(w);
        } else {
            cerr << "Invalid arguments. Using default size " << N << "x" << C << "x" << H << "x" << W << endl;
        }
    } else {
        cerr << "Usage: " << argv[0] << " <batch> <channels> <height> <width>." << endl;
    }

    size_t input_size = N * C * H * W;
    size_t output_size = input_size;
    size_t param_size = C;

    // --- MEMORY ALLOCATION ---
    float* input_original = new float[input_size];  
    float* output = new float[output_size];
    float* scale = new float[param_size];    
    float* bias = new float[param_size];     
    float* mean = new float[param_size];     
    float* variance = new float[param_size]; 

    if (!input_original || !output || !scale || !bias || !mean || !variance) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE INPUT AND PARAMETERS ---
    srand(0); 
    for (size_t i = 0; i < input_size; i++) {
        input_original[i] = (static_cast<float>(rand()) / RAND_MAX) * 4.0f - 2.0f;
    }

    for (size_t c = 0; c < param_size; c++) {
        scale[c] = 1.0f;    
        bias[c] = 0.0f;     
        mean[c] = 0.0f;     
        variance[c] = 1.0f; 
    }

    float epsilon = 1e-05f;

    cout << "Running BatchNorm on " << N << "x" << C << "x" << H << "x" << W << " tensor..." << endl;

    // Write input data for Python verification
    write_matrix_binary("./output_files/input.bin", input_original, input_size);

    // --- EXECUTION ---
    // Note: We now pass input_original as the source and output as the destination.
    // There is no need to 'reset_input' because input_original is never modified.

    /***** Scalar BatchNorm *****/
    batch_norm_scalar(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_scalar.bin", output, output_size);

    batch_norm_tiled_scalar(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_tiled_scalar.bin", output, output_size);

    /***** BatchNorm Vectorized e32mx *****/
    batch_norm_e32m1(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_e32m1.bin", output, output_size);

    batch_norm_e32m2(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_e32m2.bin", output, output_size);

    batch_norm_e32m4(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_e32m4.bin", output, output_size);

    batch_norm_e32m8(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_e32m8.bin", output, output_size);

    /***** BatchNorm Tiled Vectorized e32mx *****/
    batch_norm_tiled_e32m1(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_tiled_e32m1.bin", output, output_size);

    batch_norm_tiled_e32m2(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_tiled_e32m2.bin", output, output_size);

    batch_norm_tiled_e32m4(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_tiled_e32m4.bin", output, output_size);

    batch_norm_tiled_e32m8(input_original, output, scale, bias, mean, variance, C, H, W, epsilon);
    write_matrix_binary("./output_files/batch_norm_tiled_e32m8.bin", output, output_size);

    cout << "C++ kernels completed." << endl;

    // --- CLEANUP ---
    delete[] input_original;
    delete[] output;
    delete[] scale;
    delete[] bias;
    delete[] mean;
    delete[] variance;

    return 0;
}