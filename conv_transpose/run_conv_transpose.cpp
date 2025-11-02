#include <iostream>
#include <cstdlib>
#include <random>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    int batch_size = 1;
    int in_channels = 1;
    int out_channels = 1;
    int input_h = 4, input_w = 4;
    int kernel_h = 3, kernel_w = 3;
    int stride_h = 2, stride_w = 2;
    int pad_h = 0, pad_w = 0;
    
    if (argc >= 5) {
        input_h = input_w = atoi(argv[1]);
        in_channels = atoi(argv[2]);
        out_channels = atoi(argv[3]);
        pad_h = pad_w = atoi(argv[4]);
    } else if (argc >= 4) {
        input_h = input_w = atoi(argv[1]);
        in_channels = atoi(argv[2]);
        out_channels = atoi(argv[3]);
    } else if (argc >= 2) {
        input_h = input_w = atoi(argv[1]);
    }
    
    int out_height = (input_h - 1) * stride_h - 2 * pad_h + kernel_h;
    int out_width = (input_w - 1) * stride_w - 2 * pad_w + kernel_w;
    
    cout << "Transposed Convolution Configuration:" << endl;
    cout << "Input: " << batch_size << "x" << in_channels << "x" << input_h << "x" << input_w << endl;
    cout << "Kernel: " << in_channels << "x" << out_channels << "x" << kernel_h << "x" << kernel_w << endl;
    cout << "Output: " << batch_size << "x" << out_channels << "x" << out_height << "x" << out_width << endl;
    cout << "Stride: [" << stride_h << ", " << stride_w << "], Padding: [" << pad_h << ", " << pad_w << "]" << endl;
    
    size_t input_size = batch_size * in_channels * input_h * input_w;
    size_t kernel_size = in_channels * out_channels * kernel_h * kernel_w;
    size_t output_size = batch_size * out_channels * out_height * out_width;
    
    float* input = new float[input_size];
    float* kernel = new float[kernel_size];
    float* output = new float[output_size];
    
    if (!input || !kernel || !output) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }
    
    srand(0);
    for (size_t i = 0; i < input_size; i++) {
        input[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < kernel_size; i++) {
        kernel[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    
    write_matrix_binary("./output_files/input.bin", input, input_size);
    write_matrix_binary("./output_files/kernel.bin", kernel, kernel_size);
    
    conv_transpose_2d_scalar(input, kernel, output, batch_size, in_channels, out_channels,
                           input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    write_matrix_binary("./output_files/output_scalar.bin", output, output_size);
    
    #ifdef RVV_AVAILABLE
    conv_transpose_2d_e32m8(input, kernel, output, batch_size, in_channels, out_channels,
                          input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    write_matrix_binary("./output_files/output_e32m8.bin", output, output_size);
    #endif
    
    delete[] input;
    delete[] kernel;
    delete[] output;
    
    return 0;
}
