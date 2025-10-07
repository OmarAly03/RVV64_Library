#include <fstream>
#include <iostream>
#include <cstring>
#include "defs.h"

void write_matrix_binary(const char* filename, const float* data, size_t count) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    file.close();
}

void write_conv_transpose_output_binary(const char* filename, const float* data, 
                                       int batch_size, int out_channels, int out_h, int out_w) {
    size_t total_elements = batch_size * out_channels * out_h * out_w;
    write_matrix_binary(filename, data, total_elements);
}
