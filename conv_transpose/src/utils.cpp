#include <fstream>
#include <iostream>
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
