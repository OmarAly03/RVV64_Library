#include <cstdio>
#include <cstddef>
#include <cstdint>

using namespace std;

void write_matrix_to_file(const char* filename, float* matrix, size_t rows, size_t cols) {
    FILE* f = fopen(filename, "w");
    if (!f) return; 
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            fprintf(f, "%.6f ", matrix[i * cols + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void write_matrix_binary(const char* filename, float* matrix, size_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) return; 
    
    fwrite(matrix, sizeof(float), count, f);
    fclose(f);
}