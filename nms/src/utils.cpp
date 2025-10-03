#include <cstdio>
#include <cstddef>

void write_matrix_to_file(const char* filename, float* matrix, std::size_t rows, std::size_t cols) {
    FILE* f = fopen(filename, "w");
    if (!f) return; 
    
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            fprintf(f, "%.6f ", matrix[i * cols + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void write_matrix_binary(const char* filename, float* matrix, std::size_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) return; 
    
    fwrite(matrix, sizeof(float), count, f);
    fclose(f);
}