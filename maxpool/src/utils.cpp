#include <cstdio>
#include <cstddef>
#include <cstdint>

void read_tensor_binary(const char* filename, float* tensor, size_t count) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("Failed to open file for reading"); return; }
    fread(tensor, sizeof(float), count, f);
    fclose(f);
}

void write_tensor_binary_float(const char* filename, const float* tensor, size_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("Failed to open file for writing"); return; }
    fwrite(tensor, sizeof(float), count, f);
    fclose(f);
}

void write_tensor_binary_int64(const char* filename, const int64_t* tensor, size_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("Failed to open file for writing"); return; }
    fwrite(tensor, sizeof(int64_t), count, f);
    fclose(f);
}

