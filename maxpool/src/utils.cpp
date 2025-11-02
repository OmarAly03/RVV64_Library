#include <cstdio>
#include <cstddef>
#include <cstdint>

void read_tensor_binary(const char* filename, float* tensor, size_t count) {
    FILE* f = fopen(filename, "rb");
    if (!f) { 
        perror("Failed to open file for reading"); 
        return; 
    }
    
    // Capture the return value of fread
    size_t items_read = fread(tensor, sizeof(float), count, f);
    
    // Check if the number of items read matches what we expected
    if (items_read != count) {
        fprintf(stderr, "Warning: Read error in file %s. Expected %ld items, but read %ld.\n", 
                filename, (long)count, (long)items_read);
    }
    
    fclose(f);
}

void write_tensor_binary_float(const char* filename, const float* tensor, size_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) { 
        perror("Failed to open file for writing"); 
        return; 
    }
    fwrite(tensor, sizeof(float), count, f);
    fclose(f);
}

void write_tensor_binary_int64(const char* filename, const int64_t* tensor, size_t count) {
    FILE* f = fopen(filename, "wb");
    if (!f) { 
        perror("Failed to open file for writing"); 
        return; 
    }
    fwrite(tensor, sizeof(int64_t), count, f);
    fclose(f);
}
