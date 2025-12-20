#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "./include/defs.h"

using namespace std;

void write_matrix_binary_float(const char* filename, const float* data, size_t size) {
    FILE* f = fopen(filename, "wb");
    if (!f) return; 
    fwrite(data, sizeof(float), size, f);
    fclose(f);
}

void write_matrix_binary_int64(const char* filename, const int64_t* data, size_t size) {
    FILE* f = fopen(filename, "wb");
    if (!f) return; 
    fwrite(data, sizeof(int64_t), size, f);
    fclose(f);
}

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t SIZE = 8;
    int AXIS = 0;
    
    if (argc >= 2) {
        int size = atoi(argv[1]);
        if (size > 0) {
            SIZE = static_cast<size_t>(size);
        } else {
            cerr << "Invalid size argument. Using default: " << SIZE << endl;
        }
    }
    
    if (argc >= 3) {
        AXIS = atoi(argv[2]);
        if (AXIS < 0 || AXIS > 1) {
            cerr << "Invalid axis argument (must be 0 or 1). Using default: 0" << endl;
            AXIS = 0;
        }
    }
    
    size_t data_rows = SIZE;
    size_t data_cols = SIZE;
    size_t indices_rows = SIZE / 2;
    size_t indices_cols = SIZE;
    size_t tile_size = (SIZE >= 16) ? 8 : 4;
    
    cout << "Running ScatterElements on " << data_rows << "x" << data_cols << " data";
    cout << ", axis=" << AXIS << endl;

    // --- MEMORY ALLOCATION ---
    float* data = new float[data_rows * data_cols];
    int64_t* indices = new int64_t[indices_rows * indices_cols];
    float* updates = new float[indices_rows * indices_cols];
    float* output = new float[data_rows * data_cols];
    
    if (!data || !indices || !updates || !output) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE TEST DATA ---
    srand(42); // fixed seed for reproducibility (matching Python)
    
    // Initialize data matrix
    for (size_t i = 0; i < data_rows * data_cols; i++) {
        // Random values in range [-1, 1]
        data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Initialize indices
    for (size_t i = 0; i < indices_rows * indices_cols; i++) {
        indices[i] = rand() % SIZE; // Random indices within bounds
    }
    
    // Initialize updates
    for (size_t i = 0; i < indices_rows * indices_cols; i++) {
        // Random values in range [-1, 1]
        updates[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    // Write input data for Python to use
    write_matrix_binary_float("./output_files/data.bin", data, data_rows * data_cols);
    write_matrix_binary_int64("./output_files/indices.bin", indices, indices_rows * indices_cols);
    write_matrix_binary_float("./output_files/updates.bin", updates, indices_rows * indices_cols);

    /***** Scalar Implementation *****/
    scatter_elements_scalar(data, indices, updates, output, 
                           data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/scatter_scalar.bin", output, data_rows * data_cols);

    scatter_elements_tiled_scalar(data, indices, updates, output,
                                  data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/scatter_tiled_scalar.bin", output, data_rows * data_cols);

    /***** Vectorized Implementations *****/
    scatter_elements_e32m1(data, indices, updates, output,
                          data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/scatter_e32m1.bin", output, data_rows * data_cols);

    scatter_elements_e32m2(data, indices, updates, output,
                          data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/scatter_e32m2.bin", output, data_rows * data_cols);

    scatter_elements_e32m4(data, indices, updates, output,
                          data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/scatter_e32m4.bin", output, data_rows * data_cols);

    scatter_elements_e32m8(data, indices, updates, output,
                          data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/scatter_e32m8.bin", output, data_rows * data_cols);

    /***** Tiled Vectorized Implementations *****/
    scatter_elements_tiled_e32m1(data, indices, updates, output,
                                data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/scatter_tiled_e32m1.bin", output, data_rows * data_cols);

    scatter_elements_tiled_e32m2(data, indices, updates, output,
                                data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/scatter_tiled_e32m2.bin", output, data_rows * data_cols);

    scatter_elements_tiled_e32m4(data, indices, updates, output,
                                data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/scatter_tiled_e32m4.bin", output, data_rows * data_cols);

    scatter_elements_tiled_e32m8(data, indices, updates, output,
                                data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/scatter_tiled_e32m8.bin", output, data_rows * data_cols);

    // --- CLEANUP ---
    delete[] data;
    delete[] indices;
    delete[] updates;
    delete[] output;

    return 0;
}
