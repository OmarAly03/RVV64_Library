#include <iostream>
#include <cstdlib>
#include <cstdio>
#include "./include/defs.h"

using namespace std;

static void write_matrix_binary_float(const char* filename, const float* data, size_t size) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(data, sizeof(float), size, f);
    fclose(f);
}

static void write_matrix_binary_int64(const char* filename, const int64_t* data, size_t size) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(data, sizeof(int64_t), size, f);
    fclose(f);
}

int main(int argc, char* argv[]) {
    size_t SIZE = 8;
    int AXIS = 0;

    if (argc >= 2) {
        int size = atoi(argv[1]);
        if (size > 0) SIZE = static_cast<size_t>(size);
    }
    if (argc >= 3) {
        AXIS = atoi(argv[2]);
        if (AXIS < 0 || AXIS > 1) AXIS = 0;
    }

    size_t data_rows = SIZE;
    size_t data_cols = SIZE;
    size_t indices_rows = SIZE / 2;
    size_t indices_cols = SIZE;
    size_t tile_size = (SIZE >= 16) ? 8 : 4;

    cout << "Running Gather on " << data_rows << "x" << data_cols << " data, axis=" << AXIS << endl;

    float* data = new float[data_rows * data_cols];
    int64_t* indices = new int64_t[indices_rows * indices_cols];
    float* output = new float[indices_rows * indices_cols];

    if (!data || !indices || !output) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < data_rows * data_cols; i++) {
        data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < indices_rows * indices_cols; i++) {
        indices[i] = rand() % (AXIS == 0 ? (int)data_rows : (int)data_cols);
    }

    // inputs for Python validation
    write_matrix_binary_float("./output_files/data.bin", data, data_rows * data_cols);
    write_matrix_binary_int64("./output_files/indices.bin", indices, indices_rows * indices_cols);

    // scalar
    gather_scalar(data, indices, output,
                  data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/gather_scalar.bin", output, indices_rows * indices_cols);

    // tiled scalar
    gather_tiled_scalar(data, indices, output,
                        data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/gather_tiled_scalar.bin", output, indices_rows * indices_cols);

    // vector m1
    gather_e32m1(data, indices, output,
                 data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/gather_e32m1.bin", output, indices_rows * indices_cols);

    // tiled vector m1
    gather_tiled_e32m1(data, indices, output,
                       data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/gather_tiled_e32m1.bin", output, indices_rows * indices_cols);

    // vector m2
    gather_e32m2(data, indices, output,
                 data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/gather_e32m2.bin", output, indices_rows * indices_cols);

    // vector m4
    gather_e32m4(data, indices, output,
                 data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/gather_e32m4.bin", output, indices_rows * indices_cols);

    // vector m8
    gather_e32m8(data, indices, output,
                 data_rows, data_cols, indices_rows, indices_cols, AXIS);
    write_matrix_binary_float("./output_files/gather_e32m8.bin", output, indices_rows * indices_cols);

    // tiled vector m2
    gather_tiled_e32m2(data, indices, output,
                       data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/gather_tiled_e32m2.bin", output, indices_rows * indices_cols);

    // tiled vector m4
    gather_tiled_e32m4(data, indices, output,
                       data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/gather_tiled_e32m4.bin", output, indices_rows * indices_cols);

    // tiled vector m8
    gather_tiled_e32m8(data, indices, output,
                       data_rows, data_cols, indices_rows, indices_cols, AXIS, tile_size);
    write_matrix_binary_float("./output_files/gather_tiled_e32m8.bin", output, indices_rows * indices_cols);

    delete[] data;
    delete[] indices;
    delete[] output;
    return 0;
}


