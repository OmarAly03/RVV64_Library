#include <iostream>
#include <cstdlib>
#include <vector>
#include "./include/defs.h"

int main(int argc, char* argv[]) {
    // --- Configuration ---
    size_t N = 1, C = 3, H = 32, W = 32, KERNEL_SIZE = 3, STRIDE = 2;
    bool CEIL_MODE = false;

    if (argc >= 3) {
        H = static_cast<size_t>(atoi(argv[1]));
        W = static_cast<size_t>(atoi(argv[2]));
    }
    
    // Output dimensions are now calculated inside the tiled functions
    size_t OH_approx = H / STRIDE; // Approximate for allocation
    size_t OW_approx = W / STRIDE;

    // --- Memory Allocation ---
    std::vector<float> X(N * C * H * W);
    // Allocate extra memory, as ceil_mode might slightly increase output size
    std::vector<float> Y(N * C * (OH_approx + TILE_H) * (OW_approx + TILE_W));
    std::vector<int64_t> I(N * C * (OH_approx + TILE_H) * (OW_approx + TILE_W));

    // --- Load Input Tensor ---
    read_tensor_binary("./output_files/X.bin", X.data(), X.size());

    // --- Run Tiled Kernels and Save Outputs ---
    // Note: The actual output size is determined by the kernel
    maxpool_scalar_tiled(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    size_t actual_OH = CALC_OUT_DIM(H, KERNEL_SIZE, STRIDE, CEIL_MODE); // Get actual size
    size_t actual_OW = CALC_OUT_DIM(W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    size_t output_size = N * C * actual_OH * actual_OW;
    write_tensor_binary_float("./output_files/Y_scalar.bin", Y.data(), output_size);
    write_tensor_binary_int64("./output_files/I_scalar.bin", I.data(), output_size);

    maxpool_e32m1_tiled(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m1.bin", Y.data(), output_size);
    write_tensor_binary_int64("./output_files/I_e32m1.bin", I.data(), output_size);

    maxpool_e32m2_tiled(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m2.bin", Y.data(), output_size);
    write_tensor_binary_int64("./output_files/I_e32m2.bin", I.data(), output_size);

    maxpool_e32m4_tiled(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m4.bin", Y.data(), output_size);
    write_tensor_binary_int64("./output_files/I_e32m4.bin", I.data(), output_size);

    maxpool_e32m8_tiled(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m8.bin", Y.data(), output_size);
    write_tensor_binary_int64("./output_files/I_e32m8.bin", I.data(), output_size);
    
    return 0;
}
