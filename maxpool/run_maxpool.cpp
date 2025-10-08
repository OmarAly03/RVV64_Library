#include <iostream>
#include <cstdlib>
#include <vector>
#include "./include/defs.h"

int main(int argc, char* argv[]) {
    // --- Configuration ---
    size_t N = 1, C = 3, H = 32, W = 32, KERNEL_SIZE = 3, STRIDE = 2;
    bool CEIL_MODE = false; // Set to true to test ceil_mode

    if (argc >= 3) {
        H = static_cast<size_t>(atoi(argv[1]));
        W = static_cast<size_t>(atoi(argv[2]));
    }
    
    const size_t OH = CALC_OUT_DIM(H, KERNEL_SIZE, STRIDE, CEIL_MODE);
    const size_t OW = CALC_OUT_DIM(W, KERNEL_SIZE, STRIDE, CEIL_MODE);

    // --- Memory Allocation ---
    std::vector<float> X(N * C * H * W);
    std::vector<float> Y(N * C * OH * OW);
    std::vector<int64_t> I(N * C * OH * OW);

    // --- Load Input Tensor From File ---
    read_tensor_binary("./output_files/X.bin", X.data(), X.size());

    // --- Run ALL Kernels and Save ALL Outputs ---
    maxpool_scalar(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_scalar.bin", Y.data(), Y.size());
    write_tensor_binary_int64("./output_files/I_scalar.bin", I.data(), I.size());

    maxpool_e32m1(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m1.bin", Y.data(), Y.size());
    write_tensor_binary_int64("./output_files/I_e32m1.bin", I.data(), I.size());

    maxpool_e32m2(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m2.bin", Y.data(), Y.size());
    write_tensor_binary_int64("./output_files/I_e32m2.bin", I.data(), I.size());

    maxpool_e32m4(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m4.bin", Y.data(), Y.size());
    write_tensor_binary_int64("./output_files/I_e32m4.bin", I.data(), I.size());

    maxpool_e32m8(X.data(), Y.data(), I.data(), N, C, H, W, KERNEL_SIZE, STRIDE, CEIL_MODE);
    write_tensor_binary_float("./output_files/Y_e32m8.bin", Y.data(), Y.size());
    write_tensor_binary_int64("./output_files/I_e32m8.bin", I.data(), I.size());

    return 0;
}
