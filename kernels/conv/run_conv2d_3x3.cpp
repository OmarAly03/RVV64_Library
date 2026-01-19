#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>
#include "./include/defs.h"

using namespace std;

// Helper: compute output spatial dims
static inline void compute_out_hw(int in_h, int in_w, int k_h, int k_w,
                                  int stride_h, int stride_w, int pad_h, int pad_w,
                                  int& out_h, int& out_w) {
    out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
}

struct TimingResult {
    double elapsed_ms;
};

TimingResult time_function(function<void()> fn) {
    auto start = chrono::high_resolution_clock::now();
    fn();
    auto end = chrono::high_resolution_clock::now();
    double elapsed_ms = chrono::duration<double, milli>(end - start).count();
    return {elapsed_ms};
}

int main(int argc, char* argv[]) {
    // Defaults: 128x128 single channel, 3x3 kernel with padding
    int H = 128;          // input height
    int W = 128;          // input width
    int kH = 3;           // kernel height
    int kW = 3;           // kernel width
    int sH = 1;           // stride height
    int sW = 1;           // stride width
    int pH = 1;           // pad height (for 3x3 to maintain size)
    int pW = 1;           // pad width
    int use_padding = 1;  // use zero-padding in 3x3 funcs

    // Parse arguments: H W [kH kW sH sW pH pW use_padding]
    if (argc >= 3) {
        H = max(1, atoi(argv[1]));
        W = max(1, atoi(argv[2]));
    }
    if (argc >= 5) {
        kH = max(1, atoi(argv[3]));
        kW = max(1, atoi(argv[4]));
    }
    if (argc >= 7) {
        sH = max(1, atoi(argv[5]));
        sW = max(1, atoi(argv[6]));
    }
    if (argc >= 9) {
        pH = max(0, atoi(argv[7]));
        pW = max(0, atoi(argv[8]));
    }
    if (argc >= 10) {
        use_padding = atoi(argv[9]);
    }

    // For 3x3 single channel, output size depends on use_padding
    int out_h = use_padding ? H : (H - 2);
    int out_w = use_padding ? W : (W - 2);

    cout << "Conv2D 3x3 Test: Input HxW=" << H << "x" << W
         << ", Kernel=" << kH << "x" << kW
         << ", Stride=(" << sH << "," << sW << ")"
         << ", Pad=(" << pH << "," << pW << ")"
         << ", use_padding=" << use_padding << "\n";
    cout << "Output: " << out_h << "x" << out_w << "\n\n";

    // Allocate input and kernel (single channel)
    float* input = new float[H * W];
    float* kernel = new float[kH * kW];
    
    // Allocate output buffers for all variants
    float* out_m1 = new float[out_h * out_w];
    float* out_m2 = new float[out_h * out_w];
    float* out_m4 = new float[out_h * out_w];
    float* out_m8 = new float[out_h * out_w];
    float* out_m2_batched = new float[out_h * out_w];
    float* out_m4_batched = new float[out_h * out_w];
    float* out_m8_batched = new float[out_h * out_w];

    if (!input || !kernel || !out_m1 || !out_m2 || !out_m4 || !out_m8 ||
        !out_m2_batched || !out_m4_batched || !out_m8_batched) {
        cerr << "Allocation failed" << endl;
        return 1;
    }

    // Initialize reproducibly with small values for better numerical stability
    srand(42);
    for (int i = 0; i < H * W; ++i) {
        input[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f - 0.05f;
    }
    for (int i = 0; i < kH * kW; ++i) {
        kernel[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.1f - 0.05f;
    }

    cout << "Input range: [" << *min_element(input, input + H*W) 
         << ", " << *max_element(input, input + H*W) << "]\n";
    cout << "Kernel range: [" << *min_element(kernel, kernel + kH*kW)
         << ", " << *max_element(kernel, kernel + kH*kW) << "]\n\n";

    // Persist inputs so Python can read them
    write_matrix_binary("./output_files/input_3x3.bin", input, static_cast<size_t>(H * W));
    write_matrix_binary("./output_files/kernel_3x3.bin", kernel, static_cast<size_t>(kH * kW));

    // Run all variants with timing
    cout << "Running implementations...\n";
    cout << "==================================================\n\n";

    // M1 - Non-batched
    cout << "1. M1 (non-batched)... ";
    cout.flush();
    TimingResult tm1 = time_function([&]() {
        conv2d_3x3_m1(input, kernel, out_m1, H, W, use_padding != 0);
    });
    cout << tm1.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m1.bin", out_m1, static_cast<size_t>(out_h * out_w));

    // M2 - Non-batched
    cout << "2. M2 (non-batched)... ";
    cout.flush();
    TimingResult tm2 = time_function([&]() {
        conv2d_3x3_m2(input, kernel, out_m2, H, W, use_padding != 0);
    });
    cout << tm2.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m2.bin", out_m2, static_cast<size_t>(out_h * out_w));

    // M4 - Non-batched
    cout << "3. M4 (non-batched)... ";
    cout.flush();
    TimingResult tm4 = time_function([&]() {
        conv2d_3x3_m4(input, kernel, out_m4, H, W, use_padding != 0);
    });
    cout << tm4.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m4.bin", out_m4, static_cast<size_t>(out_h * out_w));

    // M8 - Non-batched
    cout << "4. M8 (non-batched)... ";
    cout.flush();
    TimingResult tm8 = time_function([&]() {
        conv2d_3x3_m8(input, kernel, out_m8, H, W, use_padding != 0);
    });
    cout << tm8.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m8.bin", out_m8, static_cast<size_t>(out_h * out_w));

    // M2 - Batched
    cout << "5. M2 (batched, batch_rows=4)... ";
    cout.flush();
    TimingResult tm2_batched = time_function([&]() {
        conv2d_3x3_m2_batched(input, kernel, out_m2_batched, H, W, use_padding != 0, 4);
    });
    cout << tm2_batched.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m2_batched.bin", out_m2_batched, static_cast<size_t>(out_h * out_w));

    // M4 - Batched
    cout << "6. M4 (batched, batch_rows=4)... ";
    cout.flush();
    TimingResult tm4_batched = time_function([&]() {
        conv2d_3x3_m4_batched(input, kernel, out_m4_batched, H, W, use_padding != 0, 4);
    });
    cout << tm4_batched.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m4_batched.bin", out_m4_batched, static_cast<size_t>(out_h * out_w));

    // M8 - Batched
    cout << "7. M8 (batched, batch_rows=4)... ";
    cout.flush();
    TimingResult tm8_batched = time_function([&]() {
        conv2d_3x3_m8_batched(input, kernel, out_m8_batched, H, W, use_padding != 0, 4);
    });
    cout << tm8_batched.elapsed_ms << " ms\n";
    write_matrix_binary("./output_files/c_3x3_m8_batched.bin", out_m8_batched, static_cast<size_t>(out_h * out_w));

    // Summary table
    cout << "\n==================================================\n";
    cout << "Timing Summary:\n";
    cout << "==================================================\n";
    cout << "Implementation                    Time (ms)    Speedup vs M1\n";
    cout << "--------------------------------------------------\n";
    printf("M1 (non-batched)              %8.3f         1.00x\n", tm1.elapsed_ms);
    printf("M2 (non-batched)              %8.3f         %.2fx\n", tm2.elapsed_ms, tm1.elapsed_ms / tm2.elapsed_ms);
    printf("M4 (non-batched)              %8.3f         %.2fx\n", tm4.elapsed_ms, tm1.elapsed_ms / tm4.elapsed_ms);
    printf("M8 (non-batched)              %8.3f         %.2fx\n", tm8.elapsed_ms, tm1.elapsed_ms / tm8.elapsed_ms);
    printf("M2 (batched)                  %8.3f         %.2fx\n", tm2_batched.elapsed_ms, tm1.elapsed_ms / tm2_batched.elapsed_ms);
    printf("M4 (batched)                  %8.3f         %.2fx\n", tm4_batched.elapsed_ms, tm1.elapsed_ms / tm4_batched.elapsed_ms);
    printf("M8 (batched)                  %8.3f         %.2fx\n", tm8_batched.elapsed_ms, tm1.elapsed_ms / tm8_batched.elapsed_ms);

    // Cleanup
    delete[] input;
    delete[] kernel;
    delete[] out_m1;
    delete[] out_m2;
    delete[] out_m4;
    delete[] out_m8;
    delete[] out_m2_batched;
    delete[] out_m4_batched;
    delete[] out_m8_batched;

    return 0;
}
