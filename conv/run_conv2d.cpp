#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "./include/defs.h"

// Forward declarations for specialized 3x3 implementations (defined in src/conv2d_3x3rvv.cpp)
void conv3x3_rvv_m4(const float* input, const float* kernel, float* output, int H, int W, bool use_padding);
void conv3x3_rvv_m8(const float* input, const float* kernel, float* output, int H, int W, bool use_padding);

using namespace std;

// Helper: compute output spatial dims
static inline void compute_out_hw(int in_h, int in_w, int k_h, int k_w,
                                  int stride_h, int stride_w, int pad_h, int pad_w,
                                  int& out_h, int& out_w) {
    out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
}

int main(int argc, char* argv[]) {
    // Defaults roughly matching conv/src/conv2d_onnx.py
    int N = 1;           // batch size
    int Cin = 3;         // input channels
    int Cout = 3;        // output channels
    int H = 8;           // input height
    int W = 8;           // input width
    int kH = 3;          // kernel height
    int kW = 3;          // kernel width
    int sH = 1;          // stride height
    int sW = 1;          // stride width
    int pH = 1;          // pad height
    int pW = 1;          // pad width

    // Parse arguments (optional):
    // argv: N Cin Cout H W [kH kW sH sW pH pW]
    if (argc >= 6) {
        N   = max(1, atoi(argv[1]));
        Cin = max(1, atoi(argv[2]));
        Cout= max(1, atoi(argv[3]));
        H   = max(1, atoi(argv[4]));
        W   = max(1, atoi(argv[5]));
    }
    if (argc >= 8) {
        kH = max(1, atoi(argv[6]));
        kW = max(1, atoi(argv[7]));
    }
    if (argc >= 10) {
        sH = max(1, atoi(argv[8]));
        sW = max(1, atoi(argv[9]));
    }
    if (argc >= 12) {
        pH = max(0, atoi(argv[10]));
        pW = max(0, atoi(argv[11]));
    }

    // Derived sizes
    int in_size = N * Cin * H * W;
    int ker_size = Cout * Cin * kH * kW;
    int outH = 0, outW = 0;
    compute_out_hw(H, W, kH, kW, sH, sW, pH, pW, outH, outW);
    int out_size = N * Cout * outH * outW;

    cout << "Conv2D: Input NCHW=" << N << "x" << Cin << "x" << H << "x" << W
         << ", Kernel OIHW=" << Cout << "x" << Cin << "x" << kH << "x" << kW
         << ", Stride=(" << sH << "," << sW << ")"
         << ", Pad=(" << pH << "," << pW << ")\n";
    cout << "Output NCHW=" << N << "x" << Cout << "x" << outH << "x" << outW << "\n";

    // Allocate
    float* input = new float[in_size];
    float* kernel = new float[ker_size];
    float* out_buf = new float[out_size];
    if (!input || !kernel || !out_buf) {
        cerr << "Allocation failed" << endl;
        return 1;
    }

    // Initialize reproducibly
    srand(0);
    for (int i = 0; i < in_size; ++i) {
        input[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < ker_size; ++i) {
        kernel[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    // Persist inputs so Python can read them
    write_matrix_binary("./output_files/input.bin", input, static_cast<size_t>(in_size));
    write_matrix_binary("./output_files/kernel.bin", kernel, static_cast<size_t>(ker_size));

    // Scalar reference
    conv2d_scalar(input, kernel, out_buf,
                  N, Cin, Cout, H, W, kH, kW, sH, sW, pH, pW);
    write_matrix_binary("./output_files/c_scalar.bin", out_buf, static_cast<size_t>(out_size));

    // Vectorized variants (these will only run correctly on RVV targets)
    conv2d_e32m1(input, kernel, out_buf,
                 N, Cin, Cout, H, W, kH, kW, sH, sW, pH, pW);
    write_matrix_binary("./output_files/c_e32m1.bin", out_buf, static_cast<size_t>(out_size));

    conv2d_e32m2(input, kernel, out_buf,
                 N, Cin, Cout, H, W, kH, kW, sH, sW, pH, pW);
    write_matrix_binary("./output_files/c_e32m2.bin", out_buf, static_cast<size_t>(out_size));

    conv2d_e32m4(input, kernel, out_buf,
                 N, Cin, Cout, H, W, kH, kW, sH, sW, pH, pW);
    write_matrix_binary("./output_files/c_e32m4.bin", out_buf, static_cast<size_t>(out_size));

    conv2d_e32m8(input, kernel, out_buf,
                 N, Cin, Cout, H, W, kH, kW, sH, sW, pH, pW);
    write_matrix_binary("./output_files/c_e32m8.bin", out_buf, static_cast<size_t>(out_size));

    // If kernel is 3x3 and stride==1 we can also run the specialized 3x3 variants
    if (kH == 3 && kW == 3 && sH == 1 && sW == 1) {
        // Temporary buffer per-call
        float* temp_out = new float[out_size];
        // Output buffer for specialized variants
        float* spec_out = new float[out_size];

        // M4 specialized: accumulate per (out_ch, in_ch) using 3x3 calls
        memset(spec_out, 0, out_size * sizeof(float));
        for (int oc = 0; oc < Cout; ++oc) {
            // For each input channel, compute conv and accumulate
            for (int ic = 0; ic < Cin; ++ic) {
                const float* in_chan = input + (ic * H * W);
                const float* ker_3x3 = kernel + ((oc * Cin + ic) * 9);
                // call per-channel 3x3 m4
                conv3x3_rvv_m4(in_chan, ker_3x3, temp_out, H, W, pH > 0 || pW > 0);
                // accumulate into spec_out at output channel slice
                float* out_slice = spec_out + (oc * outH * outW);
                for (int i = 0; i < outH * outW; ++i) out_slice[i] += temp_out[i];
            }
        }
        write_matrix_binary("./output_files/c_3x3_m4.bin", spec_out, static_cast<size_t>(out_size));

        // M8 specialized
        memset(spec_out, 0, out_size * sizeof(float));
        for (int oc = 0; oc < Cout; ++oc) {
            for (int ic = 0; ic < Cin; ++ic) {
                const float* in_chan = input + (ic * H * W);
                const float* ker_3x3 = kernel + ((oc * Cin + ic) * 9);
                conv3x3_rvv_m8(in_chan, ker_3x3, temp_out, H, W, pH > 0 || pW > 0);
                float* out_slice = spec_out + (oc * outH * outW);
                for (int i = 0; i < outH * outW; ++i) out_slice[i] += temp_out[i];
            }
        }
        write_matrix_binary("./output_files/c_3x3_m8.bin", spec_out, static_cast<size_t>(out_size));

        delete[] temp_out;
        delete[] spec_out;
    }

    delete[] input;
    delete[] kernel;
    delete[] out_buf;

    return 0;
}
