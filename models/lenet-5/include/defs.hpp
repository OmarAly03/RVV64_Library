#ifndef DEFS_HPP
#define DEFS_HPP

#include <vector>
#include <string>

	// --- 1. Define Model Architecture Constants ---
	const int BATCH_SIZE = 1;

	// Input
	const int IN_C = 1;
	const int IN_H = 32;
	const int IN_W = 32;
	const int IN_SIZE = BATCH_SIZE * IN_C * IN_H * IN_W;

	// C1
	const int C1_IN_C = 1;
	const int C1_OUT_C = 6;
	const int C1_K = 5;
	const int C1_OUT_H = 28;
	const int C1_OUT_W = 28;
	const int C1_OUT_SIZE = BATCH_SIZE * C1_OUT_C * C1_OUT_H * C1_OUT_W;

	// Pool1
	const int POOL1_K = 2;
	const int POOL1_S = 2;
	const int POOL1_OUT_H = 14;
	const int POOL1_OUT_W = 14;
	const int POOL1_OUT_SIZE = BATCH_SIZE * C1_OUT_C * POOL1_OUT_H * POOL1_OUT_W;
	
	// C2 (Both c2_1 and c2_2)
	const int C2_IN_C = 6;
	const int C2_OUT_C = 16;
	const int C2_K = 5;
	const int C2_OUT_H = 10;
	const int C2_OUT_W = 10;
	const int C2_OUT_SIZE = BATCH_SIZE * C2_OUT_C * C2_OUT_H * C2_OUT_W;

	// Pool2 (Both pool2_1 and pool2_2)
	const int POOL2_K = 2;
	const int POOL2_S = 2;
	const int POOL2_OUT_H = 5;
	const int POOL2_OUT_W = 5;
	const int POOL2_OUT_SIZE = BATCH_SIZE * C2_OUT_C * POOL2_OUT_H * POOL2_OUT_W;
	
	// Add node output size is same as Pool2 output
	const int ADD_OUT_SIZE = POOL2_OUT_SIZE;

	// C3 (This is the Conv-based FC layer)
	const int C3_IN_C = 16; // Input from Add node
	const int C3_OUT_C = 120;
	const int C3_K = 5; // Kernel size is 5x5
	const int C3_OUT_H = 1; // (5 - 5) / 1 + 1 = 1
	const int C3_OUT_W = 1;
	const int C3_OUT_SIZE = BATCH_SIZE * C3_OUT_C * C3_OUT_H * C3_OUT_W; // 120

	// F4
	const int F4_IN = 120;
	const int F4_OUT = 84;
	
	// F5
	const int F5_IN = 84;
	const int F5_OUT = 10;

std::vector<float> load_weights(const std::string& filename);

void load_preprocessed_image(std::vector<float>& img_buffer, const std::string& filename);

inline int conv_output_size(int input_size, int kernel_size, int stride, int pad);
void conv2d_scalar(
    const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w);

void maxpool_scalar_tile(
	const float* X, float* Y, int64_t* I,
	size_t N, size_t C, size_t H, size_t W, size_t K, size_t S, bool ceil_mode,
	size_t OH, size_t OW,
	size_t tile_oh_start, size_t tile_ow_start,
	size_t tile_oh_end, size_t tile_ow_end);

void relu_scalar(float* input, float* output, size_t size);

void dense_scalar(const float* input, const float* weights, const float* bias,
	float* output, size_t in_features, size_t out_features);

void bias_add_scalar(const float* input, const float* bias, float* output,
	size_t batch_size, size_t channels,
	size_t height, size_t width);

void tensor_add_scalar(const float* input_a, const float* input_b, float* output,
	size_t size);

void softmax_scalar(float* input, float* output, size_t size);

#endif // DEFS_HPP