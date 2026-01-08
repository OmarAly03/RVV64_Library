#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <riscv_vector.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

// Box format constants
#define CORNER_FORMAT 0  // [y1, x1, y2, x2]
#define CENTER_FORMAT 1  // [x_center, y_center, width, height]

struct SelectedIndex {
    int64_t batch_index;
    int64_t class_index;
    int64_t box_index;
};

std::vector<SelectedIndex> nms_e32m8(
    const float* boxes, const float* scores,
    size_t num_batches, size_t num_classes, size_t spatial_dimension,
    int64_t max_output_boxes_per_class, float iou_threshold, float score_threshold,
    int center_point_box
);

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

// Fixed GEMM block sizes (previously configurable)
#define GEMM_BLOCK_M 8
#define GEMM_BLOCK_N 64
#define GEMM_BLOCK_K 32

/****************** Specific Image Pre-processing Kernel ******************/
void preprocess_image(
    float* data, const float* scale, const float* bias,
    int channels, int height, int width);

/************************************ CONV ************************************/
void conv2d(
    const float* input, float* output, const float* weights,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left);

void gemm_blocked_e32m8(const float* A, const float* B, float* C,
                        int M, int N, int K,
                        int BM, int BN, int BK);

void im2col_e32m8(const float* data_im, float* data_col,
                  int channels, int height, int width,
                  int kernel_h, int kernel_w,
                  int pad_h, int pad_w,
                  int stride_h, int stride_w);

void conv2d_im2col_gemm_m8(
    const float* input, const float* kernel, const float* bias,
    float* output, float* col_buf, float* gemm_buf,
    int in_channels, int input_h, int input_w, 
    int out_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int has_bias);


/************************************ Bias Add ************************************/
void bias_add_e32m8(const float* input, const float* bias, float* output,
	size_t channels, size_t channel_size);

/************************************ Batch Norm ************************************/
void batch_norm_e32m8(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon);

/************************************ Maxpool ************************************/
void maxpool_e32m8(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

void maxpool_e32m8_fixed(const float* input, float* output,
	int batch, int channels,
	int in_h, int in_w,
	int out_h, int out_w,
	int k_h, int k_w,
	int stride_h, int stride_w,
	int pad_h, int pad_w);

/************************************ LeakyRelu ************************************/
void leaky_relu_e32m8(const float* src, float* dest, size_t n, float alpha);

#endif // KERNELS_HPP