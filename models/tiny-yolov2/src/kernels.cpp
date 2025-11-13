// kernels.cpp
#include "kernels.hpp"
#include <cfloat> // for FLT_MAX

// (x * scale) + bias
void preprocess_image(
    float* data, const float* scale,
    const float* bias,
    int channels, int height, int width)
{
    const size_t spatial_dim = height * width;
    const float s = scale[0];
    for (int c = 0; c < channels; ++c) {
        float b = bias[c];
        for (size_t i = 0; i < spatial_dim; ++i) {
            size_t idx = c * spatial_dim + i;
            data[idx] = (data[idx] * s) + b;
        }
    }
}

// NOTE: This is a *naive* implementation.
// For real performance, you would use im2col + GEMM (e.g., with Eigen/BLAS)
void conv2d(
    const float* input, float* output,
    const float* weights,
    int in_channels, int in_height, int in_width,
    int out_channels, int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left)
{
    const int in_spatial = in_height * in_width;
    const int out_spatial = out_height * out_width;
    const int kernel_spatial = kernel_size * kernel_size;

    // Clear output buffer
    std::fill(output, output + (out_channels * out_spatial), 0.0f);

    for (int c_out = 0; c_out < out_channels; ++c_out) {
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int h_out = 0; h_out < out_height; ++h_out) {
                for (int w_out = 0; w_out < out_width; ++w_out) {
                    
                    const int h_in_start = h_out * stride - pad_top;
                    const int w_in_start = w_out * stride - pad_left;
                    
                    float sum = 0.0f;
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            
                            const int h_in = h_in_start + kh;
                            const int w_in = w_in_start + kw;

                            if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                                int in_idx = (c_in * in_spatial) + (h_in * in_width) + w_in;
                                int w_idx = (c_out * in_channels * kernel_spatial) + (c_in * kernel_spatial) + (kh * kernel_size) + kw;
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                    output[(c_out * out_spatial) + (h_out * out_width) + w_out] += sum;
                }
            }
        }
    }
}

void batch_normalization(
    float* data, const float* scale,
    const float* bias, const float* mean,
    const float* variance,
    int channels, int height, int width, float epsilon)
{
    const int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float s = scale[c];
        float b = bias[c];
        float m = mean[c];
        float v = variance[c];
        
        float std_dev_inv = 1.0f / std::sqrt(v + epsilon);
        
        for (int i = 0; i < spatial_dim; ++i) {
            int idx = c * spatial_dim + i;
            data[idx] = s * ((data[idx] - m) * std_dev_inv) + b;
        }
    }
}

void leaky_relu(float* data, size_t num_elements, float alpha) {
    for (size_t i = 0; i < num_elements; ++i) {
        if (data[i] < 0) {
            data[i] = data[i] * alpha;
        }
    }
}

void max_pool_2d(
    const float* input, float* output,
    int in_channels, int in_height, int in_width,
    int out_height, int out_width,
    int kernel_size, int stride, int pad_top, int pad_left)
{
    const int in_spatial = in_height * in_width;
    const int out_spatial = out_height * out_width;

    for (int c = 0; c < in_channels; ++c) {
        for (int h_out = 0; h_out < out_height; ++h_out) {
            for (int w_out = 0; w_out < out_width; ++w_out) {
                
                const int h_in_start = h_out * stride - pad_top;
                const int w_in_start = w_out * stride - pad_left;
                
                float max_val = -FLT_MAX;
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int h_in = h_in_start + kh;
                        int w_in = w_in_start + kw;
                        
                        if (h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width) {
                            int in_idx = (c * in_spatial) + (h_in * in_width) + w_in;
                            max_val = std::max(max_val, input[in_idx]);
                        }
                    }
                }
                int out_idx = (c * out_spatial) + (h_out * out_width) + w_out;
                output[out_idx] = max_val;
            }
        }
    }
}

void add_bias(
    float* data, const float* biases,
    int channels, int height, int width)
{
    const size_t spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float b = biases[c];
        for (size_t i = 0; i < spatial_dim; ++i) {
            data[c * spatial_dim + i] += b;
        }
    }
}