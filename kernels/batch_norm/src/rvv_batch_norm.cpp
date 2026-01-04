#include <cstddef>
#include <riscv_vector.h>
#include "rvv_defs.hpp"
#include <cmath>

using namespace std;

/*********************************** Scalar ************************************/

void batch_norm_scalar(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        for (int i = 0; i < spatial_dim; ++i) {
            output[c * spatial_dim + i] = input[c * spatial_dim + i] * alpha + beta;
        }
    }
}

void batch_norm_tiled_scalar(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        int i = 0;
        for (; i <= spatial_dim - 4; i += 4) {
            out_ptr[i]   = in_ptr[i]   * alpha + beta;
            out_ptr[i+1] = in_ptr[i+1] * alpha + beta;
            out_ptr[i+2] = in_ptr[i+2] * alpha + beta;
            out_ptr[i+3] = in_ptr[i+3] * alpha + beta;
        }
        for (; i < spatial_dim; ++i) out_ptr[i] = in_ptr[i] * alpha + beta;
    }
}

/*********************************** Vectorized ************************************/

void batch_norm_e32m1(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        for (size_t i = 0; i < (size_t)spatial_dim; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M1>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M1>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M1>(VECTOR_MUL<float, M1>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M1>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

void batch_norm_e32m2(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        for (size_t i = 0; i < (size_t)spatial_dim; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M2>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M2>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M2>(VECTOR_MUL<float, M2>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M2>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

void batch_norm_e32m4(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        for (size_t i = 0; i < (size_t)spatial_dim; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M4>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M4>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M4>(VECTOR_MUL<float, M4>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M4>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

void batch_norm_e32m8(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        for (size_t i = 0; i < (size_t)spatial_dim; ) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M8>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M8>(VECTOR_MUL<float, M8>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M8>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

/*********************************** Tiled Vectorized ************************************/

void batch_norm_tiled_e32m1(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        size_t i = 0;
        size_t vlmax = SET_VECTOR_LENGTH_MAX<float, M1>();
        while (i + 2 * vlmax <= (size_t)spatial_dim) {
            auto v1 = VECTOR_LOAD<float, M1>(in_ptr + i, vlmax);
            auto v2 = VECTOR_LOAD<float, M1>(in_ptr + i + vlmax, vlmax);
            v1 = VECTOR_ADD<float, M1>(VECTOR_MUL<float, M1>(v1, alpha, vlmax), beta, vlmax);
            v2 = VECTOR_ADD<float, M1>(VECTOR_MUL<float, M1>(v2, alpha, vlmax), beta, vlmax);
            VECTOR_STORE<float, M1>(out_ptr + i, v1, vlmax);
            VECTOR_STORE<float, M1>(out_ptr + i + vlmax, v2, vlmax);
            i += 2 * vlmax;
        }
        while (i < (size_t)spatial_dim) {
            size_t vl = SET_VECTOR_LENGTH<float, M1>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M1>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M1>(VECTOR_MUL<float, M1>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M1>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

void batch_norm_tiled_e32m2(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        size_t i = 0;
        size_t vlmax = SET_VECTOR_LENGTH_MAX<float, M2>();
        while (i + 2 * vlmax <= (size_t)spatial_dim) {
            auto v1 = VECTOR_LOAD<float, M2>(in_ptr + i, vlmax);
            auto v2 = VECTOR_LOAD<float, M2>(in_ptr + i + vlmax, vlmax);
            v1 = VECTOR_ADD<float, M2>(VECTOR_MUL<float, M2>(v1, alpha, vlmax), beta, vlmax);
            v2 = VECTOR_ADD<float, M2>(VECTOR_MUL<float, M2>(v2, alpha, vlmax), beta, vlmax);
            VECTOR_STORE<float, M2>(out_ptr + i, v1, vlmax);
            VECTOR_STORE<float, M2>(out_ptr + i + vlmax, v2, vlmax);
            i += 2 * vlmax;
        }
        while (i < (size_t)spatial_dim) {
            size_t vl = SET_VECTOR_LENGTH<float, M2>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M2>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M2>(VECTOR_MUL<float, M2>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M2>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

void batch_norm_tiled_e32m4(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        size_t i = 0;
        size_t vlmax = SET_VECTOR_LENGTH_MAX<float, M4>();
        while (i + 2 * vlmax <= (size_t)spatial_dim) {
            auto v1 = VECTOR_LOAD<float, M4>(in_ptr + i, vlmax);
            auto v2 = VECTOR_LOAD<float, M4>(in_ptr + i + vlmax, vlmax);
            v1 = VECTOR_ADD<float, M4>(VECTOR_MUL<float, M4>(v1, alpha, vlmax), beta, vlmax);
            v2 = VECTOR_ADD<float, M4>(VECTOR_MUL<float, M4>(v2, alpha, vlmax), beta, vlmax);
            VECTOR_STORE<float, M4>(out_ptr + i, v1, vlmax);
            VECTOR_STORE<float, M4>(out_ptr + i + vlmax, v2, vlmax);
            i += 2 * vlmax;
        }
        while (i < (size_t)spatial_dim) {
            size_t vl = SET_VECTOR_LENGTH<float, M4>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M4>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M4>(VECTOR_MUL<float, M4>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M4>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}

void batch_norm_tiled_e32m8(const float* input, float* output, const float* scale, const float* bias, const float* mean, const float* variance, int channels, int height, int width, float epsilon) {
    int spatial_dim = height * width;
    for (int c = 0; c < channels; ++c) {
        float alpha = scale[c] / std::sqrt(variance[c] + epsilon);
        float beta = bias[c] - mean[c] * alpha;
        const float* in_ptr = input + c * spatial_dim;
        float* out_ptr = output + c * spatial_dim;
        size_t i = 0;
        size_t vlmax = SET_VECTOR_LENGTH_MAX<float, M8>();
        while (i + 2 * vlmax <= (size_t)spatial_dim) {
            auto v1 = VECTOR_LOAD<float, M8>(in_ptr + i, vlmax);
            auto v2 = VECTOR_LOAD<float, M8>(in_ptr + i + vlmax, vlmax);
            v1 = VECTOR_ADD<float, M8>(VECTOR_MUL<float, M8>(v1, alpha, vlmax), beta, vlmax);
            v2 = VECTOR_ADD<float, M8>(VECTOR_MUL<float, M8>(v2, alpha, vlmax), beta, vlmax);
            VECTOR_STORE<float, M8>(out_ptr + i, v1, vlmax);
            VECTOR_STORE<float, M8>(out_ptr + i + vlmax, v2, vlmax);
            i += 2 * vlmax;
        }
        while (i < (size_t)spatial_dim) {
            size_t vl = SET_VECTOR_LENGTH<float, M8>(spatial_dim - i);
            auto v = VECTOR_LOAD<float, M8>(in_ptr + i, vl);
            v = VECTOR_ADD<float, M8>(VECTOR_MUL<float, M8>(v, alpha, vl), beta, vl);
            VECTOR_STORE<float, M8>(out_ptr + i, v, vl);
            i += vl;
        }
    }
}