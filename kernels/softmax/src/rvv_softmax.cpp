#include <cmath>
#include <cstddef>
#include <cfloat>

extern "C"{
void softmax(
    const float* input,
    float* output,
    size_t n
) {
    // Find max value (numerical stability)
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; ++i) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = expf(input[i] - max_val);
        output[i] = e;
        sum += e;
    }

    // Normalize
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

}