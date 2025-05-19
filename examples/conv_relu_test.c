#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../lib/defs.h"

// Declarations from the implementation
void conv_layer(float *input, float *weights, float *biases,
                int in_h, int in_w, int in_c,
                int num_filters, int filter_size,
                int stride, int padding,
                float *output);

void relu_activation(float *input, int h, int w, int c, float *output);

// Helper for comparing floats
int float_eq(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// Helper to print matrices
void print_matrix(const char *name, float *data, int height, int width) {
    printf("%s (%dx%d):\n", name, height, width);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%6.2f ", data[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int test_conv_simple() {
    printf("\n===== CONVOLUTION TEST =====\n");
    
    // 1-channel 3x3 input, 1 filter of size 2, stride=1, padding=0
    float input[3*3*1] = { 1, 2, 3,
                            4, 5, 6,
                            7, 8, 9 };
    float weights[1*2*2*1] = { 1, 0,
                               0, 1 };
    float biases[1] = { 0 };
    float output[2*2*1];
    float expected[4] = { 1+5, 2+6, 4+8, 5+9 };

    // Print input data
    print_matrix("Input Matrix", input, 3, 3);
    
    // Print filter weights
    print_matrix("Filter Weights", weights, 2, 2);
    
    // Print bias
    printf("Bias: %.2f\n\n", biases[0]);
    
    // Perform convolution
    printf("Performing convolution (3x3 input, 2x2 filter, stride=1, padding=0)...\n");
    conv_layer(input, weights, biases, 3, 3, 1, 1, 2, 1, 0, output);
    
    // Print actual output
    print_matrix("Actual Output", output, 2, 2);
    
    // Print expected output
    print_matrix("Expected Output", expected, 2, 2);
    
    // Verify results
    int passed = 1;
    for (int i = 0; i < 4; i++) {
        if (!float_eq(output[i], expected[i], 1e-6f)) {
            printf("❌ Mismatch at index %d: got %.2f, expected %.2f\n", 
                   i, output[i], expected[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("✅ Convolution test passed!\n");
    }
    
    return passed;
}

int test_relu_simple() {
    printf("\n===== RELU TEST =====\n");
    
    float input[6] = { -1.0f, 0.0f, 2.5f, -3.2f, 0.1f, -0.0f };
    float output[6];
    float expected[6] = { 0.0f, 0.0f, 2.5f, 0.0f, 0.1f, 0.0f };

    // Print input data
    printf("Input values: ");
    for (int i = 0; i < 6; i++) {
        printf("%.2f ", input[i]);
    }
    printf("\n\n");
    
    // Perform ReLU activation
    printf("Applying ReLU activation...\n");
    relu_activation(input, 1, 6, 1, output);
    
    // Print output and expected values side by side
    printf("Index | Input  | Output | Expected\n");
    printf("------|--------|--------|--------\n");
    for (int i = 0; i < 6; i++) {
        printf("  %d   | %6.2f | %6.2f | %6.2f %s\n", 
               i, input[i], output[i], expected[i], 
               float_eq(output[i], expected[i], 1e-6f) ? "✓" : "❌");
    }
    printf("\n");
    
    // Verify results
    int passed = 1;
    for (int i = 0; i < 6; i++) {
        if (!float_eq(output[i], expected[i], 1e-6f)) {
            printf("ReLU mismatch at index %d: got %.2f, expected %.2f\n", 
                   i, output[i], expected[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("ReLU test passed!\n");
    }
    
    return passed;
}

int main() {
    printf("==================================\n");
    printf("CONVOLUTION AND RELU DEMONSTRATION\n");
    printf("==================================\n");
    
    int pass_conv = test_conv_simple();
    int pass_relu = test_relu_simple();

    printf("\n==================================\n");
    printf("SUMMARY:\n");
    printf("Convolution Test: %s\n", pass_conv ? "PASSED " : "FAILED ");
    printf("ReLU Test: %s\n", pass_relu ? "PASSED " : "FAILED ");
    
    if (pass_conv && pass_relu) {
        printf("\nAll tests passed! \n");
        return 0;
    } else {
        printf("\nSome tests failed. \n");
        return 1;
    }
}