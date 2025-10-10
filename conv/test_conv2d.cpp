#include <riscv_vector.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include "defs.h"


// Helper function to initialize data with random values
void initialize_data(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// Helper function to compare outputs
bool compare_outputs(const float* output1, const float* output2, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(output1[i] - output2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << output1[i] << " vs " << output2[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Helper function to print timing results
void print_timing(const std::string& name, double time_ms) {
    std::cout << std::setw(15) << name << ": " << std::setw(8) << std::fixed << std::setprecision(3) 
              << time_ms << " ms" << std::endl;
}

int main() {
    std::cout << "=== RVV Conv2D Performance Test ===" << std::endl;
    
    // Test parameters
    const int batch_size = 1;
    const int in_channels = 3;
    const int out_channels = 16;
    const int input_h = 32;
    const int input_w = 32;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int stride_h = 1;
    const int stride_w = 1;
    const int pad_h = 1;
    const int pad_w = 1;
    
    // Calculate sizes
    const int input_size = batch_size * in_channels * input_h * input_w;
    const int kernel_size = out_channels * in_channels * kernel_h * kernel_w;
    const int out_height = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    const int output_size = batch_size * out_channels * out_height * out_width;
    
    std::cout << "Input size: " << batch_size << "x" << in_channels << "x" << input_h << "x" << input_w << std::endl;
    std::cout << "Kernel size: " << out_channels << "x" << in_channels << "x" << kernel_h << "x" << kernel_w << std::endl;
    std::cout << "Output size: " << batch_size << "x" << out_channels << "x" << out_height << "x" << out_width << std::endl;
    std::cout << "Stride: " << stride_h << "x" << stride_w << ", Padding: " << pad_h << "x" << pad_w << std::endl;
    std::cout << std::endl;
    
    // Allocate memory
    float* input = new float[input_size];
    float* kernel = new float[kernel_size];
    float* output_scalar = new float[output_size];
    float* output_e32m1 = new float[output_size];
    float* output_e32m2 = new float[output_size];
    float* output_e32m4 = new float[output_size];
    float* output_e32m8 = new float[output_size];
    
    // Initialize input and kernel with random data
    initialize_data(input, input_size);
    initialize_data(kernel, kernel_size);
    
    std::cout << "Running performance tests..." << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    // Test scalar implementation
    auto start = std::chrono::high_resolution_clock::now();
    conv2d_scalar(input, kernel, output_scalar, batch_size, in_channels, out_channels,
                  input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    auto end = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
    print_timing("Scalar", scalar_time);
    
    // Test RVV e32m1
    start = std::chrono::high_resolution_clock::now();
    conv2d_e32m1(input, kernel, output_e32m1, batch_size, in_channels, out_channels,
                 input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    end = std::chrono::high_resolution_clock::now();
    double e32m1_time = std::chrono::duration<double, std::milli>(end - start).count();
    print_timing("RVV e32m1", e32m1_time);
    
    // Test RVV e32m2
    start = std::chrono::high_resolution_clock::now();
    conv2d_e32m2(input, kernel, output_e32m2, batch_size, in_channels, out_channels,
                 input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    end = std::chrono::high_resolution_clock::now();
    double e32m2_time = std::chrono::duration<double, std::milli>(end - start).count();
    print_timing("RVV e32m2", e32m2_time);
    
    // Test RVV e32m4
    start = std::chrono::high_resolution_clock::now();
    conv2d_e32m4(input, kernel, output_e32m4, batch_size, in_channels, out_channels,
                 input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    end = std::chrono::high_resolution_clock::now();
    double e32m4_time = std::chrono::duration<double, std::milli>(end - start).count();
    print_timing("RVV e32m4", e32m4_time);
    
    // Test RVV e32m8
    start = std::chrono::high_resolution_clock::now();
    conv2d_e32m8(input, kernel, output_e32m8, batch_size, in_channels, out_channels,
                 input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    end = std::chrono::high_resolution_clock::now();
    double e32m8_time = std::chrono::duration<double, std::milli>(end - start).count();
    print_timing("RVV e32m8", e32m8_time);
    
    std::cout << std::string(40, '-') << std::endl;
    
    // Performance comparison
    std::cout << "Speedup vs Scalar:" << std::endl;
    std::cout << "  e32m1: " << std::setprecision(2) << (scalar_time / e32m1_time) << "x" << std::endl;
    std::cout << "  e32m2: " << std::setprecision(2) << (scalar_time / e32m2_time) << "x" << std::endl;
    std::cout << "  e32m4: " << std::setprecision(2) << (scalar_time / e32m4_time) << "x" << std::endl;
    std::cout << "  e32m8: " << std::setprecision(2) << (scalar_time / e32m8_time) << "x" << std::endl;
    std::cout << std::endl;
    
    // Correctness verification
    std::cout << "Correctness verification:" << std::endl;
    bool e32m1_correct = compare_outputs(output_scalar, output_e32m1, output_size);
    bool e32m2_correct = compare_outputs(output_scalar, output_e32m2, output_size);
    bool e32m4_correct = compare_outputs(output_scalar, output_e32m4, output_size);
    bool e32m8_correct = compare_outputs(output_scalar, output_e32m8, output_size);
    
    std::cout << "  e32m1: " << (e32m1_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "  e32m2: " << (e32m2_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "  e32m4: " << (e32m4_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "  e32m8: " << (e32m8_correct ? "PASS" : "FAIL") << std::endl;
    
    // Print some sample outputs for manual verification
    std::cout << std::endl << "Sample outputs (first 10 values):" << std::endl;
    std::cout << "Scalar: ";
    for (int i = 0; i < 10 && i < output_size; ++i) {
        std::cout << std::setprecision(4) << output_scalar[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "e32m1:  ";
    for (int i = 0; i < 10 && i < output_size; ++i) {
        std::cout << std::setprecision(4) << output_e32m1[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up
    delete[] input;
    delete[] kernel;
    delete[] output_scalar;
    delete[] output_e32m1;
    delete[] output_e32m2;
    delete[] output_e32m4;
    delete[] output_e32m8;
    
    std::cout << std::endl << "Test completed successfully!" << std::endl;
    
    return 0;
}