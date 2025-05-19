#include <stdio.h>
#include <stdlib.h>
#include "./lib/defs.h"

#define N 8

int main() {
    // Test data with known relationship: y = 2.5*x1 - 1.8*x2 + 3.2*x3 + 0.7
    double x1[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double x2[N] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double x3[N] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
    
    // Calculate ground truth y values
    double y[N];
    for (int i = 0; i < N; i++) {
        y[i] = 2.5 * x1[i] - 1.8 * x2[i] + 3.2 * x3[i] + 0.7;
    }
    
    // Initialize model parameters
    double w1 = 0.0, w2 = 0.0, w3 = 0.0, b = 0.0;
    double best_w1 = 0.0, best_w2 = 0.0, best_w3 = 0.0, best_b = 0.0;
    double best_mse = 1e9;
    
    // Training hyperparameters
    double learning_rate = 0.008;
    int max_epochs = 10000;
    double min_delta = 1e-6;
    
    printf("Training 3-variable linear regression model...\n");
    printf("Ground truth: w1=2.5, w2=-1.8, w3=3.2, b=0.7\n\n");
    
    // Training loop
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        train_step_3v(x1, x2, x3, y, N, &w1, &w2, &w3, &b, learning_rate);
        
        double mse = calculate_mse_3v(x1, x2, x3, y, N, w1, w2, w3, b);
        
        // Save best model
        if (mse < best_mse) {
            best_mse = mse;
            best_w1 = w1;
            best_w2 = w2;
            best_w3 = w3;
            best_b = b;
        }
        
        // Print progress every 1000 epochs
        if (epoch % 1000 == 0) {
            printf("Epoch %d: w1=%.4f, w2=%.4f, w3=%.4f, b=%.4f, MSE=%.10f\n", 
                   epoch, w1, w2, w3, b, mse);
        }
        
        // Early stopping if MSE is very low
        if (mse < min_delta) {
            printf("Converged at epoch %d!\n", epoch);
            break;
        }
    }
    
    printf("\nFinal model parameters:\n");
    printf("w1=%.4f, w2=%.4f, w3=%.4f, b=%.4f, MSE=%.10f\n", 
           best_w1, best_w2, best_w3, best_b, best_mse);
    
    // Test predictions
    printf("\nPredictions vs. Actual Values:\n");
    printf("%-6s %-6s %-6s %-12s %-12s\n", "x1", "x2", "x3", "Predicted", "Actual");
    
    for (int i = 0; i < N; i++) {
        double pred = best_w1 * x1[i] + best_w2 * x2[i] + best_w3 * x3[i] + best_b;
        printf("%-6.1f %-6.1f %-6.1f %-12.4f %-12.4f\n", 
               x1[i], x2[i], x3[i], pred, y[i]);
    }
    
    // Test with new data
    printf("\nTesting with new data:\n");
    double test_x1[3] = {9.0, 10.0, 11.0};
    double test_x2[3] = {0.5, 1.5, 2.5};
    double test_x3[3] = {4.5, 5.0, 5.5};
    
    for (int i = 0; i < 3; i++) {
        double actual = 2.5 * test_x1[i] - 1.8 * test_x2[i] + 3.2 * test_x3[i] + 0.7;
        double pred = best_w1 * test_x1[i] + best_w2 * test_x2[i] + best_w3 * test_x3[i] + best_b;
        printf("Input: (%.1f, %.1f, %.1f), Predicted: %.4f, Actual: %.4f\n", 
               test_x1[i], test_x2[i], test_x3[i], pred, actual);
    }
    
    return 0;
}
