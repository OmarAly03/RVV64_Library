#include "linear_reg.h"
#include <stdio.h>
#include <stdlib.h>

#define N 8

int main(){
    double x1[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    double x2[N] = {8, 7, 6, 5, 4, 3, 2, 1};
    // y = 3.5 * x_1 - 2.1 * x_2
    double y[N]  = {3.5*1 - 2.1*8,
                    3.5*2 - 2.1*7, 
                    3.5*3 - 2.1*6, 
                    3.5*4 - 2.1*5,
                    3.5*5 - 2.1*4, 
                    3.5*6 - 2.1*3, 
                    3.5*7 - 2.1*2, 
                    3.5*8 - 2.1*1};

    double x1_train[N] = {10, 11, 12, 13, 14, 15, 16, 17};
    double x2_train[N] = {17, 16, 15, 14, 13, 12, 11, 10};
    double y_train[N] = {
        3.5*10 - 2.1*17,  
        3.5*11 - 2.1*16,  
        3.5*12 - 2.1*15,  
        3.5*13 - 2.1*14,  
        3.5*14 - 2.1*13, 
        3.5*15 - 2.1*12, 
        3.5*16 - 2.1*11, 
        3.5*17 - 2.1*10  
    };    
    

    double w1 = 0.0, w2 = 0.0, b = 0.0;
    double best_w1 = 0.0, best_w2 = 0.0, best_b = 0.0;
    double lr = 0.1;
    
    double prev_mse = 1e9;
    double min_delta = 1e-6;  // Minimum improvement threshold
    int max_epochs = 100000;
    
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        train_step_2v(x1, x2, y, N, &w1, &w2, &b, lr);
    
        if (epoch % 10 == 0) {
            double mse = calculate_mse_2v(x1, x2, y, N, w1, w2, b);
            double improvement = prev_mse - mse;
    
            printf("Epoch %d | MSE: %.6f | w1: %.4f, w2: %.4f, b: %.4f\n", epoch, mse, w1, w2, b);
    
            if (improvement > min_delta) {
                prev_mse = mse;
                best_w1 = w1;
                best_w2 = w2;
                best_b = b;
            } else if (improvement < 0.0) {
                printf("MSE increased. Early stopping at epoch %d.\n", epoch);
                w1 = best_w1;
                w2 = best_w2;
                b  = best_b;
                break;
            } else {
                printf("MSE improvement too small (%.8f). Early stopping at epoch %d.\n", improvement, epoch);
                w1 = best_w1;
                w2 = best_w2;
                b  = best_b;
                break;
            }
        }
    }
                    
    printf("\nFinal Model: w1 = %.4f, w2 = %.4f, b = %.4f\n", w1, w2, b);

    // Predict some values
    for (int i = 0; i < N; i++) {
        double y_pred = w1 * x1_train[i] + w2 * x2_train[i] + b;
        printf("x1: %.1f, x2: %.1f, y: %.1f, Predicted y: %.4f\n", x1_train[i], x2_train[i], y_train[i], y_pred);
    }

}