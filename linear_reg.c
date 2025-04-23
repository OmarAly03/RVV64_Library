#include <stdio.h>
#include <stdlib.h>

#define N 8

void train_step_1v(double *x, double *y, size_t n, double *w, double *b, double lr) {
    double dw = 0.0, db = 0.0;
    double zero = 0.0;

    // Vector length setup
    size_t vl;
    asm volatile ("vsetvli %0, %1, e64, m1" : "=r"(vl) : "r"(n));

    // Use vector registers v8–v15 (caller-saved)
    asm volatile (
        // Load x → v8
        "vle64.v v8, (%1)\n\t"
        // Load y → v9
        "vle64.v v9, (%2)\n\t"

        // y_pred = x * w → v10
        "vfmul.vf v10, v8, %3\n\t"
        // y_pred += b
        "vfadd.vf v10, v10, %4\n\t"

        // error = y_pred - y → v11
        "vfsub.vv v11, v10, v9\n\t"

        // db = sum(error) → v12
        "vfredsum.vs v12, v11, v0\n\t"
        "vfmv.f.s ft0, v12\n\t"     // Move scalar from vector to float register
        "fsd ft0, 0(%5)\n\t"        // Store from float register

        // dw = sum(error * x)
        "vfmul.vv v13, v11, v8\n\t"
        "vfredsum.vs v14, v13, v0\n\t"
        "vfmv.f.s ft0, v14\n\t"     // Move scalar from vector to float register
        "fsd ft0, 0(%6)\n\t"        // Store from float register

        : // outputs
        : "r"(vl),     // %0
          "r"(x),      // %1
          "r"(y),      // %2
          "f"(*w),     // %3
          "f"(*b),     // %4
          "r"(&db),    // %5
          "r"(&dw)     // %6
        : "ft0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "memory"
    );

    // Gradient descent update
    *w -= lr * dw / (double)N;
    *b -= lr * db / (double)N;
}

void train_step_2v(double *x1, double *x2, double *y, size_t n, double *w1, double *w2, double *b, double lr) {
    double dw1 = 0.0, dw2 = 0.0, db = 0.0;

    size_t vl;
    asm volatile("vsetvli %0, %1, e64, m1" : "=r"(vl) : "r"(n));

    asm volatile(
        // Load x1, x2, y → v8, v9, v10
        "vle64.v v8, (%0)\n\t"
        "vle64.v v9, (%1)\n\t"
        "vle64.v v10, (%2)\n\t"

        // y_pred = x1 * w1 + x2 * w2 → v11
        "vfmul.vf v11, v8, %3\n\t"
        "vfmul.vf v12, v9, %4\n\t"
        "vfadd.vv v11, v11, v12\n\t"
        "vfadd.vf v11, v11, %5\n\t"

        // error = y_pred - y → v11
        "vfsub.vv v11, v11, v10\n\t"

        // db = sum(error) → v12
        "vfredsum.vs v12, v11, v0\n\t"
        "vfmv.f.s ft0, v12\n\t"
        "fsd ft0, 0(%6)\n\t"

        // dw1 = sum(error * x1) → v13
        "vfmul.vv v13, v11, v8\n\t"
        "vfredsum.vs v14, v13, v0\n\t"
        "vfmv.f.s ft0, v14\n\t"
        "fsd ft0, 0(%7)\n\t"

        // dw2 = sum(error * x2) → v15
        "vfmul.vv v13, v11, v9\n\t"
        "vfredsum.vs v14, v13, v0\n\t"
        "vfmv.f.s ft0, v14\n\t"
        "fsd ft0, 0(%8)\n\t"

        :
        : "r"(x1), "r"(x2), "r"(y), "f"(*w1), "f"(*w2), "f"(*b),
          "r"(&db), "r"(&dw1), "r"(&dw2)
        : "ft0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v0"
    );

    *w1 -= lr * dw1 / n;
    *w2 -= lr * dw2 / n;
    *b  -= lr * db  / n;
}

void train_step_3v(double *x1, double *x2, double *x3, double *y, size_t n, double *w1, double *w2, double *w3, double *b, double lr) {
    double dw1 = 0.0, dw2 = 0.0, dw3 = 0.0, db = 0.0;

    size_t vl;
    asm volatile("vsetvli %0, %1, e64, m1" : "=r"(vl) : "r"(n));

    asm volatile(
        // Load x1, x2, x3, y → v8, v9, v10, v11
        "vle64.v v8, (%0)\n\t"
        "vle64.v v9, (%1)\n\t"
        "vle64.v v10, (%2)\n\t"
        "vle64.v v11, (%3)\n\t"

        // y_pred = x1 * w1 + x2 * w2 + x3 * w3 + b → v12
        "vfmul.vf v12, v8, %4\n\t"      // x1 * w1
        "vfmul.vf v13, v9, %5\n\t"      // x2 * w2
        "vfmul.vf v14, v10, %6\n\t"     // x3 * w3
        "vfadd.vv v12, v12, v13\n\t"    // (x1*w1 + x2*w2)
        "vfadd.vv v12, v12, v14\n\t"    // (x1*w1 + x2*w2 + x3*w3)
        "vfadd.vf v12, v12, %7\n\t"     // (x1*w1 + x2*w2 + x3*w3 + b)

        // error = y_pred - y → v12
        "vfsub.vv v12, v12, v11\n\t"

        // db = sum(error) → v15
        "vfredsum.vs v15, v12, v0\n\t"
        "vfmv.f.s ft0, v15\n\t"
        "fsd ft0, 0(%8)\n\t"

        // dw1 = sum(error * x1)
        "vfmul.vv v13, v12, v8\n\t"
        "vfredsum.vs v15, v13, v0\n\t"
        "vfmv.f.s ft0, v15\n\t"
        "fsd ft0, 0(%9)\n\t"

        // dw2 = sum(error * x2)
        "vfmul.vv v13, v12, v9\n\t"
        "vfredsum.vs v15, v13, v0\n\t"
        "vfmv.f.s ft0, v15\n\t"
        "fsd ft0, 0(%10)\n\t"

        // dw3 = sum(error * x3)
        "vfmul.vv v13, v12, v10\n\t"
        "vfredsum.vs v15, v13, v0\n\t"
        "vfmv.f.s ft0, v15\n\t"
        "fsd ft0, 0(%11)\n\t"

        :
        : "r"(x1), "r"(x2), "r"(x3), "r"(y),
        "f"(*w1), "f"(*w2), "f"(*w3), "f"(*b),
        "r"(&db), "r"(&dw1), "r"(&dw2), "r"(&dw3)
        : "ft0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v0", "memory"
    );

    *w1 -= lr * dw1 / n;
    *w2 -= lr * dw2 / n;
    *w3 -= lr * dw3 / n;
    *b  -= lr * db  / n;
}

double calculate_mse_3v(double *x1, double *x2, double *x3, double *y, size_t n, double w1, double w2, double w3, double b) {
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double y_pred = w1 * x1[i] + w2 * x2[i] + w3 * x3[i] + b;
        double diff = y_pred - y[i];
        mse += diff * diff;
    }
    return mse / n;
}

double calculate_mse_1v(double *x, double *y, size_t n, double w, double b) {
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double y_pred = w * x[i] + b;
        double diff = y_pred - y[i];
        mse += diff * diff;  
    }
    return mse / n;  // Return the average error
}

double calculate_mse_2v(double *x1, double *x2, double *y, size_t n, double w1, double w2, double b) {
    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double y_pred = w1 * x1[i] + w2 * x2[i] + b;
        double diff = y_pred - y[i];
        mse += diff * diff;
    }
    return mse / n;
}

