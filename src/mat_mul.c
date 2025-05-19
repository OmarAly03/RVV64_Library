#include <stdlib.h>
#include <stdint.h>
#include <string.h>

void vector_mul_scalar(int32_t *A, int32_t *B, int32_t *C, int rows_A, int cols_A, int cols_B) {
    /* Normal Scalar Matrix-Matrix Multiplication */
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int32_t sum = 0;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j]; // A[i][k] * B[k][j]
            }
            C[i * cols_B + j] = sum;
        }
    }
}

void transpose_rvv(int32_t *B, int32_t *BT, int rows_B, int cols_B) {
    /* Matrix Transposition before Matrix-Matrix Multiplication RVV*/
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; ) {
            size_t vl;
            asm volatile (
                "vsetvli %0, %1, e32, m1, ta, ma"
                : "=r"(vl)
                : "r"(cols_B - j)
            );
            asm volatile (
                "vle32.v v0, (%0)"
                :: "r"(&B[i * cols_B + j])
                : "v0"
            );
            asm volatile (
                "vsse32.v v0, (%0), %1"
                :: "r"(&BT[j * rows_B + i]), "r"(rows_B * sizeof(int32_t))
                : "v0"
            );
            j += vl;
        }
    }
}

void vector_mul_rvv(int32_t *A, int32_t *BT, int32_t *C, int rows_A, int cols_A, int cols_B) {
    /* Matrix-Matrix Multiplication RVV */
    memset(C, 0, rows_A * cols_B * sizeof(int32_t));
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int32_t sum = 0;
            int32_t remaining = cols_A;
            int32_t *a_ptr = &A[i * cols_A];
            int32_t *bt_ptr = &BT[j * cols_A]; // B^T[j][k] = B[k][j]
            size_t vlmax;
            asm volatile (
                "vsetvli %0, %1, e32, m1, ta, ma"
                : "=r"(vlmax)
                : "r"(cols_A)
            );
            while (remaining >= vlmax) {
                int32_t temp_sum;
                asm volatile (
                    "vle32.v v0, (%1);" // A[i][k:k+vlmax]
                    "vle32.v v4, (%2);" // B^T[j][k:k+vlmax] = B[k][j]
                    "vmul.vv v8, v0, v4;"
                    "vmv.v.i v9, 0;"
                    "vredsum.vs v8, v8, v9;"
                    "vmv.x.s %0, v8"
                    : "=r"(temp_sum)
                    : "r"(a_ptr), "r"(bt_ptr)
                    : "v0", "v4", "v8", "v9"
                );
                sum += temp_sum;
                a_ptr += vlmax;
                bt_ptr += vlmax;
                remaining -= vlmax;
            }
            if (remaining > 0) {
                size_t vl;
                int32_t temp_sum;
                asm volatile (
                    "vsetvli %0, %2, e32, m1, ta, ma;"
                    "vle32.v v0, (%3);"
                    "vle32.v v4, (%4);"
                    "vmul.vv v8, v0, v4;"
                    "vmv.v.i v9, 0;"
                    "vredsum.vs v8, v8, v9;"
                    "vmv.x.s %1, v8"
                    : "=r"(vl), "=r"(temp_sum)
                    : "r"(remaining), "r"(a_ptr), "r"(bt_ptr)
                    : "v0", "v4", "v8", "v9"
                );
                sum += temp_sum;
            }
            C[i * cols_B + j] = sum;
        }
    }
}

void vector_mat_vec_rvv(int32_t *A, int32_t *v, int32_t *C, int rows_A, int cols_A) {
    memset(C, 0, rows_A * sizeof(int32_t));

    // Set vector length once before the loop
    size_t vlmax;
    asm volatile (
        "vsetvli %0, %1, e32, m1, ta, ma"
        : "=r"(vlmax)
        : "r"(cols_A)
    );

    for (int i = 0; i < rows_A; i++) {
        int32_t sum = 0;
        int32_t remaining = cols_A;
        int32_t *a_ptr = &A[i * cols_A];
        int32_t *v_ptr = v;

        // RVV loop for chunks of size vlmax
        while (remaining >= vlmax) {
            int32_t temp_sum;
            asm volatile (
                "vle32.v v0, (%1);"         // Load A[i][k:k+vlmax]
                "vle32.v v4, (%2);"         // Load v[k:k+vlmax]
                "vmul.vv v8, v0, v4;"       // Element-wise multiply
                "vmv.v.i v9, 0;"            // Zero for reduction
                "vredsum.vs v8, v8, v9;"    // Sum reduction
                "vmv.x.s %0, v8"            // Extract scalar
                : "=r"(temp_sum)
                : "r"(a_ptr), "r"(v_ptr)
                : "v0", "v4", "v8", "v9"
            );
            sum += temp_sum;
            a_ptr += vlmax;
            v_ptr += vlmax;
            remaining -= vlmax;
        }

        // Scalar fallback for remaining elements
        while (remaining > 0) {
            sum += *a_ptr * *v_ptr;
            a_ptr++;
            v_ptr++;
            remaining--;
        }

        C[i] = sum;
    }
}

void vector_add_rvv(int32_t *u, int32_t *v, int32_t *w, int n) {
    memset(w, 0, n * sizeof(int32_t));
    int32_t *u_ptr = u;
    int32_t *v_ptr = v;
    int32_t *w_ptr = w;
    int32_t remaining = n;

    // Set vector length once before the loop
    size_t vlmax;
    asm volatile (
        "vsetvli %0, %1, e32, m1, ta, ma"
        : "=r"(vlmax)
        : "r"(n)
    );

    // RVV loop for chunks of size vlmax
    while (remaining >= vlmax) {
        asm volatile (
            "vle32.v v0, (%0);"
            "vle32.v v4, (%1);"
            "vadd.vv v8, v0, v4;"
            "vse32.v v8, (%2)"
            :
            : "r"(u_ptr), "r"(v_ptr), "r"(w_ptr)
            : "v0", "v4", "v8"
        );
        u_ptr += vlmax;
        v_ptr += vlmax;
        w_ptr += vlmax;
        remaining -= vlmax;
    }

    // Fallback scalar loop for remaining elements
    while (remaining > 0) {
        *w_ptr = *u_ptr + *v_ptr;
        u_ptr++;
        v_ptr++;
        w_ptr++;
        remaining--;
    }
}