#include <stdint.h>
#include <riscv_vector.h>
#include <uart.h>

void vector_mul_scalar(int32_t *A, int32_t *B, int32_t *C, int rows_A, int cols_A, int cols_B) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int32_t sum = 0;
            for (int k = 0; k < cols_A; k++) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("vector_mul_scalar cycles: %d\n", end_cycles - start_cycles);
}

void transpose_scalar(int32_t *B, int32_t *BT, int rows_B, int cols_B) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; j++) {
            BT[j * rows_B + i] = B[i * cols_B + j];
        }
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("transpose_scalar cycles: %d\n", end_cycles - start_cycles);
}

void transpose_rvv(int32_t *B, int32_t *BT, int rows_B, int cols_B) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < rows_B; i++) {
        for (int j = 0; j < cols_B; ) {
            size_t vl = __riscv_vsetvl_e32m1(cols_B - j);
            vint32m1_t v_row = __riscv_vle32_v_i32m1(&B[i * cols_B + j], vl);
            __riscv_vsse32_v_i32m1(&BT[j * rows_B + i], rows_B * sizeof(int32_t), v_row, vl);
            j += vl;
        }
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("transpose_rvv cycles: %d\n", end_cycles - start_cycles);
}

void vector_mul_rvv(int32_t *A, int32_t *BT, int32_t *C, int rows_A, int cols_A, int cols_B) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int32_t sum = 0;
            int32_t remaining = cols_A;
            int32_t *a_ptr = &A[i * cols_A];
            int32_t *bt_ptr = &BT[j * cols_A];
            size_t vlmax = __riscv_vsetvl_e32m1(cols_A);
            while (remaining >= vlmax) {
                vint32m1_t v_a = __riscv_vle32_v_i32m1(a_ptr, vlmax);
                vint32m1_t v_bt = __riscv_vle32_v_i32m1(bt_ptr, vlmax);
                vint32m1_t v_prod = __riscv_vmul_vv_i32m1(v_a, v_bt, vlmax);
                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
                vint32m1_t v_sum = __riscv_vredsum_vs_i32m1_i32m1(v_prod, v_zero, vlmax);
                int32_t temp_sum = __riscv_vmv_x_s_i32m1_i32(v_sum);
                sum += temp_sum;
                a_ptr += vlmax;
                bt_ptr += vlmax;
                remaining -= vlmax;
            }
            if (remaining > 0) {
                size_t vl = __riscv_vsetvl_e32m1(remaining);
                vint32m1_t v_a = __riscv_vle32_v_i32m1(a_ptr, vl);
                vint32m1_t v_bt = __riscv_vle32_v_i32m1(bt_ptr, vl);
                vint32m1_t v_prod = __riscv_vmul_vv_i32m1(v_a, v_bt, vl);
                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
                vint32m1_t v_sum = __riscv_vredsum_vs_i32m1_i32m1(v_prod, v_zero, vl);
                int32_t temp_sum = __riscv_vmv_x_s_i32m1_i32(v_sum);
                sum += temp_sum;
            }
            C[i * cols_B + j] = sum;
        }
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("vector_mul_rvv cycles: %d\n", end_cycles - start_cycles);
}

void vector_mat_vec_scalar(int32_t *A, int32_t *v, int32_t *C, int rows_A, int cols_A) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < rows_A; i++) {
        int32_t sum = 0;
        for (int j = 0; j < cols_A; j++) {
            sum += A[i * cols_A + j] * v[j];
        }
        C[i] = sum;
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("vector_mat_vec_scalar cycles: %d\n", end_cycles - start_cycles);
}

void vector_mat_vec_rvv(int32_t *A, int32_t *v, int32_t *C, int rows_A, int cols_A) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < rows_A; i++) {
        int32_t sum = 0;
        int32_t remaining = cols_A;
        int32_t *a_ptr = &A[i * cols_A];
        int32_t *v_ptr = v;
        size_t vlmax = __riscv_vsetvl_e32m1(cols_A);
        while (remaining >= vlmax) {
            vint32m1_t v_a = __riscv_vle32_v_i32m1(a_ptr, vlmax);
            vint32m1_t v_vec = __riscv_vle32_v_i32m1(v_ptr, vlmax);
            vint32m1_t v_prod = __riscv_vmul_vv_i32m1(v_a, v_vec, vlmax);
            vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax);
            vint32m1_t v_sum = __riscv_vredsum_vs_i32m1_i32m1(v_prod, v_zero, vlmax);
            int32_t temp_sum = __riscv_vmv_x_s_i32m1_i32(v_sum);
            sum += temp_sum;
            a_ptr += vlmax;
            v_ptr += vlmax;
            remaining -= vlmax;
        }
        while (remaining > 0) {
            sum += *a_ptr * *v_ptr;
            a_ptr++;
            v_ptr++;
            remaining--;
        }
        C[i] = sum;
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("vector_mat_vec_rvv cycles: %d\n", end_cycles - start_cycles);
}

void vector_add_scalar(int32_t *u, int32_t *v, int32_t *w, int n) {
    uint32_t start_cycles = read_mcycle();
    for (int i = 0; i < n; i++) {
        w[i] = u[i] + v[i];
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("vector_add_scalar cycles: %d\n", end_cycles - start_cycles);
}

void vector_add_rvv(int32_t *u, int32_t *v, int32_t *w, int n) {
    uint32_t start_cycles = read_mcycle();
    int32_t *u_ptr = u;
    int32_t *v_ptr = v;
    int32_t *w_ptr = w;
    int32_t remaining = n;
    size_t vlmax = __riscv_vsetvl_e32m1(n);
    while (remaining >= vlmax) {
        vint32m1_t v_u = __riscv_vle32_v_i32m1(u_ptr, vlmax);
        vint32m1_t v_v = __riscv_vle32_v_i32m1(v_ptr, vlmax);
        vint32m1_t v_result = __riscv_vadd_vv_i32m1(v_u, v_v, vlmax);
        __riscv_vse32_v_i32m1(w_ptr, v_result, vlmax);
        u_ptr += vlmax;
        v_ptr += vlmax;
        w_ptr += vlmax;
        remaining -= vlmax;
    }
    if (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t v_u = __riscv_vle32_v_i32m1(u_ptr, vl);
        vint32m1_t v_v = __riscv_vle32_v_i32m1(v_ptr, vl);
        vint32m1_t v_result = __riscv_vadd_vv_i32m1(v_u, v_v, vl);
        __riscv_vse32_v_i32m1(w_ptr, v_result, vl);
    }
    uint32_t end_cycles = read_mcycle();
    uart_printf("vector_add_rvv cycles: %d\n", end_cycles - start_cycles);
}

static inline uint32_t read_mcycle(void) {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}