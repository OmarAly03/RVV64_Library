#include <iostream>
#include <cstdlib>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t M = 4, N = 4, K = 4; // defaults
    if (argc >= 4) {
        int m = atoi(argv[1]);
        int n = atoi(argv[2]);
        int k = atoi(argv[3]);
        if (m > 0 && n > 0 && k > 0) {
            M = static_cast<size_t>(m);
            N = static_cast<size_t>(n);
            K = static_cast<size_t>(k);
        } else {
            cerr << "Invalid arguments. Using default size " << M << "x" << N << " (K=" << K << ")" << endl;
        }
    } else if (argc == 2) {
        int size = atoi(argv[1]);
        if (size > 0) {
            M = N = K = static_cast<size_t>(size);
        } else {
            cerr << "Invalid argument. Using default size " << M << "x" << N << " (K=" << K << ")" << endl;
        }
    }

    // --- MEMORY ALLOCATION ---
	float* A = new float[M * K];
	float* B = new float[K * N];
	float* C = new float[M * N];

    if (!A || !B || !C) {
        cerr << "Memory allocation failed!" << endl;
        return 1;
    }

    // --- INITIALIZE MATRICES ---
    srand(0); // fixed seed for reproducibility
    for (size_t i = 0; i < M * K; i++) {
        A[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }
    for (size_t i = 0; i < K * N; i++) {
        B[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    write_matrix_binary("./output_files/A.bin", A, M * K);
    write_matrix_binary("./output_files/B.bin", B, K * N);

	matmul_scalar(A, B, C, M, N, K);
	write_matrix_binary("./output_files/c_scalar.bin", C, M * N);

	matmul_e32m1(A, B, C, M, N, K);
	write_matrix_binary("./output_files/c_e32m1.bin", C, M * N);

	matmul_e32m2(A, B, C, M, N, K);
	write_matrix_binary("./output_files/c_e32m2.bin", C, M * N);

    matmul_e32m4(A, B, C, M, N, K);
	write_matrix_binary("./output_files/c_e32m4.bin", C, M * N);

	matmul_e32m8(A, B, C, M, N, K);
	write_matrix_binary("./output_files/c_e32m8.bin", C, M * N);

	delete[] A;
	delete[] B;
	delete[] C;

    return 0;
}