#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "./include/defs.h"

using namespace std;

int main(int argc, char* argv[]) {
    // --- HANDLE ARGUMENTS ---
    size_t M = 4, N = 4, K = 4; // defaults
    size_t tilesize = 8; // default tile size
    
    if (argc >= 5) {
        // Format: ./run_matmul M N K tilesize
        int m = atoi(argv[1]);
        int n = atoi(argv[2]);
        int k = atoi(argv[3]);
        int tile = atoi(argv[4]);
        if (m > 0 && n > 0 && k > 0 && tile > 0) {
            M = static_cast<size_t>(m);
            N = static_cast<size_t>(n);
            K = static_cast<size_t>(k);
            tilesize = static_cast<size_t>(tile);
        } else {
            cerr << "Invalid arguments. Usage: ./run_matmul M N K tilesize" << endl;
            cerr << "Using defaults: " << M << "x" << N << " (K=" << K << "), tile=" << tilesize << endl;
        }
    } else if (argc == 4) {
        // Format: ./run_matmul M N K (use default tile size)
        int m = atoi(argv[1]);
        int n = atoi(argv[2]);
        int k = atoi(argv[3]);
        if (m > 0 && n > 0 && k > 0) {
            M = static_cast<size_t>(m);
            N = static_cast<size_t>(n);
            K = static_cast<size_t>(k);
            tilesize = std::min(static_cast<size_t>(32), std::min(M, std::min(N, K)));
        } else {
            cerr << "Invalid arguments. Using defaults." << endl;
        }
    } else if (argc == 3) {
        // Format: ./run_matmul size tilesize (square matrices)
        int size = atoi(argv[1]);
        int tile = atoi(argv[2]);
        if (size > 0 && tile > 0) {
            M = N = K = static_cast<size_t>(size);
            tilesize = static_cast<size_t>(tile);
        } else {
            cerr << "Invalid arguments. Usage: ./run_matmul size tilesize" << endl;
            cerr << "Using defaults: " << M << "x" << N << ", tile=" << tilesize << endl;
        }
    } else if (argc == 2) {
        // Format: ./run_matmul size (square matrices, auto tile size)
        int size = atoi(argv[1]);
        if (size > 0) {
            M = N = K = static_cast<size_t>(size);
            tilesize = std::min(static_cast<size_t>(32), static_cast<size_t>(size));
        } else {
            cerr << "Invalid argument. Using defaults." << endl;
        }
    }

    // Validate tile size
    if (tilesize > M || tilesize > N || tilesize > K) {
        cout << "Warning: Tile size (" << tilesize << ") is larger than matrix dimensions." << endl;
        tilesize = std::min({M, N, K});
        cout << "Adjusting tile size to: " << tilesize << endl;
    }

    cout << "Matrix dimensions: " << M << "x" << N << " (K=" << K << ")" << endl;
    cout << "Tile size: " << tilesize << endl;
    cout << "Total operations: " << (2.0 * M * N * K / 1e6) << " million FLOPs" << endl;

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

    matmul_tiled_scalar(A, B, C, M, N, K, tilesize, tilesize, tilesize);
    write_matrix_binary("./output_files/c_tiled_scalar.bin", C, M * N);

    matmul_tiled_e32m8(A, B, C, M, N, K, tilesize, tilesize, tilesize);
    write_matrix_binary("./output_files/c_tiled_e32m8.bin", C, M * N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}