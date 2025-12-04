#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <string>

using Scalar = double;

void init_matrix(std::vector<Scalar>& M, int N) {
    // Fixed seed for fair comparison across runs
    std::mt19937 gen(42);
    std::uniform_real_distribution<Scalar> dis(0.0, 1.0);
    for (int i = 0; i < N * N; ++i) M[i] = dis(gen);
}

// 1. Basic
void multiply_basic(const std::vector<Scalar>& A, const std::vector<Scalar>& B, std::vector<Scalar>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Scalar sum = 0.0;
            for (int k = 0; k < N; ++k) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// 2. Parallel (No Transpose - shows cache issues)
void multiply_parallel(const std::vector<Scalar>& A, const std::vector<Scalar>& B, std::vector<Scalar>& C, int N) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Scalar sum = 0.0;
            for (int k = 0; k < N; ++k) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// 3. Vectorized + Parallel + Transposed (Optimized)
void multiply_vectorized(const std::vector<Scalar>& A, const std::vector<Scalar>& B, std::vector<Scalar>& C, int N) {
    std::vector<Scalar> B_T(N * N);

    // Transpose B
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            B_T[i * N + j] = B[j * N + i];

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256d vec_sum = _mm256_setzero_pd();
            int k = 0;
            for (; k <= N - 4; k += 4) {
                __m256d vec_A = _mm256_loadu_pd(&A[i * N + k]);
                __m256d vec_B = _mm256_loadu_pd(&B_T[j * N + k]);
                vec_sum = _mm256_fmadd_pd(vec_A, vec_B, vec_sum);
            }
            double temp[4];
            _mm256_storeu_pd(temp, vec_sum);
            Scalar sum = temp[0] + temp[1] + temp[2] + temp[3];
            for (; k < N; ++k) sum += A[i * N + k] * B_T[j * N + k];
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) return 1;
    int N = std::stoi(argv[1]);
    std::string mode = argv[2];

    std::vector<Scalar> A(N * N), B(N * N), C(N * N);
    init_matrix(A, N);
    init_matrix(B, N);

    if (mode == "basic") multiply_basic(A, B, C, N);
    else if (mode == "parallel") multiply_parallel(A, B, C, N);
    else if (mode == "vectorized") multiply_vectorized(A, B, C, N);

    return 0;
}