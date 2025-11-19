#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <iomanip>

// ==========================================
// 1. DATA STRUCTURES
// ==========================================
struct CSRMatrix {
    int rows;
    int cols;
    int nnz;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

// Helper for sorting raw .mtx data
struct Triplet {
    int r, c;
    double v;
    bool operator<(const Triplet& other) const {
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};

// ==========================================
// 2. PARSER (Safe Version)
// ==========================================
CSRMatrix readMTX(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] != '%') break;
    }
    std::stringstream ss(line);
    int M, N, L;
    ss >> M >> N >> L;
    
    std::vector<Triplet> triplets;
    triplets.reserve(L);
    int r, c;
    double v;
    while (file >> r >> c >> v) {
        int row_idx = r - 1;
        int col_idx = c - 1;
        if (row_idx >= M || col_idx >= N || row_idx < 0 || col_idx < 0) continue;
        triplets.push_back({row_idx, col_idx, v});
    }
    std::sort(triplets.begin(), triplets.end());

    CSRMatrix mat;
    mat.rows = M; mat.cols = N; mat.nnz = triplets.size();
    mat.values.reserve(mat.nnz);
    mat.col_indices.reserve(mat.nnz);
    mat.row_ptr.assign(M + 1, 0);

    int current_row = 0;
    int count = 0;
    for (const auto& t : triplets) {
        while (current_row < t.r && current_row < M) {
            current_row++;
            mat.row_ptr[current_row] = count;
        }
        mat.values.push_back(t.v);
        mat.col_indices.push_back(t.c);
        count++;
    }
    while (current_row < M) {
        current_row++;
        mat.row_ptr[current_row] = count;
    }
    return mat;
}

// ==========================================
// 3. ALGORITHMS
// ==========================================

// 1. Naive Dense (Baseline)
void dense_spmv_naive(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// 2. Optimized Dense (Unrolling)
void dense_spmv_optimized(const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        int j = 0;
        for (; j <= cols - 4; j += 4) {
            sum += A[i * cols + j] * x[j];
            sum += A[i * cols + j+1] * x[j+1];
            sum += A[i * cols + j+2] * x[j+2];
            sum += A[i * cols + j+3] * x[j+3];
        }
        for (; j < cols; j++) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// 3. Sparse CSR
void csr_spmv(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < A.rows; i++) {
        double sum = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; k++) {
            sum += A.values[k] * x[A.col_indices[k]];
        }
        y[i] = sum;
    }
}

// 4. Parallel Sparse CSR (OpenMP)
void csr_spmv_parallel(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; i++) {
        double sum = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; k++) {
            sum += A.values[k] * x[A.col_indices[k]];
        }
        y[i] = sum;
    }
}

// Generator
void generate_random(int rows, int cols, double sparsity, std::vector<double>& dense, CSRMatrix& sparse) {
    dense.resize(rows * cols, 0.0);
    sparse.rows = rows; sparse.cols = cols;
    sparse.row_ptr.push_back(0);
    int nnz_count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((rand() / (double)RAND_MAX) > sparsity) {
                double val = 1.0;
                dense[i * cols + j] = val;
                sparse.values.push_back(val);
                sparse.col_indices.push_back(j);
                nnz_count++;
            }
        }
        sparse.row_ptr.push_back(nnz_count);
    }
    sparse.nnz = nnz_count;
}

// ==========================================
// 4. MAIN EXPERIMENTS
// ==========================================
int main() {
    srand(42);
    
    // ---------------------------------------------------------
    // EXPERIMENT A: Sparsity Analysis (Fixed Size: 3000 x 3000)
    // ---------------------------------------------------------
    std::cout << "Running Experiment A: Sparsity Levels..." << std::endl;
    std::ofstream csv_sparsity("results/results_sparsity.csv");
    csv_sparsity << "Sparsity,NaiveDense,OptDense,SparseCSR\n";
    
    int N_fixed = 3000;
    std::vector<double> sparsities = {0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99};
    
    for (double s : sparsities) {
        std::vector<double> dense_mat, x(N_fixed, 1.0), y(N_fixed, 0.0);
        CSRMatrix sparse_mat;
        generate_random(N_fixed, N_fixed, s, dense_mat, sparse_mat);

        // 1. Naive
        auto start = std::chrono::high_resolution_clock::now();
        dense_spmv_naive(dense_mat, x, y, N_fixed, N_fixed);
        double t_naive = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        // 2. Opt Dense
        start = std::chrono::high_resolution_clock::now();
        dense_spmv_optimized(dense_mat, x, y, N_fixed, N_fixed);
        double t_opt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        // 3. Sparse
        start = std::chrono::high_resolution_clock::now();
        csr_spmv(sparse_mat, x, y);
        double t_sparse = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        csv_sparsity << s << "," << t_naive << "," << t_opt << "," << t_sparse << "\n";
        std::cout << "  Sparsity " << s * 100 << "% done." << std::endl;
    }
    csv_sparsity.close();

    // ---------------------------------------------------------
    // EXPERIMENT B: Matrix Size Scaling (Fixed Sparsity: 90%)
    // ---------------------------------------------------------
    std::cout << "Running Experiment B: Size Scaling..." << std::endl;
    std::ofstream csv_size("results/results_size.csv");
    csv_size << "Size,NaiveDense,OptDense,SparseCSR\n";

    double s_fixed = 0.90;
    
    // Testing sizes from 1000 to 10000
    for (int N = 1000; N <= 10000; N += 1000) {
        std::vector<double> dense_mat, x(N, 1.0), y(N, 0.0);
        CSRMatrix sparse_mat;
        generate_random(N, N, s_fixed, dense_mat, sparse_mat);

        // 1. Naive
        auto start = std::chrono::high_resolution_clock::now();
        dense_spmv_naive(dense_mat, x, y, N, N);
        double t_naive = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        // 2. Opt Dense
        start = std::chrono::high_resolution_clock::now();
        dense_spmv_optimized(dense_mat, x, y, N, N);
        double t_opt = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        // 3. Sparse
        start = std::chrono::high_resolution_clock::now();
        csr_spmv(sparse_mat, x, y);
        double t_sparse = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        csv_size << N << "," << t_naive << "," << t_opt << "," << t_sparse << "\n";
        std::cout << "  Size " << N << "x" << N << " done." << std::endl;
    }
    csv_size.close();

    // ---------------------------------------------------------
    // EXPERIMENT C: Huge Matrix File (Sparse Only)
    // ---------------------------------------------------------
    std::cout << "Running Experiment C: Huge Matrix (mc2depi.mtx)..." << std::endl;
    try {
        CSRMatrix bigMat = readMTX("data/mc2depi.mtx");
        std::vector<double> x_big(bigMat.cols, 1.0), y_big(bigMat.rows, 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        csr_spmv(bigMat, x_big, y_big);
        double t_basic = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        csr_spmv_parallel(bigMat, x_big, y_big);
        double t_parallel = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

        std::cout << "  Huge Matrix Results:" << std::endl;
        std::cout << "  Basic Sparse:    " << t_basic << " s" << std::endl;
        std::cout << "  Parallel Sparse: " << t_parallel << " s" << std::endl;
    } catch (...) {
        std::cout << "  Skipping huge matrix (file not found)." << std::endl;
    }

    return 0;
}