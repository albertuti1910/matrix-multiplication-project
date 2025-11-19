#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <algorithm> // Required for std::sort

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
    // Overload < operator to sort by Row, then Column
    bool operator<(const Triplet& other) const {
        if (r != other.r) return r < other.r;
        return c < other.c;
    }
};

// ==========================================
// 2. ROBUST PARSER (Safe Version)
// ==========================================
CSRMatrix readMTX(const std::string& filename) {
    std::cout << "   [Parser] Opening file..." << std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] != '%') break;
    }

    std::stringstream ss(line);
    int M, N, L;
    ss >> M >> N >> L;
    
    std::cout << "   [Parser] Header: Rows=" << M << ", Cols=" << N << ", NNZ=" << L << std::endl;

    // 1. Read all data into a list of Triplets
    std::vector<Triplet> triplets;
    triplets.reserve(L);

    int r, c;
    double v;
    int skipped_count = 0;

    while (file >> r >> c >> v) {
        // Convert 1-based MTX to 0-based C++
        int row_idx = r - 1;
        int col_idx = c - 1;

        // SAFETY CHECK: Ignore indices outside the declared dimensions
        if (row_idx >= M || col_idx >= N || row_idx < 0 || col_idx < 0) {
            skipped_count++;
            continue;
        }

        triplets.push_back({row_idx, col_idx, v});
    }

    if (skipped_count > 0) {
        std::cerr << "   [Parser] WARNING: Skipped " << skipped_count << " invalid entries (out of bounds)." << std::endl;
    }
    
    // 2. Sort the data (Critical step for CSR)
    std::cout << "   [Parser] Sorting " << triplets.size() << " entries..." << std::endl;
    std::sort(triplets.begin(), triplets.end());

    // 3. Convert to CSR Format
    std::cout << "   [Parser] Building CSR..." << std::endl;
    CSRMatrix mat;
    mat.rows = M;
    mat.cols = N;
    mat.nnz = triplets.size(); // Update NNZ to actual valid count
    mat.values.reserve(mat.nnz);
    mat.col_indices.reserve(mat.nnz);
    mat.row_ptr.assign(M + 1, 0);

    int current_row = 0;
    int count = 0;

    for (const auto& t : triplets) {
        // If we jumped to a new row, fill the row_ptr for the empty/passed rows
        while (current_row < t.r && current_row < M) {
            current_row++;
            mat.row_ptr[current_row] = count;
        }
        mat.values.push_back(t.v);
        mat.col_indices.push_back(t.c);
        count++;
    }
    // Finish filling row_ptr for the last rows
    while (current_row < M) {
        current_row++;
        mat.row_ptr[current_row] = count;
    }

    return mat;
}

// ==========================================
// 3. ALGORITHMS
// ==========================================

// Optimized Dense (Loop Unrolling)
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

// Basic CSR SpMV
void csr_spmv(const CSRMatrix& A, const std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < A.rows; i++) {
        double sum = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i+1]; k++) {
            sum += A.values[k] * x[A.col_indices[k]];
        }
        y[i] = sum;
    }
}

// Parallel CSR SpMV (OpenMP)
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

void generate_random_dense(int rows, int cols, double sparsity, std::vector<double>& dense, CSRMatrix& sparse) {
    dense.resize(rows * cols, 0.0);
    sparse.rows = rows;
    sparse.cols = cols;
    sparse.row_ptr.push_back(0);
    int nnz_count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((rand() / (double)RAND_MAX) > sparsity) {
                double val = (rand() % 100) / 10.0;
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
// 4. MAIN
// ==========================================

int main(int argc, char* argv[]) {
    // === PART 1: DENSE vs SPARSE (Comparison) ===
    std::cout << "=== PART 1: Dense vs Sparse Comparison (Generated Matrix 4000x4000) ===" << std::endl;
    int N = 4000; 
    double sparsity = 0.90; 
    
    std::vector<double> dense_matrix;
    CSRMatrix sparse_matrix_gen;
    
    std::cout << "Generating data (Sparsity: " << sparsity * 100 << "%)..." << std::endl;
    generate_random_dense(N, N, sparsity, dense_matrix, sparse_matrix_gen);
    std::vector<double> x(N, 1.0);
    std::vector<double> y(N, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    dense_spmv_optimized(dense_matrix, x, y, N, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dense_time = end - start;
    std::cout << "Optimized Dense Time: " << dense_time.count() << " s" << std::endl;

    std::fill(y.begin(), y.end(), 0.0);
    start = std::chrono::high_resolution_clock::now();
    csr_spmv(sparse_matrix_gen, x, y);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sparse_time = end - start;
    std::cout << "Basic Sparse Time:    " << sparse_time.count() << " s" << std::endl;
    std::cout << "Speedup: " << dense_time.count() / sparse_time.count() << "x" << std::endl;
    std::cout << std::endl;

    // === PART 2: HUGE FILE TEST (Sparse Only) ===
    std::cout << "=== PART 2: Huge Matrix Test (Sparse Only) ===" << std::endl;
    std::string filename = "mc2depi.mtx";
    
    std::cout << "Reading " << filename << "..." << std::endl;
    try {
        CSRMatrix bigMat = readMTX(filename);
        std::cout << "Matrix Loaded: " << bigMat.rows << " x " << bigMat.cols << " with " << bigMat.nnz << " non-zeros." << std::endl;
        
        std::vector<double> x_big(bigMat.cols, 1.0);
        std::vector<double> y_big(bigMat.rows, 0.0);

        start = std::chrono::high_resolution_clock::now();
        csr_spmv(bigMat, x_big, y_big);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> big_basic_time = end - start;
        std::cout << "Basic Sparse Time:     " << big_basic_time.count() << " s" << std::endl;

        std::fill(y_big.begin(), y_big.end(), 0.0);
        start = std::chrono::high_resolution_clock::now();
        csr_spmv_parallel(bigMat, x_big, y_big);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> big_parallel_time = end - start;
        std::cout << "Parallel Sparse Time:  " << big_parallel_time.count() << " s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}