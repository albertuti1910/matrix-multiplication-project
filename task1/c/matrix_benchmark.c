#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "matrix_c.h"

// Forward declarations
void run_benchmark_with_results(int n, int iterations, double* avg_time_out, double* min_time_out,
                                 double* max_time_out, long* mem_kb_out, double* gflops_out);

double get_time_seconds(struct timeval start, struct timeval stop) {
    return (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) * 1e-6;
}

long get_memory_usage_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    long memory = 0;
    char line[128];

    if (file != NULL) {
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line + 6, "%ld", &memory);
                break;
            }
        }
        fclose(file);
    }

    // If /proc/self/status is not available (e.g., on macOS or in containers)
    // return 0 to indicate memory tracking is not available
    return memory;
}

void print_memory_stats(const char* label, int n) {
    long mem_kb = get_memory_usage_kb();

    if (mem_kb > 0) {
        double mem_mb = mem_kb / 1024.0;
        // Calculate theoretical memory for matrices
        double matrix_size_mb = (3.0 * n * n * sizeof(double)) / (1024.0 * 1024.0);

        printf("[%s] Matrix size: %d, RSS Memory: %.2f MB, Theoretical matrix size: %.2f MB\n",
               label, n, mem_mb, matrix_size_mb);
    } else {
        // Memory tracking not available
        double matrix_size_mb = (3.0 * n * n * sizeof(double)) / (1024.0 * 1024.0);
        printf("[%s] Matrix size: %d, Theoretical matrix size: %.2f MB (RSS unavailable)\n",
               label, n, matrix_size_mb);
    }
}

void run_benchmark(int n, int iterations) {
    double avg_time, min_time, max_time, gflops;
    long mem_kb;
    run_benchmark_with_results(n, iterations, &avg_time, &min_time, &max_time, &mem_kb, &gflops);
}

void run_benchmark_with_results(int n, int iterations, double* avg_time_out, double* min_time_out,
                                 double* max_time_out, long* mem_kb_out, double* gflops_out) {
    printf("\n=== Benchmarking %dx%d matrices ===\n", n, n);

    long mem_before = get_memory_usage_kb();

    double** a = create_matrix(n);
    double** b = create_matrix(n);
    double** c = create_matrix(n);

    initialize_random_matrix(a, n, 42);
    initialize_random_matrix(b, n, 43);
    initialize_zero_matrix(c, n);

    long mem_after_alloc = get_memory_usage_kb();
    long mem_allocated = mem_after_alloc - mem_before;

    if (mem_allocated > 0) {
        printf("Memory allocated for matrices: %.2f MB\n", mem_allocated / 1024.0);
    } else {
        double theoretical_mem = (3.0 * n * n * sizeof(double)) / (1024.0 * 1024.0);
        printf("Theoretical memory for matrices: %.2f MB\n", theoretical_mem);
    }
    print_memory_stats("After allocation", n);

    struct timeval start, stop;
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;

    // Warmup
    printf("Warming up...\n");
    for (int iter = 0; iter < 2; iter++) {
        matrix_multiply(a, b, c, n);
    }

    // Actual benchmark
    printf("Running %d measurement iterations...\n", iterations);
    for (int iter = 0; iter < iterations; iter++) {
        initialize_zero_matrix(c, n);

        gettimeofday(&start, NULL);
        matrix_multiply(a, b, c, n);
        gettimeofday(&stop, NULL);

        double iter_time = get_time_seconds(start, stop);
        total_time += iter_time;

        if (iter_time < min_time) min_time = iter_time;
        if (iter_time > max_time) max_time = iter_time;

        printf("  Iteration %d: %.6f s\n", iter + 1, iter_time);
    }

    double avg_time = total_time / iterations;
    double gflops = (2.0 * n * n * n / avg_time) / 1e9;

    printf("\n--- Results for %dx%d ---\n", n, n);
    printf("Iterations: %d\n", iterations);
    printf("Average time: %.6f s (%.2f ms)\n", avg_time, avg_time * 1000);
    printf("Min time: %.6f s\n", min_time);
    printf("Max time: %.6f s\n", max_time);
    printf("Total time: %.6f s\n", total_time);
    printf("GFLOPS: %.3f\n", gflops);

    print_memory_stats("After benchmark", n);

    // Return values for CSV export
    *avg_time_out = avg_time;
    *min_time_out = min_time;
    *max_time_out = max_time;
    *mem_kb_out = get_memory_usage_kb();
    *gflops_out = gflops;

    free_matrix(a, n);
    free_matrix(b, n);
    free_matrix(c, n);
}

int main(void) {
    int sizes[] = {128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int iterations = 5;

    // Open CSV file for results
    FILE* csv_file = fopen("../results/c_results.csv", "w");
    if (csv_file != NULL) {
        fprintf(csv_file, "language,matrix_size,mean_time_ms,min_time_ms,max_time_ms,memory_mb,iterations,gflops\n");
    }

    printf("========================================\n");
    printf("Matrix Multiplication Benchmark (C)\n");
    printf("========================================\n");
    printf("Algorithm: Basic O(n^3)\n");
    printf("Compiler: GCC with -O2 optimization\n");
    printf("Data type: double (8 bytes)\n");
    printf("========================================\n");

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        double avg_time, min_time, max_time, gflops;
        long mem_kb;

        // Run benchmark and capture results
        run_benchmark_with_results(n, iterations, &avg_time, &min_time, &max_time, &mem_kb, &gflops);

        // Write to CSV
        if (csv_file != NULL) {
            fprintf(csv_file, "C,%d,%.6f,%.6f,%.6f,%.2f,%d,%.3f\n",
                    n,
                    avg_time * 1000,  // Convert to ms
                    min_time * 1000,
                    max_time * 1000,
                    mem_kb / 1024.0,
                    iterations,
                    gflops);
        }

        printf("\n");
    }

    if (csv_file != NULL) {
        fclose(csv_file);
        printf("CSV results exported to: ../results/c_results.csv\n");
    }

    printf("========================================\n");
    printf("Benchmark completed successfully!\n");
    printf("========================================\n");

    return 0;
}
