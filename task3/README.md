# Task 3: Parallel & Vectorized Matrix Multiplication

Benchmark comparing Basic, Parallel (OpenMP), and Vectorized (AVX2 + Transpose) Matrix Multiplication performance.

## Requirements

- **C++:** GCC compiler with OpenMP support
- **Python:** Python 3.x with `pandas` and `matplotlib`
- **Tools:** `perf` (Linux hardware counter profiler)

## Quick Start

### Run Benchmark
The Python script automatically compiles the C++ code and runs the tests.

```bash
python run_task3.py
```

**Output:**
- Graphs and CSV data will be saved in the `results/` directory.

---

## Algorithms Implemented

1.  **Basic:** Standard naïve implementation ($O(N^3)$).
2.  **Parallel (OpenMP):**
    -   Uses `#pragma omp parallel for` on the outer loops.
    -   **Observation:** Suffers from "Cache Thrashing" (High L3 Cache Misses) due to column-major memory access on Matrix B.
3.  **Vectorized (Optimized):**
    -   **Transpose:** Matrix B is transposed to ensure linear memory access.
    -   **AVX2 Intrinsics:** Uses `_mm256_fmadd_pd` to calculate 4 double-precision floating-point numbers per cycle.
    -   **OpenMP:** The vectorized blocks are distributed across cores.

---

## Project Structure

```text
task3/
├── task3_matrix_perf.cpp  # C++ Source (Contains all 3 implementations)
├── run_task3.py           # Automation script
├── README.md              # This file
├── report_task3.pdf       # Final PDF Report
└── results/               # Generated Output
    ├── advanced_results.csv
    ├── plot_time.pdf
    ├── plot_speedup.pdf
    ├── plot_gflops.pdf
    └── plot_cache_misses.pdf
```

---

## Expected Results (Intel i7-9750H)

| Metric | Basic | Parallel | Vectorized |
| :--- | :--- | :--- | :--- |
| **Time (N=2048)** | ~50s | ~20s | **~0.55s** |
| **Speedup** | 1.0x | ~2.5x | **~91x** |
| **L3 Cache Misses** | ~7.9B | ~6.9B | **~0.14B** |

*Note: The naive Parallel version hits a "Memory Wall" (bandwidth saturation), while the Vectorized version maximizes CPU throughput.*

---
