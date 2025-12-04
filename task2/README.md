# Optimized Sparse Matrix-Vector Multiplication (SpMV) Benchmark

Benchmark comparing Dense and Sparse Matrix-Vector Multiplication performance using C++ and OpenMP.

## Requirements

- **C++:** GCC compiler with OpenMP support
- **Python:** Python 3.x with pandas and matplotlib
- **Tools:** perf (optional, for bottleneck analysis)

## Quick Start

### 1. Compile Benchmark

```bash
g++ -O2 -fopenmp -o spmv_bench spmv_final.cpp
```

### 2. Run Benchmark

```bash
./spmv_bench
```

**Note:** If `mc2depi.mtx` is present in the directory, the "Huge Matrix" test will run automatically.

**Output:**
- `results_sparsity.csv`
- `results_size.csv`

### 3. Generate Graphs

```bash
python plot_results.py
```

**Output:**
- `graph_1_time_vs_sparsity.png`
- `graph_2_speedup.png`
- `graph_3_scalability.png`
- `graph_4_memory_usage.png`

---

## Algorithms Implemented

1. **Naive Dense:** Standard nested loops ($O(N^2)$).
2. **Optimized Dense:** Manual loop unrolling (factor 4) to improve pipelining.
3. **Sparse CSR:** Compressed Sparse Row format ($O(NNZ)$).
4. **Parallel Sparse CSR:** OpenMP multithreading optimization.

---

## Experiments

The benchmark performs three specific tests:

1. **Sparsity Analysis:** Fixed size (3000×3000), varying sparsity from 0% to 99%.
2. **Size Scaling:** Fixed sparsity (90%), varying size from N=1000 to N=10000.
3. **Huge Matrix Test:** Loads `mc2depi.mtx` (525,825 × 525,825) to test memory limits.

---


## Troubleshooting

- **Missing OpenMP:** Ensure `libgomp` or equivalent is installed (`sudo pacman -S gcc` on Arch).
- **Python Errors:** Install dependencies: `pip install pandas matplotlib`.
- **Perf Permission Denied:** Run with sudo: `sudo perf stat -d ./spmv_bench`.

---

## References

- **Optimization of Sparse Matrix-Vector Multiplication on Emerging Multicore Platforms**: S. Williams. https://sparse.tamu.edu/Williams/mc2depi
- **Matrix Market Format**: https://math.nist.gov/MatrixMarket/formats.html