# Big Data Course Assignments

**Institution:** Universidad de Las Palmas de Gran Canaria  
**Degree:** Grado en Ciencia e Ingeniería de Datos  
**Subject:** Big Data

## Overview

This repository contains the individual assignments for the Big Data course. The core focus of these projects is **Matrix Multiplication Benchmarking**.

The goal is to evaluate the performance, scalability, and resource usage of various algorithms under different computational conditions. This involves testing implementations on square matrices ranging from small sizes ($10 \times 10$) to large-scale datasets ($10,000 \times 10,000+$) to simulate Big Data workloads.

## Project Structure

Each assignment is isolated in its own directory containing the source code, benchmark scripts, and performance reports.

### [Task 1: Basic Matrix Multiplication](./task1)
**Topic:** Language Performance Comparison  
**Status:** Completed

Comparison of a standard $O(n^3)$ matrix multiplication algorithm implemented in three languages:
- **C:** Native compiled performance (GCC).
- **Java:** JVM performance (JMH).
- **Python:** Interpreted performance.

**Key Metrics:** Execution time and memory usage.

---

### [Task 2: Optimized & Sparse Matrices](./task2)
**Topic:** Optimization Techniques and Sparse Data Structures  
**Status:** Completed

Investigation into techniques to improve computational cost and handle memory constraints:
- **Optimized Dense:** Manual loop unrolling and compiler optimizations.
- **Sparse Matrices:** Implementation of Compressed Sparse Row (CSR) format.
- **Parallelism:** Multi-threading using OpenMP.

**Key Metrics:** Sparsity impact, Speedup factors, and Hardware bottlenecks (Cache misses).

---

### Task 3
*To be added.*

---

### Task 4
*To be added.*

## Benchmarking Methodology

All assignments follow a strict benchmarking protocol:
1.  **Scalability:** Testing on increasingly larger matrix sizes.
2.  **Metrics:** Measurement of Execution Time, Memory Usage, and CPU throughput.
3.  **Profiling:** Use of tools like `perf` (Linux), `JMH` (Java), and `pytest-benchmark` (Python).