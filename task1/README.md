# Matrix Multiplication Benchmark

Simple benchmark comparing C, Java, and Python matrix multiplication performance.

## Requirements

- **Windows users:** Use WSL2 (Windows Subsystem for Linux)
- **Linux/Mac users:** Run directly in terminal
- **Java**: JDK 11+ and Maven + JMH
- **Python**: Python 3.8+
- **C**: GCC compiler and perf (requires WSL2 on Windows)

## Quick Start

### 1. C Benchmark

```bash
cd c
make clean
make perf
./run_benchmark.sh
```

**Results:** `results/c_results.csv`

---

### 2. Java Benchmark

```bash
cd java
mvn clean install
java -jar target/benchmarks.jar -rf csv -rff results/java_results.csv
```

**Results:** `results/java_results.csv`

---

### 3. Python Benchmark

```bash
cd python
python -m venv .venv
source .venv/bin/activate  # On Windows WSL: source .venv/bin/activate
pip install -r requirements.txt
pytest test_matrix_benchmark.py --benchmark-only --benchmark-save=results
```

**Results:** `results/python_results.csv`

---

## View Results

All CSV files are saved in `results/` directory:

Open with Excel or any CSV viewer.

---

## Generate Graphs

After running all three benchmarks:

```bash
# Generate graphs
cd results
python generate_graphs.py
```

**Output:**
- `graph1_execution_time.pdf` - Main comparison
- `graph2_speedup.pdf` - Speedup analysis
- `graph3_complexity.pdf` - O(n³) verification
- `graph4_memory.pdf` - Memory usage
- `table_comparison.png` - Summary table

---

## Matrix Sizes Tested

- 128 × 128
- 256 × 256
- 512 × 512
- 1024 × 1024

---

## Expected Runtime

- **C:** ~2 minutes total
- **Java:** ~5 minutes total (includes JVM warmup)
- **Python:** ~30 minutes total (1024×1024 is slow)

---


## Project Structure

```
matrix-benchmark-project/
├── c/
│   ├── matrix_benchmark.c
│   ├── matrix_c.c
│   ├── matrix_c.h
│   ├── Makefile
├── java/
│   ├── src/
│   │   ├── MatrixBenchmark.java
│   │   └── MatrixMultiplier.java
│   ├── pom.xml
├── python/
│   ├── matrix_multiplier.py
│   ├── test_matrix_benchmark.py
│   ├── requirements.txt
└── results/
    └── generate_graphs.py
```

---

## Need Help?

1. Check you're in the correct directory
2. Make sure all dependencies are installed
3. Read error messages
4. Run again

---

## TLDR

Run each benchmark, collect the CSVs, generate graphs, and you're done.

## References

- **JMH Documentation**: https://openjdk.java.net/projects/code-tools/jmh/
- **pytest-benchmark**: https://pytest-benchmark.readthedocs.io/
- **perf Documentation**: https://perf.wiki.kernel.org/