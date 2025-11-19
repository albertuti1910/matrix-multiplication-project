import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    df_sparsity = pd.read_csv("results/results_sparsity.csv")
    df_size = pd.read_csv("results/results_size.csv")
except FileNotFoundError:
    print("Error: CSV files not found. Run the C++ benchmark first.")
    exit()

IMAGE_FOLDER = "images/"

# Settings for report graphs
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.grid'] = True

# Graph 1: Execution Time vs Sparsity
plt.figure()
plt.plot(df_sparsity['Sparsity']*100, df_sparsity['NaiveDense'], marker='o', label='Naive Dense')
plt.plot(df_sparsity['Sparsity']*100, df_sparsity['OptDense'], marker='s', label='Optimized Dense')
plt.plot(df_sparsity['Sparsity']*100, df_sparsity['SparseCSR'], marker='^', label='Sparse CSR', linewidth=2)

plt.xlabel('Sparsity (% of Zeros)')
plt.ylabel('Time (seconds)')
plt.title('Impact of Sparsity on Performance (Size 3000x3000)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}graph_1_time_vs_sparsity.png")
plt.close()
print("Saved: graph_1_time_vs_sparsity.png")

# Graph 2: Speedup vs Sparsity
plt.figure()
speedup = df_sparsity['OptDense'] / df_sparsity['SparseCSR']

plt.plot(df_sparsity['Sparsity']*100, speedup, color='green', marker='D', linestyle='--', label='Speedup (Opt Dense / Sparse)')
plt.axhline(1, color='red', linestyle=':', label='Breakeven Point (1x)')

plt.xlabel('Sparsity (% of Zeros)')
plt.ylabel('Speedup Factor (x times)')
plt.title('Speedup of Sparse CSR over Dense')
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}graph_2_speedup.png")
plt.close()
print("Saved: graph_2_speedup.png")

# Graph 3: Scalability (Time vs Size)
plt.figure()
plt.plot(df_size['Size'], df_size['NaiveDense'], label='Naive Dense')
plt.plot(df_size['Size'], df_size['OptDense'], label='Optimized Dense')
plt.plot(df_size['Size'], df_size['SparseCSR'], label='Sparse CSR', linewidth=2)

plt.xlabel('Matrix Dimension (N)')
plt.ylabel('Time (seconds)')
plt.title('Scalability: Execution Time vs Matrix Size (90% Sparsity)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}graph_3_scalability.png")
plt.close()
print("Saved: graph_3_scalability.png")

# Graph 4: Theoretical Memory Usage
plt.figure()
sizes = df_size['Size']

# Dense Memory: N * N * 8 bytes (double precision)
mem_dense = (sizes**2 * 8) / (1024**2) # Convert to MB

# Sparse Memory (90% zeros):
# Values (doubles) + Col_Indices (ints) + Row_Ptrs (ints)
# NNZ approx = 0.1 * N * N
nnz = 0.1 * sizes**2
mem_sparse = (nnz * 8 + nnz * 4 + (sizes+1) * 4) / (1024**2) # MB

plt.plot(sizes, mem_dense, color='red', label='Dense Matrix RAM')
plt.plot(sizes, mem_sparse, color='blue', label='Sparse CSR RAM')

plt.xlabel('Matrix Dimension (N)')
plt.ylabel('Memory Usage (MB)')
plt.title('Theoretical Memory Usage (90% Sparsity)')
plt.yscale('log') # Log scale is critical here
plt.grid(True, which="both", ls="-") # finer grid for log scale
plt.legend()
plt.tight_layout()
plt.savefig(f"{IMAGE_FOLDER}graph_4_memory_usage.png")
plt.close()
print("Saved: graph_4_memory_usage.png")