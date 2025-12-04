import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import sys
import re


SIZES = [128, 256, 512, 1024, 2048]
MODES = ["basic", "parallel", "vectorized"]

SOURCE = "task3_matrix.cpp"
EXE = "task3_matrix"
CSV_FILE = "results/task3_results.csv"

def compile_code():
    print("Compiling...")
    cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-mfma", SOURCE, "-o", EXE]
    subprocess.check_call(cmd)

def parse_perf_output(stderr_output):
    """ Extract cycles, instructions, and cache-misses from perf output """
    metrics = {"seconds": 0.0, "cache-misses": 0, "instructions": 0, "cycles": 0}

    for line in stderr_output.splitlines():
        line = line.strip()
        if "seconds time elapsed" in line:
            metrics["seconds"] = float(line.split()[0].replace(',', ''))
        elif "cache-misses" in line:
            parts = line.split()
            if parts[0] != "<not":
                metrics["cache-misses"] = int(parts[0].replace(',', ''))
        elif "instructions" in line:
            metrics["instructions"] = int(line.split()[0].replace(',', ''))
        elif "cycles" in line:
            metrics["cycles"] = int(line.split()[0].replace(',', ''))

    return metrics

def run_benchmarks():
    data = []
    print(f"{'Mode':<12} {'Size':<6} {'Time(s)':<10} {'GFLOPS':<10} {'CacheMiss':<12}")
    print("-" * 60)

    for size in SIZES:
        for mode in MODES:
            # Construct perf command
            cmd = [
                "perf", "stat",
                "-e", "cache-misses,cycles,instructions",
                f"./{EXE}", str(size), mode
            ]

            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error in {mode} {size}")
                continue

            metrics = parse_perf_output(result.stderr)

            # Calculate GFLOPS: (2 * N^3) / Time / 10^9
            gflops = (2 * (size**3)) / metrics["seconds"] / 1e9

            record = {
                "Size": size,
                "Mode": mode,
                "Time": metrics["seconds"],
                "GFLOPS": gflops,
                "CacheMisses": metrics["cache-misses"],
                "Instructions": metrics["instructions"],
                "Cycles": metrics["cycles"],
                "IPC": metrics["instructions"] / metrics["cycles"] if metrics["cycles"] > 0 else 0
            }
            data.append(record)

            print(f"{mode:<12} {size:<6} {metrics['seconds']:<10.4f} {gflops:<10.2f} {metrics['cache-misses']:<12}")

    return pd.DataFrame(data)

def generate_plots(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'basic': 'red', 'parallel': 'orange', 'vectorized': 'green'}
    markers = {'basic': 'o', 'parallel': 's', 'vectorized': '^'}

    def plot_metric(metric_col, ylabel, title, filename, log_y=False):
        plt.figure(figsize=(10, 6))
        for mode in MODES:
            subset = df[df["Mode"] == mode]
            if subset.empty: continue
            plt.plot(subset["Size"], subset[metric_col],
                     label=mode.capitalize(), color=colors[mode],
                     marker=markers[mode], linewidth=2)

        plt.xlabel("Matrix Size (N)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        if log_y: plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig(filename)
        plt.close()

    # 1. Execution Time (Log Scale)
    plot_metric("Time", "Time (seconds)", "Execution Time (Lower is Better)",
                "results/plot_time.png", log_y=True)
    plot_metric("Time", "Time (seconds)", "Execution Time (Lower is Better)",
                "results/plot_time.pdf", log_y=True)

    # 2. GFLOPS (Linear Scale)
    plot_metric("GFLOPS", "GigaFLOPS", "Computational Throughput (Higher is Better)",
                "results/plot_gflops.png")
    plot_metric("GFLOPS", "GigaFLOPS", "Computational Throughput (Higher is Better)",
                "results/plot_gflops.pdf")

    # 3. Cache Misses (Log Scale)
    plot_metric("CacheMisses", "L3 Cache Misses", "Memory Inefficiency (Lower is Better)",
                "results/plot_cache_misses.png", log_y=True)
    plot_metric("CacheMisses", "L3 Cache Misses", "Memory Inefficiency (Lower is Better)",
                "results/plot_cache_misses.pdf", log_y=True)

    # 4. Speedup Calculation
    plt.figure(figsize=(10, 6))
    basic_df = df[df["Mode"] == "basic"].set_index("Size")["Time"]

    for mode in ["parallel", "vectorized"]:
        subset = df[df["Mode"] == mode].set_index("Size")
        common_sizes = basic_df.index.intersection(subset.index)
        if common_sizes.empty: continue

        speedup = basic_df.loc[common_sizes] / subset.loc[common_sizes]["Time"]
        plt.plot(common_sizes, speedup, label=f"{mode.capitalize()} Speedup",
                 marker=markers[mode], color=colors[mode])

    plt.xlabel("Matrix Size (N)")
    plt.ylabel("Speedup Factor (x times Basic)")
    plt.title("Speedup Relative to Basic Implementation")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/plot_speedup.png")
    plt.savefig("results/plot_speedup.pdf")
    plt.close()

    print("All 4 graphs generated successfully.")

if __name__ == "__main__":
    compile_code()
    df = run_benchmarks()
    df.to_csv(CSV_FILE, index=False)
    generate_plots(df)
