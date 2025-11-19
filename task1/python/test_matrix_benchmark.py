"""Enhanced benchmark tests with memory profiling for matrix multiplication."""

import pytest
import psutil
import os
import csv
from pathlib import Path
from matrix_multiplier import multiply, create_random_matrix


# Matrix sizes to test
MATRIX_SIZES = [128, 256, 512, 1024]
# MATRIX_SIZES = [16, 32, 64, 128]

# CSV results storage
csv_results = []


@pytest.fixture(params=MATRIX_SIZES)
def matrix_data(request):
    """Fixture that provides matrices of different sizes with memory tracking."""
    n = request.param

    # Get memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    a = create_random_matrix(n, seed=42)
    b = create_random_matrix(n, seed=43)

    # Get memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before

    print(f"\n[Setup] Matrix size: {n}x{n}, Memory used: {mem_used:.2f} MB")

    return a, b, n, mem_after


def test_matrix_multiplication(benchmark, matrix_data):
    """Benchmark matrix multiplication with memory tracking."""
    a, b, n, mem_used = matrix_data

    # Get memory info
    process = psutil.Process(os.getpid())

    # Run the benchmark
    result = benchmark.pedantic(
        multiply,
        args=(a, b),
        iterations=3,
        rounds=5,
        warmup_rounds=2
    )

    mem_after = process.memory_info().rss / 1024 / 1024

    print(f"[Test] Matrix size: {n}x{n}, Memory: {mem_after:.2f} MB")

    # Store results for CSV export
    stats = benchmark.stats.stats
    csv_results.append({
        'language': 'Python',
        'matrix_size': n,
        'mean_time_ms': stats.mean * 1000,  # Convert to ms
        'min_time_ms': stats.min * 1000,
        'max_time_ms': stats.max * 1000,
        'stddev_ms': stats.stddev * 1000,
        'median_time_ms': stats.median * 1000,
        'memory_mb': mem_after,
        'iterations': 3,  # Fixed value from pedantic call
        'rounds': 5       # Fixed value from pedantic call
    })

    # Verify the result is correct size
    assert len(result) == n
    assert len(result[0]) == n


def test_matrix_correctness():
    """Verify matrix multiplication correctness with small example."""
    # 2x2 test case
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    c = multiply(a, b)

    # Expected: [[19, 22], [43, 50]]
    assert c[0][0] == 19
    assert c[0][1] == 22
    assert c[1][0] == 43
    assert c[1][1] == 50

    print("Correctness test passed!")


@pytest.fixture(scope="session", autouse=True)
def export_csv(request):
    """Export results to CSV after all tests complete."""
    def finalizer():
        if csv_results:
            results_dir = Path('../results')
            results_dir.mkdir(exist_ok=True)

            csv_path = results_dir / 'python_results.csv'

            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'language', 'matrix_size', 'mean_time_ms', 'min_time_ms',
                    'max_time_ms', 'stddev_ms', 'median_time_ms',
                    'memory_mb', 'iterations', 'rounds'
                ])
                writer.writeheader()
                writer.writerows(csv_results)

            print(f"\n✅ CSV results exported to: {csv_path.absolute()}")

    request.addfinalizer(finalizer)
