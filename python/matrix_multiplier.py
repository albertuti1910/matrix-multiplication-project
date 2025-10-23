"""Matrix multiplication module with basic O(n^3) algorithm."""

import random
from typing import List


def multiply_naive(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Naive O(n^3) matrix multiplication used for baseline comparison."""
    n = len(a)
    c = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            sum_val = 0.0
            for k in range(n):
                sum_val += a[i][k] * b[k][j]
            c[i][j] = sum_val

    return c


def multiply_blocked(a: List[List[float]], b: List[List[float]], block_size: int = 32) -> List[List[float]]:
    """Blocked (tiled) matrix multiplication with B transposed for better locality.

    This reduces Python-level indexing overhead and improves cache locality.
    Block size tuned for typical L1/L2 caches; 32 or 64 is a good starting point.
    """
    n = len(a)
    # Create result initialized to 0.0
    c = [[0.0 for _ in range(n)] for _ in range(n)]

    # Transpose B to allow sequential access when iterating rows of B
    b_t = [[b[j][i] for j in range(n)] for i in range(n)]

    # Tiled multiplication
    for ii in range(0, n, block_size):
        i_max = min(ii + block_size, n)
        for jj in range(0, n, block_size):
            j_max = min(jj + block_size, n)
            for kk in range(0, n, block_size):
                k_max = min(kk + block_size, n)
                for i in range(ii, i_max):
                    a_i = a[i]
                    c_i = c[i]
                    for j in range(jj, j_max):
                        sum_val = c_i[j]
                        b_t_j = b_t[j]
                        for k in range(kk, k_max):
                            # note: b_t_j[k] == b[k][j]
                            sum_val += a_i[k] * b_t_j[k]
                        c_i[j] = sum_val

    return c


def multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Default multiply entry point — uses the blocked implementation for speed.

    Falls back to naive algorithm for very small matrices.
    """
    n = len(a)
    if n <= 64:
        # small matrices: naive overhead is acceptable
        return multiply_naive(a, b)
    # choose block size heuristically
    block = 32 if n >= 256 else 16
    return multiply_blocked(a, b, block_size=block)


def create_random_matrix(n: int, seed: int = 42) -> List[List[float]]:
    """
    Creates a random matrix of size n x n

    Args:
        n: Size of the matrix
        seed: Random seed for reproducibility

    Returns:
        Random matrix
    """
    random.seed(seed)
    return [[random.random() for _ in range(n)] for _ in range(n)]
