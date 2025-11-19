#ifndef MATRIX_C_H
#define MATRIX_C_H

/**
 * Performs matrix multiplication C = A * B
 * @param a First matrix (n x n)
 * @param b Second matrix (n x n)
 * @param c Result matrix (n x n)
 * @param n Size of matrices
 */
void matrix_multiply(double** a, double** b, double** c, int n);

/**
 * Allocates memory for an n x n matrix
 * @param n Size of the matrix
 * @return Pointer to allocated matrix
 */
double** create_matrix(int n);

/**
 * Frees memory allocated for a matrix
 * @param matrix Pointer to the matrix
 * @param n Size of the matrix
 */
void free_matrix(double** matrix, int n);

/**
 * Initializes a matrix with random values
 * @param matrix Pointer to the matrix
 * @param n Size of the matrix
 * @param seed Random seed for reproducibility
 */
void initialize_random_matrix(double** matrix, int n, unsigned int seed);

/**
 * Initializes a matrix with zeros
 * @param matrix Pointer to the matrix
 * @param n Size of the matrix
 */
void initialize_zero_matrix(double** matrix, int n);

#endif // MATRIX_C_H
