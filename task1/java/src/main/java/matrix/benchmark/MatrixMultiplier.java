package matrix.benchmark;

public class MatrixMultiplier {

    /**
     * Performs basic matrix multiplication C = A * B
     * @param a First matrix (n x n)
     * @param b Second matrix (n x n)
     * @return Result matrix C (n x n)
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int n = a.length;
        double[][] c = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }

        return c;
    }

    /**
     * Creates a random matrix of size n x n
     * @param n Size of the matrix
     * @param seed Random seed for reproducibility
     * @return Random matrix
     */
    public static double[][] createRandomMatrix(int n, long seed) {
        java.util.Random random = new java.util.Random(seed);
        double[][] matrix = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = random.nextDouble();
            }
        }

        return matrix;
    }
}
