package matrix.benchmark;

import org.openjdk.jmh.annotations.*;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"-Xms2g", "-Xmx4g"})
public class MatrixBenchmark {

    @Param({"128", "256", "512", "1024"})
    private int matrixSize;

    private double[][] a;
    private double[][] b;

    @Setup(Level.Trial)
    public void setUp() {
        // Print memory info at start
        Runtime runtime = Runtime.getRuntime();
        long beforeMemory = runtime.totalMemory() - runtime.freeMemory();

        a = MatrixMultiplier.createRandomMatrix(matrixSize, 42L);
        b = MatrixMultiplier.createRandomMatrix(matrixSize, 43L);

        runtime.gc(); // Suggest garbage collection
        long afterMemory = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = afterMemory - beforeMemory;

        System.out.println(String.format(
            "[Setup] Matrix size: %d, Estimated memory: %.2f MB",
            matrixSize, memoryUsed / (1024.0 * 1024.0)
        ));
    }

    @Benchmark
    public double[][] benchmarkMatrixMultiplication() {
        return MatrixMultiplier.multiply(a, b);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        Runtime runtime = Runtime.getRuntime();
        System.out.println(String.format(
            "[Teardown] Matrix size: %d, Total memory: %.2f MB, Free: %.2f MB, Used: %.2f MB",
            matrixSize,
            runtime.totalMemory() / (1024.0 * 1024.0),
            runtime.freeMemory() / (1024.0 * 1024.0),
            (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0)
        ));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
