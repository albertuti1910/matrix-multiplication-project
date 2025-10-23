CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c99
TARGET = matrix_benchmark
SOURCES = matrix_benchmark.c matrix_c.c
HEADERS = matrix_c.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o perf.data perf.data.old

run: $(TARGET)
	./$(TARGET)

perf: $(TARGET)
	perf stat -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses ./$(TARGET)

perf-record: $(TARGET)
	perf record -g ./$(TARGET)
	perf report

.PHONY: all clean run perf perf-record
