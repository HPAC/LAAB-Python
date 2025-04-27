#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>


#define BILLION 1000000000L

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s n k\n", argv[0]);
        return -1;
    }

    int REP = 3; // default repetitions

    char* rep_env = getenv("REP");
    if (rep_env != NULL) {
        REP = atoi(rep_env);
    }

    int n = atoi(argv[1]); // rows
    int k = atoi(argv[2]); // columns of B

    // Tridiagonal matrix in CSR format
    int nnz = 3 * n - 2;
    float *values = (float*)malloc(nnz * sizeof(float));
    int *col_idx = (int*)malloc(nnz * sizeof(int));
    int *row_ptr = (int*)malloc((n+1) * sizeof(int));

    float *B = (float*)malloc(n * k * sizeof(float));
    float *ret = (float*)calloc(n * k, sizeof(float));

    srand48((unsigned)time(NULL));

    // Fill tridiagonal A
    int pos = 0;
    for (int i = 0; i < n; i++) {
        row_ptr[i] = pos;
        if (i > 0) {
            values[pos] = (float)drand48(); // lower diag
            col_idx[pos++] = i - 1;
        }
        values[pos] = (float)drand48();     // diag
        col_idx[pos++] = i;
        if (i < n-1) {
            values[pos] = (float)drand48(); // upper diag
            col_idx[pos++] = i + 1;
        }
    }
    row_ptr[n] = pos;

    // Fill random B
    for (int i = 0; i < n * k; i++) {
        B[i] = (float)drand48();
    }

    for (int it = 0; it < REP; it++) {
        memset(ret, 0, n * k * sizeof(float)); // zero output each time

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // PARALLELIZE over rows
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            for (int p = row_ptr[i]; p < row_ptr[i+1]; p++) {
                int col = col_idx[p];
                float val = values[p];
                for (int j = 0; j < k; j++) {
                    ret[i*k + j] += val * B[col*k + j];
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / (double)BILLION;

        printf("[LAAB] C | mp_tridiag | optimized=%.6f s\n", elapsed);
    }

    free(values);
    free(col_idx);
    free(row_ptr);
    free(B);
    free(ret);

    return 0;
}

