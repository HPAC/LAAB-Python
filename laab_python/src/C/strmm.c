#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
// #include "mkl.h"


#define BILLION 1000000000L
#define SCRUB_SIZE (50 * 1024 * 1024) // ~200 MB

void cache_scrub()
{
    void *scrub = malloc(SCRUB_SIZE * sizeof(float));
    memset(scrub, 0, SCRUB_SIZE * sizeof(float));
    free(scrub);
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: %s m n\n", argv[0]);
        return -1;
    }

    int REP = 3; // default repetitions

    char* rep_env = getenv("REP");
    if (rep_env != NULL) {
        REP = atoi(rep_env);
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    float alpha = 1.0;

    float *A = (float*)malloc(m * m * sizeof(float)); // A must be triangular, square
    float *B = (float*)malloc(m * n * sizeof(float));

    // seed the random number generator with the current time
    srand48((unsigned)time(NULL));
    for (int i = 0; i < m * m; i++) A[i] = (float)drand48();
    for (int i = 0; i < m * n; i++) B[i] = (float)drand48();

    for (int it = 0; it < REP; it++) {
        cache_scrub();

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        cblas_strmm(
            CblasRowMajor,   // Row-major layout
            CblasLeft,       // A multiplies from the left: op(A) * B
            CblasLower,      // A is lower triangular
            CblasNoTrans,    // Don't transpose A
            CblasNonUnit,    // Diagonal elements are not assumed to be 1
            m,               // Number of rows of B
            n,               // Number of columns of B
            alpha,           // Scalar multiplier
            A,               // Triangular matrix A
            m,               // Leading dimension of A
            B,               // Matrix B
            n                // Leading dimension of B
        );

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / (double)BILLION;

        printf("[LAAB] C | mm_trmm | ref_positive=%.3f s\n", elapsed);
    }

    free(A);
    free(B);

    return 0;
}

