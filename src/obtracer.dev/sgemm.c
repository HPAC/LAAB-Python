#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>
// #include "mkl.h"

#ifndef REP
#define REP 3
#endif

#define SCRUB_SIZE (50 * 1024 * 1024) // ~200 MB

void cache_scrub()
{
    void *scrub = malloc(SCRUB_SIZE * sizeof(float));
    memset(scrub, 0, SCRUB_SIZE * sizeof(float));
    free(scrub);
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        printf("Usage: %s m k n\n", argv[0]);
        return -1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));

    // seed the random number generator with the current time
    srand48((unsigned)time(NULL));
    for (int i = 0; i < m * k; i++) A[i] = (float)drand48();
    for (int i = 0; i < k * n; i++) B[i] = (float)drand48();

    for (int it = 0; it < REP; it++) {
        for (int i = 0; i < m * n; i++) C[i] = 0.0f;

        cache_scrub();

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}

