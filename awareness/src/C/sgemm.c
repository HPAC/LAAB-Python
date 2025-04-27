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
    if (argc < 4) {
        printf("Usage: %s m k n\n", argv[0]);
        return -1;
    }

    int REP = 3; // default repetitions

    char* rep_env = getenv("REP");
    if (rep_env != NULL) {
        REP = atoi(rep_env);
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);
    float alpha = 1.0;
    float beta = 0.0;

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
        
	struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
            clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / (double)BILLION;

        //printf("Time for cblas_sgemm: %.6f seconds\n", elapsed);
        printf("[LAAB] C | sgemm | optimized=%.3f s\n", elapsed);
    
    }

    free(A);
    free(B);
    free(C);

    return 0;
}




