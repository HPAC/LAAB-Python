#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

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

    int m = atoi(argv[1]); // rows of A
    int k = atoi(argv[2]); // cols of A, rows of B
    int n = atoi(argv[3]); // cols of B
    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate memory
    float *A = (float*)malloc(m * k * sizeof(float));
    float *B = (float*)malloc(k * n * sizeof(float));
    float *C = (float*)malloc(m * n * sizeof(float));

    // Seed RNG
    srand48((unsigned)time(NULL));
    for (int i = 0; i < m * k; i++) A[i] = (float)drand48();
    for (int i = 0; i < k * n; i++) B[i] = (float)drand48();

    for (int it = 0; it < REP; it++) {
        for (int i = 0; i < m * n; i++) C[i] = 0.0f;

        cache_scrub();

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Loop over columns of B → one SGEMV per column
        for (int j = 0; j < n; j++) {
            // x points to column j of B
            float *x = &B[j * k];   // B is row-major: column j is stride k
            float *y = &C[j * m];   // column j of C

            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        m, k, alpha, A, k,
                        x, 1, beta, y, 1);
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / (double)BILLION;

        printf("[LAAB] C | mm_sgemm | ref_negative=%.6f s\n", elapsed);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
