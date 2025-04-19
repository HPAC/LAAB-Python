#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <cblas.h>

typedef void (*sgemm_t)(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE,
                        const enum CBLAS_TRANSPOSE, const int, const int, const int,
                        const float, const float*, const int,
                        const float*, const int, const float,
                        float*, const int);

int main() {
    void* handle = dlopen("libopenblas.so", RTLD_LAZY);
    sgemm_t real_sgemm = (sgemm_t)dlsym(handle, "cblas_sgemm");

    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6] = {1, 0, 0, 1, 0, 0};
    float C[4] = {0};

    real_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
               2, 2, 3, 1.0, A, 3, B, 2, 0.0, C, 2);

    printf("C[0] = %f\n", C[0]);
    return 0;
}
