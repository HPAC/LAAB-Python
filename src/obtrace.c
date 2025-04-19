#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>
#include <cblas.h>

static pid_t get_tid() {
    return syscall(SYS_gettid);
}

static void print_timestamp(char* buffer, size_t len) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    snprintf(buffer, len, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

// Function pointer to the original cblas_sgemm
static void (*real_cblas_sgemm)(const CBLAS_LAYOUT, const CBLAS_TRANSPOSE,
                                const CBLAS_TRANSPOSE, const int, const int,
                                const int, const float, const float *,
                                const int, const float *, const int,
                                const float, float *, const int) = NULL;

void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc) {
    if (!real_cblas_sgemm) {
        real_cblas_sgemm = dlsym(RTLD_NEXT, "cblas_sgemm");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    
    real_cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    clock_gettime(CLOCK_REALTIME, &end);

    double duration_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                         (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr, "[OBTRACE] %s | timestamp: %s | thread: %d | M=%d N=%d K=%d alpha=%.1f beta=%.1f | duration: %.3f ms\n",
            "cblas_sgemm", timestamp, get_tid(), M, N, K, alpha, beta, duration_ms);
}


static void (*real_sgemm_)(char*, char*, int*, int*, int*,
                           float*, float*, int*, float*, int*,
                           float*, float*, int*) = NULL;

void sgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K,
            float* ALPHA, float* A, int* LDA,
            float* B, int* LDB,
            float* BETA, float* C, int* LDC) {
    if (!real_sgemm_) {
        real_sgemm_ = dlsym(RTLD_NEXT, "sgemm_");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    real_sgemm_(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr,
        "[OBTRACE] sgemm_ | %s | tid: %ld | M=%d N=%d K=%d alpha=%.1f beta=%.1f | %.3f ms\n",
        timestamp, syscall(SYS_gettid), *M, *N, *K, *ALPHA, *BETA, elapsed);
}

// Function pointer to the original cblas_dgemm
static void (*real_cblas_dgemm)(const CBLAS_LAYOUT, const CBLAS_TRANSPOSE,
                                const CBLAS_TRANSPOSE, const int, const int,
                                const int, const double, const double *,
                                const int, const double *, const int,
                                const double, double *, const int) = NULL;


void cblas_dgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) {

    if (!real_cblas_dgemm) {
        real_cblas_dgemm = dlsym(RTLD_NEXT, "cblas_dgemm");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    real_cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    clock_gettime(CLOCK_REALTIME, &end);

    double duration_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                         (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr, "[OBTRACE] %s | timestamp: %s | thread: %d | M=%d N=%d K=%d alpha=%.1f beta=%.1f | duration: %.3f ms\n",
            "cblas_dgemm", timestamp, get_tid(), M, N, K, alpha, beta, duration_ms);
}

static void (*real_dgemm_)(char*, char*, int*, int*, int*,
                           double*, double*, int*, double*, int*,
                           double*, double*, int*) = NULL;

void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K,
            double* ALPHA, double* A, int* LDA,
            double* B, int* LDB,
            double* BETA, double* C, int* LDC) {
    if (!real_dgemm_) {
        real_dgemm_ = dlsym(RTLD_NEXT, "dgemm_");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    real_dgemm_(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr,
        "[OBTRACE] dgemm_ | %s | tid: %ld | M=%d N=%d K=%d alpha=%.1f beta=%.1f | %.3f ms\n",
        timestamp, syscall(SYS_gettid), *M, *N, *K, *ALPHA, *BETA, elapsed);
}

static void (*real_dtrsm_)(char*, char*, char*, char*, int*, int*,
                           double*, double*, int*, double*, int*) = NULL;

void dtrsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG,
            int* M, int* N,
            double* ALPHA, double* A, int* LDA,
            double* B, int* LDB) {
    if (!real_dtrsm_) {
        real_dtrsm_ = dlsym(RTLD_NEXT, "dtrsm_");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    real_dtrsm_(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr,
        "[OBTRACE] dtrsm_ | %s | tid: %ld | M=%d N=%d alpha=%.1f | %.3f ms\n",
        timestamp, syscall(SYS_gettid), *M, *N, *ALPHA, elapsed);
}

static void (*real_dtrsv_)(char*, char*, char*, int*,
                           double*, int*, double*, int*) = NULL;

void dtrsv_(char* UPLO, char* TRANS, char* DIAG, int* N,
            double* A, int* LDA,
            double* X, int* INCX) {
    if (!real_dtrsv_) {
        real_dtrsv_ = dlsym(RTLD_NEXT, "dtrsv_");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    real_dtrsv_(UPLO, TRANS, DIAG, N, A, LDA, X, INCX);
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr,
        "[OBTRACE] dtrsv_ | %s | tid: %ld | N=%d | %.3f ms\n",
        timestamp, syscall(SYS_gettid), *N, elapsed);
}

static void (*real_strsm_)(char*, char*, char*, char*, int*, int*,
                           float*, float*, int*, float*, int*) = NULL;

void strsm_(char* SIDE, char* UPLO, char* TRANSA, char* DIAG,
            int* M, int* N,
            float* ALPHA, float* A, int* LDA,
            float* B, int* LDB) {
    if (!real_strsm_) {
        real_strsm_ = dlsym(RTLD_NEXT, "strsm_");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    real_strsm_(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB);
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr,
        "[OBTRACE] strsm_ | %s | tid: %ld | M=%d N=%d alpha=%.1f | %.3f ms\n",
        timestamp, syscall(SYS_gettid), *M, *N, *ALPHA, elapsed);
}

static void (*real_strsv_)(char*, char*, char*, int*,
                           float*, int*, float*, int*) = NULL;

void strsv_(char* UPLO, char* TRANS, char* DIAG, int* N,
            float* A, int* LDA,
            float* X, int* INCX) {
    if (!real_strsv_) {
        real_strsv_ = dlsym(RTLD_NEXT, "strsv_");
    }

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    real_strsv_(UPLO, TRANS, DIAG, N, A, LDA, X, INCX);
    clock_gettime(CLOCK_REALTIME, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1.0e6;

    char timestamp[64];
    print_timestamp(timestamp, sizeof(timestamp));

    fprintf(stderr,
        "[OBTRACE] strsv_ | %s | tid: %ld | N=%d | %.3f ms\n",
        timestamp, syscall(SYS_gettid), *N, elapsed);
}
