// Benchmark + correctness check for the fp32 GEMM ladder against cuBLAS SGEMM.
//
//   ./bench_sgemm                      # default shape sweep, all variants
//   ./bench_sgemm --m=4096 --n=4096 --k=4096 --variant=3 --iters=50
//   ./bench_sgemm --all                # also run the naive variant on the largest shapes
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table
// exit code 1 if any variant disagrees with cuBLAS.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

namespace {

#define CUBLAS_CHECK(expr)                                                                      \
    do {                                                                                        \
        cublasStatus_t _st = (expr);                                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(_st), __FILE__, \
                         __LINE__);                                                             \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

struct Shape {
    int M, N, K;
};

// Row-major C = A * B via column-major cuBLAS: treat the row-major matrices as their
// column-major transposes, so C^T[N,M] = B^T[N,K] * A^T[K,M].
void cublas_sgemm_rowmajor(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N,
                           int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

double tflops_of(const Shape& s, double ms) {
    return 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e12;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace spark::bench;
    Args args(argc, argv);
    print_device_banner();

    const int iters = args.geti("iters", 100);
    const int warmup = args.geti("warmup", 10);
    const int only_variant = args.geti("variant", -1);
    const bool run_all = args.has("all");

    std::vector<Shape> shapes;
    if (args.has("m") || args.has("n") || args.has("k")) {
        const int m = args.geti("m", 1024);
        shapes.push_back({m, args.geti("n", m), args.geti("k", m)});
    } else {
        shapes = {{512, 512, 512},    {1024, 1024, 1024},  {2048, 2048, 2048},
                  {4096, 4096, 4096}, {4096, 4096, 11008}, {4096, 11008, 4096}};
    }

    cudaStream_t stream;
    SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    // Keep cuBLAS on the plain fp32 path (no TF32) so the comparison is like-for-like.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    print_header();
    bool all_ok = true;

    for (const Shape& s : shapes) {
        const size_t nA = static_cast<size_t>(s.M) * s.K;
        const size_t nB = static_cast<size_t>(s.K) * s.N;
        const size_t nC = static_cast<size_t>(s.M) * s.N;

        std::vector<float> hA(nA), hB(nB), hRef(nC), hOut(nC);
        fill_uniform(hA, -1.0f, 1.0f, 1);
        fill_uniform(hB, -1.0f, 1.0f, 2);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
        SPARK_CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

        // ---- cuBLAS reference (result + timing)
        cublas_sgemm_rowmajor(handle, dA, dB, dRef, s.M, s.N, s.K);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));
        const Timing tref =
            time_kernel([&] { cublas_sgemm_rowmajor(handle, dA, dB, dRef, s.M, s.N, s.K); }, stream,
                        warmup, iters);

        const std::string shape_str =
            std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);

        {
            Row r;
            r.kernel = "sgemm_cublas";
            r.dtype = "f32";
            r.variant = -1;
            r.shape = shape_str;
            r.median_ms = tref.median_ms;
            r.tflops = tflops_of(s, tref.median_ms);
            r.ref_ms = tref.median_ms;
            print_row(r);
        }

        // Different summation order than cuBLAS => allow an error that scales with K.
        const double tol = 2e-5 * s.K + 1e-3;

        for (int v = 0; v < spark::sgemm_num_variants(); ++v) {
            if (only_variant >= 0 && v != only_variant) continue;
            // The naive kernel is ~10-20x slower than cuBLAS; skip the huge shapes by default.
            if (v == 0 && !run_all && static_cast<double>(s.M) * s.N * s.K > 2048.0 * 2048 * 2048)
                continue;

            SPARK_CUDA_CHECK(cudaMemsetAsync(dC, 0, nC * sizeof(float), stream));
            spark::sgemm(dA, dB, dC, s.M, s.N, s.K, v, stream);
            SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            SPARK_CUDA_CHECK(
                cudaMemcpy(hOut.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
            const ErrorStats err = compare(hOut.data(), hRef.data(), nC);
            const bool ok = err.max_abs <= tol && std::isfinite(err.max_abs);

            const Timing t = time_kernel(
                [&] { spark::sgemm(dA, dB, dC, s.M, s.N, s.K, v, stream); }, stream, warmup, iters);

            Row r;
            r.kernel = "sgemm";
            r.dtype = "f32";
            r.variant = v;
            r.shape = shape_str;
            r.median_ms = t.median_ms;
            r.tflops = tflops_of(s, t.median_ms);
            r.ref_ms = tref.median_ms;
            r.max_abs_err = err.max_abs;
            r.max_rel_err = err.max_rel;
            r.ok = ok;
            print_row(r);
            if (!ok) {
                std::fprintf(stderr, "  MISMATCH: variant %d shape %s max_abs=%.3e tol=%.3e\n", v,
                             shape_str.c_str(), err.max_abs, tol);
                all_ok = false;
            }
        }

        SPARK_CUDA_CHECK(cudaFree(dA));
        SPARK_CUDA_CHECK(cudaFree(dB));
        SPARK_CUDA_CHECK(cudaFree(dC));
        SPARK_CUDA_CHECK(cudaFree(dRef));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
    if (!all_ok) {
        std::fprintf(stderr, "FAILED: at least one variant disagrees with cuBLAS\n");
        return 1;
    }
    std::fprintf(stderr, "all variants match cuBLAS\n");
    return 0;
}
