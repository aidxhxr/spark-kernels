// Shared helpers for every kernel in spark-kernels.
// Primary target: NVIDIA RTX 5090 (GB202), compute capability 12.0 (sm_120), CUDA 13.x.
// Secondary target: NVIDIA GB10 (DGX Spark), compute capability 12.1 (sm_121).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace spark {

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
#define SPARK_CUDA_CHECK(expr)                                                                \
    do {                                                                                      \
        cudaError_t _err = (expr);                                                            \
        if (_err != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                                     \
    } while (0)

// Check the most recent kernel launch (call right after <<<>>>).
#define SPARK_CHECK_LAUNCH() SPARK_CUDA_CHECK(cudaGetLastError())

#define SPARK_REQUIRE(cond, msg)                                    \
    do {                                                            \
        if (!(cond)) throw std::invalid_argument(std::string(msg)); \
    } while (0)

// ---------------------------------------------------------------------------
// Small host/device utilities
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) {
    return (a + b - 1) / b;
}
__host__ __device__ __forceinline__ constexpr int64_t cdiv64(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// ---------------------------------------------------------------------------
// Warp-level reductions (all lanes receive the result)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(kFullMask, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(kFullMask, v, offset));
    }
    return v;
}

// Block-level reductions. `smem` must hold at least 32 floats. All threads in
// the block receive the result. Requires blockDim.x to be a multiple of 32.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nwarps) ? smem[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();  // allow smem reuse by the caller
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < nwarps) ? smem[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    if (threadIdx.x == 0) smem[0] = v;
    __syncthreads();
    v = smem[0];
    __syncthreads();
    return v;
}

// ---------------------------------------------------------------------------
// bf16 <-> f32 conversion helpers (uniform names for both element types)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float to_f32(float v) {
    return v;
}
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Vector-of-8 loads for bf16 (16 bytes) and vector-of-4 for f32 (16 bytes).
// Both are "one 128-bit transaction per thread" — the ideal width on Blackwell sm_12x.
struct __align__(16) bf16x8 {
    __nv_bfloat162 h[4];
};
struct __align__(16) f32x4 {
    float v[4];
};

// ---------------------------------------------------------------------------
// cp.async (Ampere+; available on sm_120 / sm_121). 16-byte global->shared copy that
// bypasses registers. Used by the double-buffered GEMM pipelines.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
// The "memory" clobber keeps the compiler from hoisting shared-memory reads above the wait.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

}  // namespace spark
