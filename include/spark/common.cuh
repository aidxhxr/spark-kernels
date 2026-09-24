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

// SM count of the current device; the grid-stride kernels size their grids from it. Cached
// after the first call. If the attribute query reports nothing, assume the primary target.
constexpr int kFallbackSMs = 170;  // RTX 5090 (the GB10 has 48)
inline int num_sms() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        if (sms <= 0) sms = kFallbackSMs;
    }
    return sms;
}

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// True if `p` can be read or written as one 128-bit vector. cudaMalloc returns 256-byte
// aligned buffers, but a pointer into the middle of one (a PyTorch storage offset, say) may
// not be; the vectorized kernels must refuse those instead of faulting with a misaligned
// address, which poisons the CUDA context for the rest of the process.
inline bool is_aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
}

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
// Same, but with `valid == false` nothing is read and the 16 smem bytes are zero-filled
// (src-size 0): the way a tile row past the end of a matrix is loaded without a branch.
__device__ __forceinline__ void cp_async_16_zfill(void* smem_ptr, const void* gmem_ptr,
                                                  bool valid) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem_ptr),
                 "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
// The "memory" clobber keeps the compiler from hoisting shared-memory reads above the wait.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// Raw tensor-core primitives (sm_80+; used by the hgemm mma.sync variants).
// ---------------------------------------------------------------------------
// ldmatrix: four 8x8 b16 matrices from shared memory into one register per matrix per lane.
// Lanes 8i..8i+7 supply the row addresses of matrix i (16 bytes per row).
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// Same, transposing each 8x8 matrix on the way: turns a k-major (row = k) tile of B into the
// "col" operand layout mma.sync wants.
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* smem_ptr) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
// D[16x8] (+)= A[16x16] * B[16x8], bf16 inputs, fp32 accumulate. Fragment layouts are the
// PTX ISA ones for m16n8k16: a[4] = (rows g / g+8) x (k 0-7 / 8-15), b[2] = k 0-7 / 8-15,
// d[4] = (row g, cols 2c..2c+1), (row g+8, same cols) with g = lane/4, c = lane%4.
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const unsigned (&a)[4],
                                               const unsigned (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace spark
