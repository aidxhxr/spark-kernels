// Row-wise softmax: naive three-pass -> warp-per-row online softmax -> block-per-row online softmax.
// See docs/design/softmax.md for the optimization story.
//
// All variants compute in fp32 regardless of the storage type T (float or __nv_bfloat16).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

// ---------------------------------------------------------------------------
// Online-softmax state merge.
//
// Two partial states (m1, s1) and (m2, s2), where m is the running max and
// s = sum_i exp(x_i - m) over the elements seen so far, merge into
//     m  = max(m1, m2)
//     s  = s1 * exp(m1 - m) + s2 * exp(m2 - m)
// (Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018).
// The merge is associative and commutative, so it can be applied element by
// element, across a warp with shuffles, and across warps through shared memory.
//
// Edge case: if both maxes are -inf (nothing seen yet, or all -inf inputs)
// then m1 - m = -inf - (-inf) = NaN. We define the merged sum as 0 there.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void online_merge(float& m, float& s, float m2, float s2) {
    const float m_new = fmaxf(m, m2);
    if (m_new == -INFINITY) {
        m = -INFINITY;
        s = 0.0f;
        return;
    }
    s = s * __expf(m - m_new) + s2 * __expf(m2 - m_new);
    m = m_new;
}

// Butterfly reduction of (m, s) pairs across a warp. Every lane ends with the
// full-warp state (the merge is commutative, so all lanes agree bit-for-bit).
__device__ __forceinline__ void warp_reduce_online(float& m, float& s) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const float m2 = __shfl_xor_sync(kFullMask, m, offset);
        const float s2 = __shfl_xor_sync(kFullMask, s, offset);
        online_merge(m, s, m2, s2);
    }
}

// ---------------------------------------------------------------------------
// 16-byte vector load/store: 4 x f32 or 8 x bf16 per thread per transaction.
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;
    __device__ __forceinline__ static void load(const float* p, float* f) {
        const float4 v = *reinterpret_cast<const float4*>(p);
        f[0] = v.x;
        f[1] = v.y;
        f[2] = v.z;
        f[3] = v.w;
    }
    __device__ __forceinline__ static void store(float* p, const float* f) {
        *reinterpret_cast<float4*>(p) = make_float4(f[0], f[1], f[2], f[3]);
    }
};

template <>
struct VecTraits<__nv_bfloat16> {
    static constexpr int kWidth = 8;
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float* f) {
        const bf16x8 v = *reinterpret_cast<const bf16x8*>(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f2 = __bfloat1622float2(v.h[i]);
            f[2 * i] = f2.x;
            f[2 * i + 1] = f2.y;
        }
    }
    __device__ __forceinline__ static void store(__nv_bfloat16* p, const float* f) {
        bf16x8 v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v.h[i] = __float22bfloat162_rn(make_float2(f[2 * i], f[2 * i + 1]));
        }
        *reinterpret_cast<bf16x8*>(p) = v;
    }
};

// ---------------------------------------------------------------------------
// Variant 0: naive three-pass softmax, one thread per row.
// Pass 1 max, pass 2 sum of exp, pass 3 normalize. Three full reads of the row
// and adjacent threads touch addresses `cols` elements apart: fully uncoalesced.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void softmax_naive_kernel(const T* __restrict__ x, T* __restrict__ out, int rows,
                                     int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const T* xr = x + static_cast<size_t>(row) * cols;
    T* orow = out + static_cast<size_t>(row) * cols;

    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, to_f32(xr[c]));
    float s = 0.0f;
    for (int c = 0; c < cols; ++c) s += __expf(to_f32(xr[c]) - m);
    const float inv = 1.0f / s;
    for (int c = 0; c < cols; ++c) orow[c] = from_f32<T>(__expf(to_f32(xr[c]) - m) * inv);
}

// ---------------------------------------------------------------------------
// Per-thread accumulation of the online state over a strided slice of a row.
// kVec: 16-byte vector loads (cols % kWidth == 0 and 16-byte aligned pointers).
// `stride` is the number of threads cooperating on this row (32 or blockDim.x).
// ---------------------------------------------------------------------------
template <typename T, bool kVec>
__device__ __forceinline__ void accumulate_online(const T* __restrict__ xr, int cols, int tid,
                                                  int stride, float& m, float& s) {
    m = -INFINITY;
    s = 0.0f;
    if constexpr (kVec) {
        constexpr int VEC = VecTraits<T>::kWidth;
        const int nvec = cols / VEC;
        for (int v = tid; v < nvec; v += stride) {
            float f[VEC];
            VecTraits<T>::load(xr + v * VEC, f);
            // Local (max, sum) over the VEC elements, then one merge into the running state.
            float lm = f[0];
#pragma unroll
            for (int i = 1; i < VEC; ++i) lm = fmaxf(lm, f[i]);
            float ls = 0.0f;
#pragma unroll
            for (int i = 0; i < VEC; ++i) ls += __expf(f[i] - lm);
            online_merge(m, s, lm, ls);
        }
    } else {
        for (int c = tid; c < cols; c += stride) {
            online_merge(m, s, to_f32(xr[c]), 1.0f);
        }
    }
}

// Second pass: out = exp(x - m) * inv over the same strided slice.
template <typename T, bool kVec>
__device__ __forceinline__ void write_normalized(const T* __restrict__ xr, T* __restrict__ orow,
                                                 int cols, int tid, int stride, float m,
                                                 float inv) {
    if constexpr (kVec) {
        constexpr int VEC = VecTraits<T>::kWidth;
        const int nvec = cols / VEC;
        for (int v = tid; v < nvec; v += stride) {
            float f[VEC];
            VecTraits<T>::load(xr + v * VEC, f);
#pragma unroll
            for (int i = 0; i < VEC; ++i) f[i] = __expf(f[i] - m) * inv;
            VecTraits<T>::store(orow + v * VEC, f);
        }
    } else {
        for (int c = tid; c < cols; c += stride) {
            orow[c] = from_f32<T>(__expf(to_f32(xr[c]) - m) * inv);
        }
    }
}

// ---------------------------------------------------------------------------
// Variant 1: one warp per row, online softmax.
// Pass 1 (single sweep) builds (m, s) per lane, warp-reduces, pass 2 writes.
// The row is read twice, but the second read hits L2 (rows are <= 64 KB).
// ---------------------------------------------------------------------------
constexpr int kWarpsPerBlock = 4;

template <typename T, bool kVec>
__global__ void softmax_warp_kernel(const T* __restrict__ x, T* __restrict__ out, int rows,
                                    int cols) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;  // uniform across the warp: no divergent shuffles below
    const T* xr = x + static_cast<size_t>(row) * cols;
    T* orow = out + static_cast<size_t>(row) * cols;

    float m, s;
    accumulate_online<T, kVec>(xr, cols, lane, kWarpSize, m, s);
    warp_reduce_online(m, s);
    const float inv = 1.0f / s;
    write_normalized<T, kVec>(xr, orow, cols, lane, kWarpSize, m, inv);
}

// ---------------------------------------------------------------------------
// Variant 2: one block (256 threads) per row, online softmax.
// For long rows a single warp cannot keep enough loads in flight; 8 warps share
// the row and merge their (m, s) states through shared memory.
// ---------------------------------------------------------------------------
constexpr int kBlockThreads = 256;

template <typename T, bool kVec>
__global__ void softmax_block_kernel(const T* __restrict__ x, T* __restrict__ out, int rows,
                                     int cols) {
    __shared__ float sm_m[kWarpSize];
    __shared__ float sm_s[kWarpSize];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const T* xr = x + static_cast<size_t>(row) * cols;
    T* orow = out + static_cast<size_t>(row) * cols;

    float m, s;
    accumulate_online<T, kVec>(xr, cols, threadIdx.x, blockDim.x, m, s);
    warp_reduce_online(m, s);
    if (lane == 0) {
        sm_m[warp] = m;
        sm_s[warp] = s;
    }
    __syncthreads();
    if (warp == 0) {
        float m2 = (lane < nwarps) ? sm_m[lane] : -INFINITY;
        float s2 = (lane < nwarps) ? sm_s[lane] : 0.0f;
        warp_reduce_online(m2, s2);
        if (lane == 0) {
            sm_m[0] = m2;
            sm_s[0] = s2;
        }
    }
    __syncthreads();
    m = sm_m[0];
    const float inv = 1.0f / sm_s[0];
    write_normalized<T, kVec>(xr, orow, cols, threadIdx.x, blockDim.x, m, inv);
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
constexpr int kNumVariants = 3;

template <typename T>
void softmax_impl(const T* x, T* out, int rows, int cols, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(x != nullptr && out != nullptr, "softmax: null pointer");
    SPARK_REQUIRE(rows >= 0 && cols >= 1, "softmax: rows must be >= 0 and cols >= 1");
    SPARK_REQUIRE(variant >= 0 && variant < kNumVariants, "softmax: variant out of range");
    if (rows == 0) return;

    constexpr int VEC = VecTraits<T>::kWidth;
    const bool vec_ok = (cols % VEC == 0) && (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
                        (reinterpret_cast<uintptr_t>(out) % 16 == 0);

    switch (variant) {
        case 0: {
            const int threads = 128;
            const int grid = cdiv(rows, threads);
            softmax_naive_kernel<T><<<grid, threads, 0, stream>>>(x, out, rows, cols);
            break;
        }
        case 1: {
            const int threads = kWarpsPerBlock * kWarpSize;
            const int grid = cdiv(rows, kWarpsPerBlock);
            if (vec_ok) {
                softmax_warp_kernel<T, true><<<grid, threads, 0, stream>>>(x, out, rows, cols);
            } else {
                softmax_warp_kernel<T, false><<<grid, threads, 0, stream>>>(x, out, rows, cols);
            }
            break;
        }
        case 2: {
            const int grid = rows;
            if (vec_ok) {
                softmax_block_kernel<T, true>
                    <<<grid, kBlockThreads, 0, stream>>>(x, out, rows, cols);
            } else {
                softmax_block_kernel<T, false>
                    <<<grid, kBlockThreads, 0, stream>>>(x, out, rows, cols);
            }
            break;
        }
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace

void softmax_f32(const float* x, float* out, int rows, int cols, int variant,
                 cudaStream_t stream) {
    softmax_impl<float>(x, out, rows, cols, variant, stream);
}

void softmax_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int cols, int variant,
                  cudaStream_t stream) {
    softmax_impl<__nv_bfloat16>(x, out, rows, cols, variant, stream);
}

int softmax_num_variants() { return kNumVariants; }

}  // namespace spark
