// RMSNorm kernels: out[r, c] = x[r, c] * rsqrt(mean_c(x[r, :]^2) + eps) * w[c]
//
// Optimization ladder (see docs/design/rmsnorm.md):
//   variant 0: one thread per row            (naive: uncoalesced, serial reduction)
//   variant 1: one warp per row              (coalesced, shuffle reduction)
//   variant 2: one warp per row, 128-bit loads (4 f32 / 8 bf16 per lane per transaction)
//   variant 3: one 256-thread block per row  (for very wide rows)
// Plus the fused residual-add + RMSNorm used in Llama/Qwen decoder blocks.
//
// All math accumulates in fp32 regardless of the storage type T.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

// ---------------------------------------------------------------------------
// 128-bit vector traits: f32 -> 4 lanes, bf16 -> 8 lanes.
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;
    __device__ __forceinline__ static void load(const float* p, float (&out)[4]) {
        const f32x4 v = *reinterpret_cast<const f32x4*>(p);
        out[0] = v.v[0];
        out[1] = v.v[1];
        out[2] = v.v[2];
        out[3] = v.v[3];
    }
    __device__ __forceinline__ static void store(float* p, const float (&in)[4]) {
        f32x4 v;
        v.v[0] = in[0];
        v.v[1] = in[1];
        v.v[2] = in[2];
        v.v[3] = in[3];
        *reinterpret_cast<f32x4*>(p) = v;
    }
};

template <>
struct VecTraits<__nv_bfloat16> {
    static constexpr int kWidth = 8;
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float (&out)[8]) {
        const bf16x8 v = *reinterpret_cast<const bf16x8*>(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v.h[i]);
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
        }
    }
    __device__ __forceinline__ static void store(__nv_bfloat16* p, const float (&in)[8]) {
        bf16x8 v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v.h[i] = __float22bfloat162_rn(make_float2(in[2 * i], in[2 * i + 1]));
        }
        *reinterpret_cast<bf16x8*>(p) = v;
    }
};

// ---------------------------------------------------------------------------
// Variant 0: one thread per row.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_naive_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                     T* __restrict__ out, int rows, int cols, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        const float v = to_f32(xr[c]);
        sumsq += v * v;
    }
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    for (int c = 0; c < cols; ++c) {
        orow[c] = from_f32<T>(to_f32(xr[c]) * inv * to_f32(w[c]));
    }
}

// ---------------------------------------------------------------------------
// Variant 1: one warp per row, scalar loads strided by lane.
// kWarpsPerBlock rows are processed per block.
// ---------------------------------------------------------------------------
template <typename T, int kWarpsPerBlock>
__global__ void rmsnorm_warp_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                    T* __restrict__ out, int rows, int cols, float eps) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = lane; c < cols; c += kWarpSize) {
        const float v = to_f32(xr[c]);
        sumsq += v * v;
    }
    sumsq = warp_reduce_sum(sumsq);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    for (int c = lane; c < cols; c += kWarpSize) {
        orow[c] = from_f32<T>(to_f32(xr[c]) * inv * to_f32(w[c]));
    }
}

// ---------------------------------------------------------------------------
// Variant 2: one warp per row, 128-bit vectorized loads/stores.
// Pass 1 reads the row once from DRAM to compute the sum of squares; pass 2
// re-reads it (row <= 16 KB, served from the 24 MB L2) and writes the output.
// Requires cols % kWidth == 0.
// ---------------------------------------------------------------------------
template <typename T, int kWarpsPerBlock>
__global__ void rmsnorm_warp_vec_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                        T* __restrict__ out, int rows, int cols, float eps) {
    using VT = VecTraits<T>;
    constexpr int kW = VT::kWidth;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float v[kW];
        VT::load(xr + c, v);
#pragma unroll
        for (int i = 0; i < kW; ++i) sumsq += v[i] * v[i];
    }
    sumsq = warp_reduce_sum(sumsq);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);

    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float v[kW];
        float wv[kW];
        float o[kW];
        VT::load(xr + c, v);
        VT::load(w + c, wv);
#pragma unroll
        for (int i = 0; i < kW; ++i) o[i] = v[i] * inv * wv[i];
        VT::store(orow + c, o);
    }
}

// ---------------------------------------------------------------------------
// Variant 3: one block per row, block-wide reduction. Consecutive threads read
// consecutive columns so every access is coalesced.
// ---------------------------------------------------------------------------
template <typename T, int kBlockSize>
__global__ void rmsnorm_block_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                     T* __restrict__ out, int rows, int cols, float eps) {
    __shared__ float smem[kWarpSize];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
        const float v = to_f32(xr[c]);
        sumsq += v * v;
    }
    sumsq = block_reduce_sum(sumsq, smem);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
        orow[c] = from_f32<T>(to_f32(xr[c]) * inv * to_f32(w[c]));
    }
}

// ---------------------------------------------------------------------------
// Fused residual-add + RMSNorm (bf16), warp-per-row, 128-bit vectors.
//   resid[r, :] += x[r, :];   out[r, :] = rmsnorm(resid[r, :]) * w
// Pass 1 reads x and resid, writes the updated resid, accumulates sum of squares.
// Pass 2 re-reads the fresh resid (L2-resident) and writes out.
// ---------------------------------------------------------------------------
template <int kWarpsPerBlock>
__global__ void add_rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                        __nv_bfloat16* __restrict__ resid,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ out, int rows, int cols,
                                        float eps) {
    using VT = VecTraits<__nv_bfloat16>;
    constexpr int kW = VT::kWidth;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const int64_t off = static_cast<int64_t>(row) * cols;
    const __nv_bfloat16* xr = x + off;
    __nv_bfloat16* rr = resid + off;
    __nv_bfloat16* orow = out + off;

    float sumsq = 0.0f;
    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float xv[kW];
        float rv[kW];
        float sv[kW];
        VT::load(xr + c, xv);
        VT::load(rr + c, rv);
#pragma unroll
        for (int i = 0; i < kW; ++i) {
            // Round the residual sum to bf16 first so the stored residual and the
            // value we normalize agree exactly (matches PyTorch's semantics of
            // `resid = resid + x` in bf16 followed by a norm over `resid`).
            const float s = __bfloat162float(__float2bfloat16(xv[i] + rv[i]));
            sv[i] = s;
            sumsq += s * s;
        }
        VT::store(rr + c, sv);
    }
    sumsq = warp_reduce_sum(sumsq);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);

    __syncwarp();
    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float rv[kW];
        float wv[kW];
        float o[kW];
        VT::load(rr + c, rv);
        VT::load(w + c, wv);
#pragma unroll
        for (int i = 0; i < kW; ++i) o[i] = rv[i] * inv * wv[i];
        VT::store(orow + c, o);
    }
}

// ---------------------------------------------------------------------------
// Host dispatch shared by both element types.
// ---------------------------------------------------------------------------
template <typename T>
void rmsnorm_dispatch(const T* x, const T* w, T* out, int rows, int cols, float eps, int variant,
                      cudaStream_t stream) {
    SPARK_REQUIRE(rows >= 0 && cols > 0, "rmsnorm: rows must be >= 0 and cols > 0");
    SPARK_REQUIRE(variant >= 0 && variant < 4, "rmsnorm: variant must be in [0, 3]");
    SPARK_REQUIRE(x != nullptr && w != nullptr && out != nullptr, "rmsnorm: null pointer");
    if (rows == 0) return;

    constexpr int kWarps = 8;  // 256 threads, 8 rows per block for variants 1/2
    switch (variant) {
        case 0: {
            const int block = 256;
            const int grid = cdiv(rows, block);
            rmsnorm_naive_kernel<T><<<grid, block, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        case 1: {
            const int grid = cdiv(rows, kWarps);
            rmsnorm_warp_kernel<T, kWarps>
                <<<grid, kWarps * kWarpSize, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        case 2: {
            SPARK_REQUIRE(cols % VecTraits<T>::kWidth == 0,
                          "rmsnorm variant 2: cols must be a multiple of 4 (f32) / 8 (bf16)");
            const int grid = cdiv(rows, kWarps);
            rmsnorm_warp_vec_kernel<T, kWarps>
                <<<grid, kWarps * kWarpSize, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        case 3: {
            constexpr int kBlock = 256;
            rmsnorm_block_kernel<T, kBlock>
                <<<rows, kBlock, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int rmsnorm_num_variants() {
    return 4;
}

void rmsnorm_f32(const float* x, const float* w, float* out, int rows, int cols, float eps,
                 int variant, cudaStream_t stream) {
    rmsnorm_dispatch<float>(x, w, out, rows, cols, eps, variant, stream);
}

void rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int rows,
                  int cols, float eps, int variant, cudaStream_t stream) {
    rmsnorm_dispatch<__nv_bfloat16>(x, w, out, rows, cols, eps, variant, stream);
}

void add_rmsnorm_bf16(const __nv_bfloat16* x, __nv_bfloat16* resid, const __nv_bfloat16* w,
                      __nv_bfloat16* out, int rows, int cols, float eps, cudaStream_t stream) {
    SPARK_REQUIRE(rows >= 0 && cols > 0, "add_rmsnorm: rows must be >= 0 and cols > 0");
    SPARK_REQUIRE(cols % 8 == 0, "add_rmsnorm_bf16: cols must be a multiple of 8");
    SPARK_REQUIRE(x != nullptr && resid != nullptr && w != nullptr && out != nullptr,
                  "add_rmsnorm: null pointer");
    if (rows == 0) return;
    constexpr int kWarps = 8;
    const int grid = cdiv(rows, kWarps);
    add_rmsnorm_bf16_kernel<kWarps>
        <<<grid, kWarps * kWarpSize, 0, stream>>>(x, resid, w, out, rows, cols, eps);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
