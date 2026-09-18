// Fused SwiGLU gated activation: out[i] = silu(gate[i]) * up[i].
//
// In a Llama/Qwen MLP this sits between the gate/up projections and the down projection.
// PyTorch eager runs it as two kernels (F.silu, then mul) that stream the intermediate
// activation through memory twice; fusing it reads gate and up once and writes out once.
// The kernel is purely memory-bound, so the only levers are (a) bytes moved and (b) how
// efficiently we move them: variant 1 issues one 128-bit load per operand per thread.
//   0: scalar, one element per thread
//   1: 16-byte vectorized (float4 for f32, 8 x bf16 for bf16), grid-stride, in-kernel tail
#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

template <typename T>
__global__ void swiglu_scalar_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                     T* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = to_f32(gate[i]);
        const float u = to_f32(up[i]);
        out[i] = from_f32<T>(silu_f(g) * u);
    }
}

// f32: 4 elements per thread via f32x4 (one 128-bit load per operand).
__global__ void swiglu_vec_f32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                      float* __restrict__ out, int64_t n) {
    constexpr int VEC = 4;
    const int64_t n_vec = n / VEC;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const f32x4* gate4 = reinterpret_cast<const f32x4*>(gate);
    const f32x4* up4 = reinterpret_cast<const f32x4*>(up);
    f32x4* out4 = reinterpret_cast<f32x4*>(out);

    for (int64_t i = tid; i < n_vec; i += stride) {
        const f32x4 g = gate4[i];
        const f32x4 u = up4[i];
        f32x4 o;
#pragma unroll
        for (int k = 0; k < VEC; ++k) o.v[k] = silu_f(g.v[k]) * u.v[k];
        out4[i] = o;
    }

    // Scalar tail (n % 4 elements), handled by the first few threads of the grid.
    const int64_t tail_start = n_vec * VEC;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        out[i] = silu_f(gate[i]) * up[i];
    }
}

// bf16: 8 elements per thread via bf16x8 (4 x __nv_bfloat162 = 16 bytes). Math in fp32.
__global__ void swiglu_vec_bf16_kernel(const __nv_bfloat16* __restrict__ gate,
                                       const __nv_bfloat16* __restrict__ up,
                                       __nv_bfloat16* __restrict__ out, int64_t n) {
    constexpr int VEC = 8;
    const int64_t n_vec = n / VEC;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const bf16x8* gate8 = reinterpret_cast<const bf16x8*>(gate);
    const bf16x8* up8 = reinterpret_cast<const bf16x8*>(up);
    bf16x8* out8 = reinterpret_cast<bf16x8*>(out);

    for (int64_t i = tid; i < n_vec; i += stride) {
        const bf16x8 g = gate8[i];
        const bf16x8 u = up8[i];
        bf16x8 o;
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) {
            const float2 gf = __bfloat1622float2(g.h[k]);
            const float2 uf = __bfloat1622float2(u.h[k]);
            float2 of;
            of.x = silu_f(gf.x) * uf.x;
            of.y = silu_f(gf.y) * uf.y;
            o.h[k] = __float22bfloat162_rn(of);
        }
        out8[i] = o;
    }

    const int64_t tail_start = n_vec * VEC;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(silu_f(g) * u);
    }
}

int num_sms() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        if (sms <= 0) sms = 48;  // GB10 SM count (RTX 5090: 170)
    }
    return sms;
}

bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) % 16) == 0;
}

// Grid for the grid-stride vector kernels: enough blocks to fill the machine several times
// over (8 blocks x 256 threads per SM), but never more blocks than there is vector work.
// The factor of 8 was picked for GB10 (48 SMs); re-tune on the RTX 5090 (170 SMs).
unsigned vec_grid(int64_t n_vec) {
    const int64_t want = static_cast<int64_t>(num_sms()) * 8;
    const int64_t need = cdiv64(n_vec, kBlock);
    int64_t g = want < need ? want : need;
    if (g < 1) g = 1;  // n < VEC: a single block still runs the tail loop
    return static_cast<unsigned>(g);
}

template <typename T>
void swiglu_impl(const T* gate, const T* up, T* out, int64_t n, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(gate != nullptr && up != nullptr && out != nullptr, "swiglu: null pointer");
    SPARK_REQUIRE(n >= 0, "swiglu: n must be >= 0");
    SPARK_REQUIRE(variant >= 0 && variant < swiglu_num_variants(), "swiglu: unknown variant");
    if (n == 0) return;

    if (variant == 0) {
        const int64_t blocks = cdiv64(n, kBlock);
        SPARK_REQUIRE(blocks < (int64_t{1} << 31), "swiglu: n too large for variant 0");
        swiglu_scalar_kernel<T>
            <<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(gate, up, out, n);
        SPARK_CHECK_LAUNCH();
        return;
    }

    SPARK_REQUIRE(aligned16(gate) && aligned16(up) && aligned16(out),
                  "swiglu: variant 1 needs 16-byte aligned pointers");
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    const unsigned grid = vec_grid(n / VEC);
    if constexpr (sizeof(T) == 4) {
        swiglu_vec_f32_kernel<<<grid, kBlock, 0, stream>>>(reinterpret_cast<const float*>(gate),
                                                           reinterpret_cast<const float*>(up),
                                                           reinterpret_cast<float*>(out), n);
    } else {
        swiglu_vec_bf16_kernel<<<grid, kBlock, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate),
            reinterpret_cast<const __nv_bfloat16*>(up), reinterpret_cast<__nv_bfloat16*>(out), n);
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int swiglu_num_variants() {
    return 2;
}

void swiglu_f32(const float* gate, const float* up, float* out, int64_t n, int variant,
                cudaStream_t stream) {
    swiglu_impl<float>(gate, up, out, n, variant, stream);
}

void swiglu_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, int64_t n,
                 int variant, cudaStream_t stream) {
    swiglu_impl<__nv_bfloat16>(gate, up, out, n, variant, stream);
}

}  // namespace spark
