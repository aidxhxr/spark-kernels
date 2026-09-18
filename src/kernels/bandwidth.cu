// Memory-bandwidth probe: y = x over n floats.
//
// This kernel does no arithmetic, so its throughput is the practical ceiling for every
// memory-bound kernel in this repo (RMSNorm, SwiGLU, softmax). The three variants walk the
// usual optimization ladder for a streaming kernel on GB10:
//   0: one float per thread, one block per 256 elements (naive)
//   1: one float4 (16 bytes) per thread -> one 128-bit transaction per lane
//   2: float4 + grid-stride loop with a fixed, occupancy-sized grid (~4 blocks per SM)
#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

constexpr int kBlock = 256;

__global__ void copy_scalar_kernel(const float* __restrict__ x, float* __restrict__ y,
                                   int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

__global__ void copy_float4_kernel(const float4* __restrict__ x, float4* __restrict__ y,
                                   int64_t n4) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n4) y[i] = x[i];
}

__global__ void copy_float4_gridstride_kernel(const float4* __restrict__ x,
                                              float4* __restrict__ y, int64_t n4) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4;
         i += stride) {
        y[i] = x[i];
    }
}

// Scalar remainder for the vectorized variants (n % 4 != 0). At most 3 elements, one block.
__global__ void copy_tail_kernel(const float* __restrict__ x, float* __restrict__ y,
                                 int64_t start, int64_t n) {
    const int64_t i = start + threadIdx.x;
    if (i < n) y[i] = x[i];
}

int num_sms() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        if (sms <= 0) sms = 48;  // GB10 fallback
    }
    return sms;
}

unsigned grid_for(int64_t work_items, int block) {
    const int64_t blocks = cdiv64(work_items, block);
    SPARK_REQUIRE(blocks > 0 && blocks < (int64_t{1} << 31), "bandwidth_copy: grid too large");
    return static_cast<unsigned>(blocks);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) % 16) == 0; }

}  // namespace

int bandwidth_num_variants() { return 3; }

void bandwidth_copy(const float* x, float* y, int64_t n, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(x != nullptr && y != nullptr, "bandwidth_copy: null pointer");
    SPARK_REQUIRE(n >= 0, "bandwidth_copy: n must be >= 0");
    SPARK_REQUIRE(variant >= 0 && variant < bandwidth_num_variants(),
                  "bandwidth_copy: unknown variant");
    if (n == 0) return;

    if (variant == 0) {
        copy_scalar_kernel<<<grid_for(n, kBlock), kBlock, 0, stream>>>(x, y, n);
        SPARK_CHECK_LAUNCH();
        return;
    }

    // Vectorized variants need 16-byte aligned pointers (cudaMalloc guarantees 256 bytes).
    SPARK_REQUIRE(aligned16(x) && aligned16(y),
                  "bandwidth_copy: vectorized variants need 16-byte aligned pointers");
    const int64_t n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    if (n4 > 0) {
        if (variant == 1) {
            copy_float4_kernel<<<grid_for(n4, kBlock), kBlock, 0, stream>>>(x4, y4, n4);
        } else {
            // ~4 resident blocks of 256 threads per SM keeps every SM busy without a long
            // tail of partially-filled waves; each thread then loops over the array.
            const int64_t want = static_cast<int64_t>(num_sms()) * 4;
            const int64_t need = cdiv64(n4, kBlock);
            const unsigned grid = static_cast<unsigned>(want < need ? want : need);
            copy_float4_gridstride_kernel<<<grid, kBlock, 0, stream>>>(x4, y4, n4);
        }
        SPARK_CHECK_LAUNCH();
    }

    const int64_t tail_start = n4 * 4;
    if (tail_start < n) {
        copy_tail_kernel<<<1, 32, 0, stream>>>(x, y, tail_start, n);
        SPARK_CHECK_LAUNCH();
    }
}

}  // namespace spark
