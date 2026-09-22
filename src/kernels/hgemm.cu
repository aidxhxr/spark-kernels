// bf16 tensor-core GEMM ladder for Blackwell sm_12x: RTX 5090 (sm_120), GB10 (sm_121).
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// All variants use the WMMA API, which lowers to mma.sync tensor-core instructions.
// The RTX 5090 and GB10 are consumer/workstation-lineage Blackwell (sm_120 / sm_121): they
// support mma.sync but NOT the datacenter-only tcgen05 / TMA path of sm_100 (B200). WMMA /
// mma.sync is therefore the right portable tool here.
//
//   variant 0: one warp per 16x16 C tile, fragments loaded straight from global memory.
//   variant 1: 128x128x32 block tile, 8 warps (2x4), shared-memory staged with +8 padding.
//   variant 2: variant 1 + two-stage cp.async pipeline (load tile k+1 while computing tile k).
//
// Tile sizes come from the GB10 analysis (48 SMs, 273 GB/s, 24 MB L2) in
// docs/design/hgemm.md; re-tune on the RTX 5090 (170 SMs, 1,792 GB/s GDDR7).

#include <mma.h>  // after cuda_bf16.h (pulled in by common.cuh)

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

using namespace nvcuda;

namespace {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

using FragA =
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using FragB =
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// Convert 8 fp32 values (16-byte aligned source) into 8 bf16 packed as one uint4 store.
__device__ __forceinline__ uint4 pack8_bf16(const float* src) {
    __nv_bfloat162 h0 = __floats2bfloat162_rn(src[0], src[1]);
    __nv_bfloat162 h1 = __floats2bfloat162_rn(src[2], src[3]);
    __nv_bfloat162 h2 = __floats2bfloat162_rn(src[4], src[5]);
    __nv_bfloat162 h3 = __floats2bfloat162_rn(src[6], src[7]);
    uint4 out;
    out.x = *reinterpret_cast<unsigned*>(&h0);
    out.y = *reinterpret_cast<unsigned*>(&h1);
    out.z = *reinterpret_cast<unsigned*>(&h2);
    out.w = *reinterpret_cast<unsigned*>(&h3);
    return out;
}

// ---------------------------------------------------------------------------------------
// Variant 0: WMMA baseline. Block = 4 warps covering a 32x32 region (2x2 warp tiles of 16x16).
// Each warp walks K in steps of 16 loading fragments directly from global memory.
// ---------------------------------------------------------------------------------------
constexpr int V0_TILE = 32;
constexpr int V0_THREADS = 128;

__global__ void __launch_bounds__(V0_THREADS)
    hgemm_v0_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) float scratch[4][WMMA_M * WMMA_N];

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tile_row = blockIdx.y * V0_TILE + (warp_id >> 1) * WMMA_M;
    const int tile_col = blockIdx.x * V0_TILE + (warp_id & 1) * WMMA_N;
    // M, N are multiples of 16, so a tile is either fully inside or fully outside.
    if (tile_row >= M || tile_col >= N) return;

    FragA a_frag;
    FragB b_frag;
    FragC c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(tile_row) * K + k0, K);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(k0) * N + tile_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* s = scratch[warp_id];
    wmma::store_matrix_sync(s, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();
    // 32 lanes x 8 elements = 256 = one 16x16 tile.
    const int r = lane >> 1;
    const int c0 = (lane & 1) * 8;
    const size_t out = static_cast<size_t>(tile_row + r) * N + tile_col + c0;
    *reinterpret_cast<uint4*>(C + out) = pack8_bf16(s + r * WMMA_N + c0);
}

// ---------------------------------------------------------------------------------------
// Shared tiling parameters for variants 1 and 2.
// ---------------------------------------------------------------------------------------
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int PAD = 8;             // +8 bf16 = +16 bytes per row: kills bank conflicts
constexpr int A_LD = BK + PAD;     // 40 elements (80 bytes) per smem row of A
constexpr int B_LD = BN + PAD;     // 136 elements (272 bytes) per smem row of B
constexpr int TILE_THREADS = 256;  // 8 warps
constexpr int WARPS_M = 2;         // warp grid 2 (M) x 4 (N)
constexpr int WARPS_N = 4;
constexpr int WARP_TILE_M = BM / WARPS_M;      // 64
constexpr int WARP_TILE_N = BN / WARPS_N;      // 32
constexpr int FRAGS_M = WARP_TILE_M / WMMA_M;  // 4
constexpr int FRAGS_N = WARP_TILE_N / WMMA_N;  // 2

static_assert(WARPS_M * WARPS_N * 32 == TILE_THREADS, "warp layout must match block size");
static_assert(A_LD % 8 == 0 && B_LD % 8 == 0, "WMMA ldm must be a multiple of 8 bf16");

// A tile: BM x BK = 4096 bf16 = 512 chunks of 8; B tile: BK x BN = 4096 bf16 = 512 chunks.
constexpr int A_CHUNKS_PER_ROW = BK / 8;                         // 4
constexpr int B_CHUNKS_PER_ROW = BN / 8;                         // 16
constexpr int CHUNKS_PER_THREAD = (BM * BK / 8) / TILE_THREADS;  // 2
static_assert(CHUNKS_PER_THREAD * TILE_THREADS == BM * BK / 8, "A tile must divide evenly");
static_assert(CHUNKS_PER_THREAD * TILE_THREADS == BK * BN / 8, "B tile must divide evenly");

// Bounds-checked (zero-filled) register-path tile load used by variant 1.
__device__ __forceinline__ void load_tile_sync(__nv_bfloat16 (*As)[A_LD], __nv_bfloat16 (*Bs)[B_LD],
                                               const __nv_bfloat16* __restrict__ A,
                                               const __nv_bfloat16* __restrict__ B, int M, int N,
                                               int K, int bm, int bn, int k0) {
    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int i = 0; i < CHUNKS_PER_THREAD; ++i) {
        const int c = threadIdx.x + i * TILE_THREADS;
        // A: row = c / 4, col = (c % 4) * 8
        {
            const int row = c / A_CHUNKS_PER_ROW;
            const int col = (c % A_CHUNKS_PER_ROW) * 8;
            const int grow = bm + row;
            const int gcol = k0 + col;
            uint4 v = zero;
            if (grow < M && gcol < K) {  // K % 16 == 0 => a chunk is fully in or fully out
                v = *reinterpret_cast<const uint4*>(A + static_cast<size_t>(grow) * K + gcol);
            }
            *reinterpret_cast<uint4*>(&As[row][col]) = v;
        }
        // B: row = c / 16, col = (c % 16) * 8
        {
            const int row = c / B_CHUNKS_PER_ROW;
            const int col = (c % B_CHUNKS_PER_ROW) * 8;
            const int grow = k0 + row;
            const int gcol = bn + col;
            uint4 v = zero;
            if (grow < K && gcol < N) {  // N % 16 == 0 => a chunk is fully in or fully out
                v = *reinterpret_cast<const uint4*>(B + static_cast<size_t>(grow) * N + gcol);
            }
            *reinterpret_cast<uint4*>(&Bs[row][col]) = v;
        }
    }
}

// In-bounds cp.async tile load used by variant 2 (host guarantees M,N % 128 == 0, K % 32 == 0).
__device__ __forceinline__ void load_tile_async(__nv_bfloat16 (*As)[A_LD],
                                                __nv_bfloat16 (*Bs)[B_LD],
                                                const __nv_bfloat16* __restrict__ A,
                                                const __nv_bfloat16* __restrict__ B, int N, int K,
                                                int bm, int bn, int k0) {
#pragma unroll
    for (int i = 0; i < CHUNKS_PER_THREAD; ++i) {
        const int c = threadIdx.x + i * TILE_THREADS;
        {
            const int row = c / A_CHUNKS_PER_ROW;
            const int col = (c % A_CHUNKS_PER_ROW) * 8;
            cp_async_16(&As[row][col], A + static_cast<size_t>(bm + row) * K + k0 + col);
        }
        {
            const int row = c / B_CHUNKS_PER_ROW;
            const int col = (c % B_CHUNKS_PER_ROW) * 8;
            cp_async_16(&Bs[row][col], B + static_cast<size_t>(k0 + row) * N + bn + col);
        }
    }
}

// One BK=32 step: two k-slices of 16, FRAGS_M x FRAGS_N mma per slice per warp.
__device__ __forceinline__ void compute_tile(FragC (&acc)[FRAGS_M][FRAGS_N],
                                             __nv_bfloat16 (*As)[A_LD], __nv_bfloat16 (*Bs)[B_LD],
                                             int warp_m, int warp_n) {
#pragma unroll
    for (int kk = 0; kk < BK; kk += WMMA_K) {
        FragA a_frag[FRAGS_M];
        FragB b_frag[FRAGS_N];
#pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            wmma::load_matrix_sync(a_frag[i], &As[warp_m * WARP_TILE_M + i * WMMA_M][kk], A_LD);
        }
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::load_matrix_sync(b_frag[j], &Bs[kk][warp_n * WARP_TILE_N + j * WMMA_N], B_LD);
        }
#pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
    }
}

// Epilogue: stage one 16x16 fp32 fragment per warp at a time, convert to bf16, vector store.
__device__ __forceinline__ void store_tile(FragC (&acc)[FRAGS_M][FRAGS_N], float* warp_scratch,
                                           __nv_bfloat16* __restrict__ C, int M, int N, int bm,
                                           int bn, int warp_m, int warp_n, int lane) {
    const int r = lane >> 1;
    const int c0 = (lane & 1) * 8;
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(warp_scratch, acc[i][j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            const int grow = bm + warp_m * WARP_TILE_M + i * WMMA_M + r;
            const int gcol = bn + warp_n * WARP_TILE_N + j * WMMA_N + c0;
            if (grow < M && gcol < N) {  // N % 16 == 0 => the 8-wide chunk is fully inside
                *reinterpret_cast<uint4*>(C + static_cast<size_t>(grow) * N + gcol) =
                    pack8_bf16(warp_scratch + r * WMMA_N + c0);
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------------------
// Variant 1: shared-memory staged 128x128x32 tile, single buffer.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    hgemm_v1_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) __nv_bfloat16 As[BM][A_LD];                    // 10,240 B
    __shared__ __align__(32) __nv_bfloat16 Bs[BK][B_LD];                    //  8,704 B
    __shared__ __align__(32) float Cs[TILE_THREADS / 32][WMMA_M * WMMA_N];  // 8,192 B

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    FragC acc[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    for (int k0 = 0; k0 < K; k0 += BK) {
        load_tile_sync(As, Bs, A, B, M, N, K, bm, bn, k0);
        __syncthreads();
        compute_tile(acc, As, Bs, warp_m, warp_n);
        __syncthreads();
    }

    store_tile(acc, Cs[warp_id], C, M, N, bm, bn, warp_m, warp_n, lane);
}

// ---------------------------------------------------------------------------------------
// Variant 2: two-stage cp.async pipeline. Global->shared copies for tile k+1 are in flight
// while the tensor cores chew on tile k.
// ---------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    hgemm_v2_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(32) __nv_bfloat16 As[2][BM][A_LD];  // 20,480 B
    __shared__ __align__(32) __nv_bfloat16 Bs[2][BK][B_LD];  // 17,408 B
    __shared__ __align__(
        32) float Cs[TILE_THREADS / 32][WMMA_M * WMMA_N];  // 8,192 B  (46,080 B total)

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    FragC acc[FRAGS_M][FRAGS_N];
#pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    const int num_tiles = K / BK;

    // Prologue: kick off tile 0.
    load_tile_async(As[0], Bs[0], A, B, N, K, bm, bn, 0);
    cp_async_commit();

    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < num_tiles) {
            load_tile_async(As[cur ^ 1], Bs[cur ^ 1], A, B, N, K, bm, bn, (t + 1) * BK);
            cp_async_commit();
            cp_async_wait<1>();  // everything except the newest group (tile t+1) has landed
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();  // make tile t visible to all warps
        compute_tile(acc, As[cur], Bs[cur], warp_m, warp_n);
        __syncthreads();  // everyone is done reading stage `cur` before it is refilled at t+1
    }

    store_tile(acc, Cs[warp_id], C, M, N, bm, bn, warp_m, warp_n, lane);
}

}  // namespace

int hgemm_num_variants() {
    return 3;
}

bool hgemm_supports(int M, int N, int K, int variant) {
    if (variant < 0 || variant >= hgemm_num_variants()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) return false;
    if (variant == 2) return M % BM == 0 && N % BN == 0 && K % BK == 0;
    return true;
}

void hgemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                int K, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(A != nullptr && B != nullptr && C != nullptr, "hgemm: null pointer");
    SPARK_REQUIRE(M > 0 && N > 0 && K > 0, "hgemm: M, N, K must be positive");
    SPARK_REQUIRE(M % WMMA_M == 0 && N % WMMA_N == 0 && K % WMMA_K == 0,
                  "hgemm: M, N, K must be multiples of 16");
    SPARK_REQUIRE(variant >= 0 && variant < hgemm_num_variants(), "hgemm: unknown variant");

    switch (variant) {
        case 0: {
            const dim3 grid(cdiv(N, V0_TILE), cdiv(M, V0_TILE));
            hgemm_v0_kernel<<<grid, V0_THREADS, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 1: {
            const dim3 grid(cdiv(N, BN), cdiv(M, BM));
            hgemm_v1_kernel<<<grid, TILE_THREADS, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 2: {
            SPARK_REQUIRE(hgemm_supports(M, N, K, 2),
                          "hgemm variant 2: requires M % 128 == 0, N % 128 == 0, K % 32 == 0");
            const dim3 grid(N / BN, M / BM);
            hgemm_v2_kernel<<<grid, TILE_THREADS, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
