// Benchmark: memory-bandwidth probe (device-to-device copy).
//   ./bench_bandwidth [--n=<floats>] [--iters=<N>]
// Reference (ref_ms) is cudaMemcpyAsync D2D on the same buffers: the practical ceiling.
#include <cstdint>
#include <cstring>
#include <exception>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark::bench;

namespace {

std::string shape_str(int64_t n) {
    const int64_t mib = n >> 20;
    return "n=" + std::to_string(mib) + "M";
}

bool run_size(int64_t n, int iters, cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> h_x(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        h_x[static_cast<size_t>(i)] = static_cast<float>(i % 8191) * 0.25f;
    std::vector<float> h_y(static_cast<size_t>(n));

    float *d_x = nullptr, *d_y = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d_x, bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&d_y, bytes));
    SPARK_CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    // Reference: cudaMemcpy device-to-device.
    const Timing ref = time_kernel(
        [&] {
            SPARK_CUDA_CHECK(cudaMemcpyAsync(d_y, d_x, bytes, cudaMemcpyDeviceToDevice, stream));
        },
        stream, 5, iters);
    const double moved_gb = 2.0 * static_cast<double>(bytes) / 1e9;
    {
        Row r;
        r.kernel = "cudaMemcpy_d2d";
        r.dtype = "f32";
        r.variant = -1;
        r.shape = shape_str(n);
        r.median_ms = ref.median_ms;
        r.min_ms = ref.min_ms;
        r.gbps = moved_gb / (ref.median_ms * 1e-3);
        r.ref_ms = ref.median_ms;
        r.ok = true;
        print_row(r);
    }

    bool all_ok = true;
    for (int v = 0; v < spark::bandwidth_num_variants(); ++v) {
        SPARK_CUDA_CHECK(cudaMemsetAsync(d_y, 0, bytes, stream));
        spark::bandwidth_copy(d_x, d_y, n, v, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
        const bool ok = std::memcmp(h_x.data(), h_y.data(), bytes) == 0;
        all_ok = all_ok && ok;

        const Timing t =
            time_kernel([&] { spark::bandwidth_copy(d_x, d_y, n, v, stream); }, stream, 5, iters);
        Row r;
        r.kernel = "bandwidth_copy";
        r.dtype = "f32";
        r.variant = v;
        r.shape = shape_str(n);
        r.median_ms = t.median_ms;
        r.min_ms = t.min_ms;
        r.gbps = moved_gb / (t.median_ms * 1e-3);
        r.ref_ms = ref.median_ms;
        r.max_abs_err = ok ? 0.0 : 1.0;
        r.max_rel_err = ok ? 0.0 : 1.0;
        r.ok = ok;
        print_row(r);
    }

    SPARK_CUDA_CHECK(cudaFree(d_x));
    SPARK_CUDA_CHECK(cudaFree(d_y));
    return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        print_device_banner();
        const int iters = args.geti("iters", 50);
        std::vector<int64_t> sizes;
        if (args.has("n")) {
            sizes.push_back(args.geti64("n", 0));
        } else {
            sizes.push_back(int64_t{64} << 20);   // 64M floats  = 256 MiB per buffer
            sizes.push_back(int64_t{256} << 20);  // 256M floats = 1 GiB per buffer
        }
        cudaStream_t stream;
        SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
        print_header();
        bool ok = true;
        for (int64_t n : sizes) ok = run_size(n, iters, stream) && ok;
        SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
        if (!ok) {
            std::fprintf(stderr, "VALIDATION FAILED\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
