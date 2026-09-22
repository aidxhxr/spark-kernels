// Benchmark: fused SwiGLU (silu(gate) * up) for f32 and bf16.
//   ./bench_swiglu [--n=<elements>] [--iters=<N>]
// Default shapes: 4096 tokens x {2048, 5632, 11008, 14336} (Llama/Qwen MLP intermediate sizes).
// Validates every variant against a CPU reference; exits 1 on mismatch.
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark::bench;

namespace {

float silu_host(float x) {
    return x / (1.0f + std::exp(-x));
}

// |got - ref| <= atol + rtol * |ref| for every element.
bool within_tol(const std::vector<float>& got, const std::vector<float>& ref, double atol,
                double rtol) {
    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (!(d <= atol + rtol * std::fabs(static_cast<double>(ref[i])))) return false;
    }
    return true;
}

template <typename T>
struct Traits;
template <>
struct Traits<float> {
    static const char* name() { return "f32"; }
    static float to_host(float v) { return v; }
    static float from_host(float v) { return v; }
    static double atol() { return 1e-5; }
    static double rtol() { return 1e-5; }
    static void launch(const float* g, const float* u, float* o, int64_t n, int v, cudaStream_t s) {
        spark::swiglu_f32(g, u, o, n, v, s);
    }
};
template <>
struct Traits<__nv_bfloat16> {
    static const char* name() { return "bf16"; }
    static float to_host(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __nv_bfloat16 from_host(float v) { return __float2bfloat16(v); }
    static double atol() { return 2e-2; }
    static double rtol() { return 2e-2; }
    static void launch(const __nv_bfloat16* g, const __nv_bfloat16* u, __nv_bfloat16* o, int64_t n,
                       int v, cudaStream_t s) {
        spark::swiglu_bf16(g, u, o, n, v, s);
    }
};

template <typename T>
bool run_shape(int64_t n, const std::string& shape, int iters, cudaStream_t stream) {
    using Tr = Traits<T>;
    const size_t count = static_cast<size_t>(n);
    const size_t bytes = count * sizeof(T);

    std::vector<float> h_gate(count), h_up(count);
    fill_uniform(h_gate, -3.0f, 3.0f, 1);
    fill_uniform(h_up, -3.0f, 3.0f, 2);

    // Round inputs through the storage type so the reference sees exactly what the GPU sees.
    std::vector<T> t_gate(count), t_up(count), t_out(count);
    std::vector<float> ref(count);
    for (size_t i = 0; i < count; ++i) {
        t_gate[i] = Tr::from_host(h_gate[i]);
        t_up[i] = Tr::from_host(h_up[i]);
        const float g = Tr::to_host(t_gate[i]);
        const float u = Tr::to_host(t_up[i]);
        ref[i] = silu_host(g) * u;
    }

    T *d_gate = nullptr, *d_up = nullptr, *d_out = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d_gate, bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&d_up, bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&d_out, bytes));
    SPARK_CUDA_CHECK(cudaMemcpy(d_gate, t_gate.data(), bytes, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(d_up, t_up.data(), bytes, cudaMemcpyHostToDevice));

    const double moved_gb = 3.0 * static_cast<double>(bytes) / 1e9;
    std::vector<float> got(count);
    bool all_ok = true;
    double v0_ms = 0.0;

    for (int v = 0; v < spark::swiglu_num_variants(); ++v) {
        SPARK_CUDA_CHECK(cudaMemsetAsync(d_out, 0, bytes, stream));
        Tr::launch(d_gate, d_up, d_out, n, v, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(t_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) got[i] = Tr::to_host(t_out[i]);

        const ErrorStats err = compare(got.data(), ref.data(), count);
        const bool ok = within_tol(got, ref, Tr::atol(), Tr::rtol());
        all_ok = all_ok && ok;

        const Timing t =
            time_kernel([&] { Tr::launch(d_gate, d_up, d_out, n, v, stream); }, stream, 10, iters);
        if (v == 0) v0_ms = t.median_ms;

        Row r;
        r.kernel = "swiglu";
        r.dtype = Tr::name();
        r.variant = v;
        r.shape = shape;
        r.median_ms = t.median_ms;
        r.min_ms = t.min_ms;
        r.gbps = moved_gb / (t.median_ms * 1e-3);
        r.ref_ms = v0_ms;  // reference = the naive variant on the same shape
        r.max_abs_err = err.max_abs;
        r.max_rel_err = err.max_rel;
        r.ok = ok;
        print_row(r);
    }

    SPARK_CUDA_CHECK(cudaFree(d_gate));
    SPARK_CUDA_CHECK(cudaFree(d_up));
    SPARK_CUDA_CHECK(cudaFree(d_out));
    return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        print_device_banner();
        const int iters = args.geti("iters", 100);

        std::vector<std::pair<int64_t, std::string>> shapes;
        if (args.has("n")) {
            const int64_t n = args.geti64("n", 0);
            shapes.emplace_back(n, "n=" + std::to_string(n));
        } else {
            const int rows = 4096;
            for (int cols : {2048, 5632, 11008, 14336}) {
                shapes.emplace_back(static_cast<int64_t>(rows) * cols,
                                    std::to_string(rows) + "x" + std::to_string(cols));
            }
        }

        cudaStream_t stream;
        SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
        print_header();
        bool ok = true;
        for (const auto& s : shapes) {
            ok = run_shape<float>(s.first, s.second, iters, stream) && ok;
            ok = run_shape<__nv_bfloat16>(s.first, s.second, iters, stream) && ok;
        }
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
