// Shared benchmark harness. Every bench binary:
//   1. validates the kernel against a reference (CPU or cuBLAS) and exits 1 on mismatch,
//   2. times it with CUDA events (warmup + median and min of N iterations),
//   3. prints a human table to stderr and one JSON object per row to stdout.
// scripts/run_all_benches.sh redirects stdout into results/<kernel>.json.
//
// Flags understood by every bench through this header: --iters=N is read by each main();
// --warmup=N overrides the per-call-site warmup count everywhere (0 also disables the
// clock ramp, which is what scripts/profile_ncu.sh wants).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "spark/common.cuh"

namespace spark::bench {

// Process-wide settings, filled in by Args (--warmup) and print_device_banner (device name).
struct Config {
    int warmup = -1;  // < 0: keep each call site's default
    std::string device = "unknown";
};
inline Config& config() {
    static Config c;
    return c;
}

// A discrete board like the RTX 5090 idles at low clocks and needs a few hundred ms of load
// to reach its boost state (the GB10 ramps too, just less). Spin the first kernel of the
// process for kRampMs once, so the first row of a results file is not measured cold.
constexpr int kRampMs = 300;
inline void ramp_clocks(const std::function<void()>& fn, cudaStream_t stream) {
    static bool done = false;
    if (done) return;
    done = true;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(kRampMs)) {
        fn();
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

struct Timing {
    float median_ms = 0.f;
    float min_ms = 0.f;
    int iters = 0;
};

// Times `fn` (which must enqueue work on `stream`) and returns the median wall time.
inline Timing time_kernel(const std::function<void()>& fn, cudaStream_t stream, int warmup = 10,
                          int iters = 100) {
    // A bad --iters (0, negative, or non-numeric, which atoi maps to 0) would otherwise index
    // an empty sample vector below.
    SPARK_REQUIRE(iters > 0, "time_kernel: iters must be >= 1");
    if (config().warmup >= 0) warmup = config().warmup;
    if (warmup > 0) ramp_clocks(fn, stream);
    cudaEvent_t start, stop;
    SPARK_CUDA_CHECK(cudaEventCreate(&start));
    SPARK_CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) fn();
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        SPARK_CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        SPARK_CUDA_CHECK(cudaEventRecord(stop, stream));
        SPARK_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        SPARK_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(ms);
    }
    SPARK_CUDA_CHECK(cudaEventDestroy(start));
    SPARK_CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    Timing t;
    t.iters = iters;
    t.min_ms = samples.front();
    t.median_ms = samples[samples.size() / 2];
    return t;
}

// Max absolute and max relative error between two host arrays.
struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
};
inline ErrorStats compare(const float* got, const float* ref, size_t n) {
    ErrorStats e;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref[i]);
        e.max_abs = std::max(e.max_abs, d);
        const double denom = std::max(1e-6, std::fabs(static_cast<double>(ref[i])));
        e.max_rel = std::max(e.max_rel, d / denom);
    }
    return e;
}

inline void fill_uniform(std::vector<float>& v, float lo, float hi, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

// One result row. `extra` is appended verbatim inside the JSON object (e.g. "\"M\":4096,").
struct Row {
    std::string kernel;
    std::string dtype;
    int variant = -1;
    std::string shape;
    double median_ms = 0;
    double min_ms = 0;  // best iteration: the gap to the median shows boost/throttle jitter
    double gbps = 0;    // achieved memory bandwidth, if meaningful
    double tflops = 0;  // achieved compute, if meaningful
    double ref_ms = 0;  // reference (cuBLAS / naive) time for the same shape, 0 if none
    double max_abs_err = 0;
    double max_rel_err = 0;
    bool ok = true;
};

inline void print_header() {
    std::fprintf(stderr, "%-14s %-5s %-3s %-20s %10s %10s %9s %9s %10s %10s %s\n", "kernel",
                 "dtype", "var", "shape", "median_ms", "min_ms", "GB/s", "TFLOPS", "ref_ms",
                 "max_abs", "ok");
}

inline void print_row(const Row& r) {
    std::fprintf(stderr, "%-14s %-5s %-3d %-20s %10.4f %10.4f %9.1f %9.2f %10.4f %10.2e %s\n",
                 r.kernel.c_str(), r.dtype.c_str(), r.variant, r.shape.c_str(), r.median_ms,
                 r.min_ms, r.gbps, r.tflops, r.ref_ms, r.max_abs_err, r.ok ? "yes" : "NO");
    std::printf(
        "{\"device\":\"%s\",\"kernel\":\"%s\",\"dtype\":\"%s\",\"variant\":%d,\"shape\":\"%s\","
        "\"median_ms\":%.6f,\"min_ms\":%.6f,\"gbps\":%.3f,\"tflops\":%.4f,\"ref_ms\":%.6f,\"max_"
        "abs_err\":%.6e,\"max_rel_err\":%.6e,"
        "\"ok\":%s}\n",
        config().device.c_str(), r.kernel.c_str(), r.dtype.c_str(), r.variant, r.shape.c_str(),
        r.median_ms, r.min_ms, r.gbps, r.tflops, r.ref_ms, r.max_abs_err, r.max_rel_err,
        r.ok ? "true" : "false");
    std::fflush(stdout);
}

// Simple "--key=value" argument parser.
struct Args {
    std::vector<std::pair<std::string, std::string>> kv;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string s = argv[i];
            if (s.rfind("--", 0) != 0) continue;
            s = s.substr(2);
            auto eq = s.find('=');
            if (eq == std::string::npos) {
                kv.emplace_back(s, "1");
            } else {
                kv.emplace_back(s.substr(0, eq), s.substr(eq + 1));
            }
        }
        if (has("warmup")) config().warmup = std::max(0, geti("warmup", 0));
    }
    std::string get(const std::string& k, const std::string& def) const {
        for (auto& p : kv)
            if (p.first == k) return p.second;
        return def;
    }
    int geti(const std::string& k, int def) const {
        return std::atoi(get(k, std::to_string(def)).c_str());
    }
    bool has(const std::string& k) const {
        for (auto& p : kv)
            if (p.first == k) return true;
        return false;
    }
};

// Device-info banner (stderr). Also records the device name, which print_row writes into
// every JSON row so the results scripts know which machine's peaks to compare against.
inline void print_device_banner() {
    int dev = 0;
    SPARK_CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    SPARK_CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    config().device = p.name;
    // cudaDeviceProp::clockRate was removed in CUDA 13; query the attribute instead.
    int clock_khz = 0;
    SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
    std::fprintf(stderr, "device: %s | sm_%d%d | %d SMs | %.0f MHz | L2 %d MB\n", p.name, p.major,
                 p.minor, p.multiProcessorCount, clock_khz / 1000.0, p.l2CacheSize >> 20);
}

}  // namespace spark::bench
