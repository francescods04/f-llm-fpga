// silu_lut_tb.cpp — host-compilable testbench for silu_lut.hpp.
//
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas -I. fpga/silu_lut_tb.cpp -o /tmp/silu_lut_tb
// Run:
//   /tmp/silu_lut_tb

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/silu_lut.hpp"

static inline float silu_exact(float x) {
    return x / (1.0f + std::exp(-x));
}

int main() {
    constexpr int N = 4096;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    SiLULUT lut(1024, -8.0f, 8.0f);
    hls::stream<float> in_s(8192);
    hls::stream<float> out_s(8192);

    std::vector<float> inputs;
    for (int i = 0; i < N; ++i) {
        float x = dist(rng);
        inputs.push_back(x);
        in_s.write(x);
    }

    silu_lut_tile(in_s, out_s, lut);

    float max_err = 0.0f, mean_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float hw = out_s.read();
        float ref = silu_exact(inputs[i]);
        float err = std::abs(hw - ref);
        if (err > max_err) max_err = err;
        mean_err += err;
    }
    mean_err /= N;

    std::printf("silu_lut  N=%d entries=%d\n", N, lut.num_entries);
    std::printf("max_abs_err  = %.6f\n", max_err);
    std::printf("mean_abs_err = %.6f\n", mean_err);
    std::printf("%s\n", (max_err < 3e-4f) ? "PASS" : "FAIL");
    return (max_err < 3e-4f) ? 0 : 1;
}
