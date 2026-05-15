// softmax_engine_tb.cpp — host-compilable testbench for softmax_engine.hpp.
//
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas -I. fpga/softmax_engine_tb.cpp -o /tmp/softmax_engine_tb
// Run:
//   /tmp/softmax_engine_tb

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/softmax_engine.hpp"

static inline float softmax_exact(const std::vector<float>& x,
                                   std::vector<float>& out) {
    float max_val = x[0];
    for (auto v : x) if (v > max_val) max_val = v;
    float sum = 0.0f;
    for (auto v : x) {
        float e = std::exp(v - max_val);
        out.push_back(e);
        sum += e;
    }
    for (auto& v : out) v /= sum;
    return sum;
}

int main() {
    constexpr int SEQ = 128;

    std::mt19937 rng(2025);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> x(SEQ);
    for (int i = 0; i < SEQ; ++i) x[i] = dist(rng);

    hls::stream<float> in_s;
    hls::stream<float> out_s;
    for (int i = 0; i < SEQ; ++i) in_s.write(x[i]);

    DefaultExpLUT exp_lut;
    softmax_tile<SEQ>(in_s, out_s, exp_lut);

    std::vector<float> ref;
    softmax_exact(x, ref);

    float max_err = 0.0f;
    float sum_hw = 0.0f;
    for (int i = 0; i < SEQ; ++i) {
        float hw = out_s.read();
        sum_hw += hw;
        float err = std::abs(hw - ref[i]);
        if (err > max_err) max_err = err;
    }

    std::printf("softmax_engine SEQ=%d\n", SEQ);
    std::printf("sum_hw      = %.8f\n", sum_hw);
    std::printf("max_abs_err = %.6f\n", max_err);
    std::printf("%s\n", (max_err < 1e-2f && std::abs(sum_hw - 1.0f) < 1e-3f) ? "PASS" : "FAIL");
    return (max_err < 1e-2f && std::abs(sum_hw - 1.0f) < 1e-3f) ? 0 : 1;
}
