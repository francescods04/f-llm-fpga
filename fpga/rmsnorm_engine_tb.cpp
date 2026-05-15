// rmsnorm_engine_tb.cpp — host-compilable testbench for rmsnorm_engine.hpp.

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/rmsnorm_engine.hpp"

int main() {
    constexpr int HIDDEN = 512;  // kept small to avoid hls::stream overflow in stub
    constexpr float EPS = 1e-6f;

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> x(HIDDEN);
    float weight[HIDDEN];
    for (int i = 0; i < HIDDEN; ++i) {
        x[i] = dist(rng);
        weight[i] = 1.0f + 0.1f * dist(rng);
    }

    hls::stream<float> in_s(1024);
    hls::stream<float> out_s(1024);
    for (int i = 0; i < HIDDEN; ++i) in_s.write(x[i]);

    DefaultRSqrtLUT rsqrt_lut;
    rmsnorm_tile<HIDDEN>(in_s, out_s, weight, rsqrt_lut, EPS);

    float sum_sq = 0.0f;
    for (auto v : x) sum_sq += v * v;
    float mean_sq = sum_sq / HIDDEN;
    float scale = 1.0f / std::sqrt(mean_sq + EPS);

    float max_err = 0.0f;
    for (int i = 0; i < HIDDEN; ++i) {
        float hw = out_s.read();
        float ref = x[i] * scale * weight[i];
        float err = std::abs(hw - ref);
        if (err > max_err) max_err = err;
    }

    std::printf("rmsnorm_engine HIDDEN=%d\n", HIDDEN);
    std::printf("max_abs_err = %.6f\n", max_err);
    std::printf("%s\n", (max_err < 1e-3f) ? "PASS" : "FAIL");
    return (max_err < 1e-3f) ? 0 : 1;
}
