// rope_engine_tb.cpp — host-compilable testbench for RoPE.
//
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas -I. fpga/rope_engine_tb.cpp -o /tmp/rope_engine_tb
// Run:
//   /tmp/rope_engine_tb

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/rope_engine.hpp"

static inline void apply_rope_ref(
    const std::vector<float>& in,
    std::vector<float>& out,
    int head_dim,
    int position,
    float theta
) {
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
        float angle = position * freq;
        float c = std::cos(angle);
        float s = std::sin(angle);
        float x1 = in[2 * i];
        float x2 = in[2 * i + 1];
        out.push_back(x1 * c - x2 * s);
        out.push_back(x1 * s + x2 * c);
    }
}

int main() {
    constexpr int NUM_HEADS = 4;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int HEAD_DIM = 64;
    constexpr int MAX_CTX = 512;
    constexpr float THETA = 10000.0f;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Build Q and K vectors
    int q_size = NUM_HEADS * HEAD_DIM;
    int k_size = NUM_KV_HEADS * HEAD_DIM;
    std::vector<float> q_in, k_in;
    for (int i = 0; i < q_size; ++i) q_in.push_back(dist(rng));
    for (int i = 0; i < k_size; ++i) k_in.push_back(dist(rng));

    hls::stream<float> q_s(4096);
    hls::stream<float> k_s(4096);
    hls::stream<float> q_out_s(4096);
    hls::stream<float> k_out_s(4096);

    for (auto v : q_in) q_s.write(v);
    for (auto v : k_in) k_s.write(v);

    RoPETables<MAX_CTX, HEAD_DIM> tables(THETA);
    int position = 77;
    rope_tile<NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_CTX>(
        q_s, k_s, q_out_s, k_out_s, tables, position);

    // Compare against reference
    std::vector<float> q_ref, k_ref;
    for (int h = 0; h < NUM_HEADS; ++h) {
        std::vector<float> head_in(q_in.begin() + h * HEAD_DIM, q_in.begin() + (h + 1) * HEAD_DIM);
        apply_rope_ref(head_in, q_ref, HEAD_DIM, position, THETA);
    }
    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        std::vector<float> head_in(k_in.begin() + h * HEAD_DIM, k_in.begin() + (h + 1) * HEAD_DIM);
        apply_rope_ref(head_in, k_ref, HEAD_DIM, position, THETA);
    }

    float max_err = 0.0f;
    for (int i = 0; i < q_size; ++i) {
        float hw = q_out_s.read();
        float err = std::abs(hw - q_ref[i]);
        if (err > max_err) max_err = err;
    }
    for (int i = 0; i < k_size; ++i) {
        float hw = k_out_s.read();
        float err = std::abs(hw - k_ref[i]);
        if (err > max_err) max_err = err;
    }

    std::printf("rope_engine heads=%d/%d head_dim=%d pos=%d\n", NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, position);
    std::printf("max_abs_err = %.6f\n", max_err);
    std::printf("%s\n", (max_err < 1e-5f) ? "PASS" : "FAIL");
    return (max_err < 1e-5f) ? 0 : 1;
}
