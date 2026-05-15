// sampler_engine_tb.cpp — host-compilable testbench for sampler engine.
//
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas -I. fpga/sampler_engine_tb.cpp -o /tmp/sampler_engine_tb
// Run:
//   /tmp/sampler_engine_tb

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/sampler_engine.hpp"

static inline int greedy_ref(const std::vector<float>& logits) {
    int best = 0;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) best = static_cast<int>(i);
    }
    return best;
}

static inline int categorical_ref(const std::vector<float>& logits, float threshold) {
    float max_val = logits[0];
    for (auto v : logits) if (v > max_val) max_val = v;
    float sum = 0.0f;
    std::vector<float> probs;
    for (auto v : logits) {
        float p = std::exp(v - max_val);
        probs.push_back(p);
        sum += p;
    }
    float target = threshold * sum;
    float cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i];
        if (cum >= target) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size()) - 1;
}

int main() {
    constexpr int VOCAB = 128;

    std::mt19937 rng(2025);
    std::normal_distribution<float> dist(0.0f, 2.0f);

    std::vector<float> logits;
    for (int i = 0; i < VOCAB; ++i) logits.push_back(dist(rng));

    SamplerExpLUT exp_lut;

    // Test greedy
    {
        hls::stream<float> in_s(1024);
        for (auto v : logits) in_s.write(v);
        int hw = greedy_sampler<VOCAB>(in_s, exp_lut);
        int ref = greedy_ref(logits);
        std::printf("greedy: hw=%d ref=%d %s\n", hw, ref, (hw == ref) ? "PASS" : "FAIL");
        if (hw != ref) return 1;
    }

    // Test categorical with known threshold
    {
        XorshiftPRNG prng(42);
        // Force deterministic threshold by pre-seeding and reading once
        float threshold = prng.next_unit();
        // Re-seed so the kernel gets the same first value
        XorshiftPRNG prng2(42);

        hls::stream<float> in_s(1024);
        for (auto v : logits) in_s.write(v);
        int hw = categorical_sampler<VOCAB>(in_s, exp_lut, prng2);
        int ref = categorical_ref(logits, threshold);
        std::printf("categorical: hw=%d ref=%d thresh=%.6f %s\n",
                    hw, ref, threshold, (hw == ref) ? "PASS" : "FAIL");
        if (hw != ref) return 1;
    }

    std::printf("sampler_engine ALL PASS\n");
    return 0;
}
