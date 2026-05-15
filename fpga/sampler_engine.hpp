// sampler_engine.hpp — hardware-style categorical sampler.
//
// Algorithm (matching src/fllm/sampler.py):
//   1. Receive logits (float) over vocab.
//   2. Subtract max (shift for numerical stability).
//   3. Optional top-k masking (stream + cutoff).
//   4. LUT-based softmax over the stream.
//   5. Running cumulative sum while consuming probabilities.
//   6. Generate threshold = prng.next_unit() * total_sum.
//   7. Emit the first token id where cum_sum >= threshold.
//
// This is the decode-step bottleneck on FPGA: the token loop does not
// return to the host between tokens.  The entire sampler lives inside
// the token_loop_ctrl FSM.

#ifndef FLLM_SAMPLER_ENGINE_HPP
#define FLLM_SAMPLER_ENGINE_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Xorshift 32-bit PRNG.  Non-zero state is mandatory.
// ---------------------------------------------------------------------------
struct XorshiftPRNG {
    uint32_t state;

    XorshiftPRNG(uint32_t seed = 0xDEADBEEFu) : state(seed != 0 ? seed : 0xDEADBEEFu) {}

    uint32_t next_u32() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }

    float next_unit() {
        return static_cast<float>(next_u32()) / 4294967296.0f;
    }
};

// ---------------------------------------------------------------------------
// softmax_exp_lut — same table as softmax_engine.hpp but exposed for reuse.
// ---------------------------------------------------------------------------
struct SamplerExpLUT {
    static constexpr int ENTRIES = 256;
    static constexpr float MIN = -16.0f;
    static constexpr float MAX = 0.0f;
    static constexpr float STEP = (MAX - MIN) / ENTRIES;
    float table[ENTRIES + 1];

    SamplerExpLUT() {
        for (int i = 0; i <= ENTRIES; ++i) {
            float x = MIN + STEP * i;
            table[i] = std::exp(x);
        }
    }
};

inline float exp_lut_apply(float x, const SamplerExpLUT& lut) {
    if (x < lut.MIN) x = lut.MIN;
    if (x >= lut.MAX - lut.STEP) x = lut.MAX - lut.STEP;
    float idx_f = (x - lut.MIN) / lut.STEP;
    int idx_lo = static_cast<int>(std::floor(idx_f));
    if (idx_lo < 0) idx_lo = 0;
    if (idx_lo >= SamplerExpLUT::ENTRIES) idx_lo = SamplerExpLUT::ENTRIES - 1;
    float frac = idx_f - static_cast<float>(idx_lo);
    float lo = lut.table[idx_lo];
    float hi = lut.table[idx_lo + 1];
    return lo + (hi - lo) * frac;
}

// ---------------------------------------------------------------------------
// greedy_sampler — always picks argmax (no PRNG).
// ---------------------------------------------------------------------------
template<int VOCAB_SIZE>
int greedy_sampler(
    hls::stream<float>& logits_in,
    const SamplerExpLUT& exp_lut
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=logits_in
#pragma HLS INTERFACE bram      port=exp_lut

    // Pass 1: track max and index
    float max_val = -1e30f;
    int   max_idx = -1;
    float buf[VOCAB_SIZE];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

    for (int i = 0; i < VOCAB_SIZE; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float v = logits_in.read();
        buf[i] = v;
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    return max_idx;
}

// ---------------------------------------------------------------------------
// categorical_sampler — PRNG-based sampling with shifted-softmax.
// ---------------------------------------------------------------------------
template<int VOCAB_SIZE>
int categorical_sampler(
    hls::stream<float>& logits_in,
    const SamplerExpLUT& exp_lut,
    XorshiftPRNG& prng,
    float temperature = 1.0f
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=logits_in
#pragma HLS INTERFACE bram      port=exp_lut

    // Pass 1: max tracking + buffer
    float max_val = -1e30f;
    float buf[VOCAB_SIZE];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

    for (int i = 0; i < VOCAB_SIZE; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float v = logits_in.read();
        if (temperature > 0.0f && temperature != 1.0f) {
            v /= temperature;
        }
        buf[i] = v;
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp and cumulative sum
    float cum[VOCAB_SIZE];
#pragma HLS ARRAY_PARTITION variable=cum complete dim=1
    float sum_exp = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float shifted = buf[i] - max_val;
        float ev = exp_lut_apply(shifted, exp_lut);
        sum_exp += ev;
        cum[i] = sum_exp;
    }

    // Pass 3: threshold crossing
    float threshold = prng.next_unit() * sum_exp;
    for (int i = 0; i < VOCAB_SIZE; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        if (cum[i] >= threshold) {
            return i;
        }
    }
    return VOCAB_SIZE - 1;
}

#endif  // FLLM_SAMPLER_ENGINE_HPP
