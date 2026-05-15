// softmax_engine.hpp — shifted softmax with LUT-based exp.
//
// Algorithm (3-pass, all streaming):
//   Pass 1: read elements, track running max.
//   Pass 2: shifted = x - max;  exp_val = LUT_exp(shifted);
//           accumulate sum_exp.
//   Pass 3: read elements again (buffered), output = exp_val / sum_exp.
//
// The exp LUT covers [-16, 0] with 256 entries (1 BRAM).
// Inputs are float; outputs are float probabilities.
//
// Oracolo Python: src/fllm/lut_activations.softmax_shifted

#ifndef FLLM_SOFTMAX_ENGINE_HPP
#define FLLM_SOFTMAX_ENGINE_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// exp LUT over [-16.0, 0.0]
// ---------------------------------------------------------------------------
template<int NUM_ENTRIES>
struct ExpLUT {
    static constexpr int ENTRIES = NUM_ENTRIES;
    static constexpr float MIN = -16.0f;
    static constexpr float MAX = 0.0f;
    static constexpr float STEP = (MAX - MIN) / ENTRIES;
    float table[ENTRIES + 1];

    ExpLUT() {
        for (int i = 0; i <= ENTRIES; ++i) {
            float x = MIN + STEP * i;
            table[i] = std::exp(x);
        }
    }
};

using DefaultExpLUT = ExpLUT<256>;

template<int NUM_ENTRIES>
float exp_lut_apply(float x, const ExpLUT<NUM_ENTRIES>& lut) {
    if (x < lut.MIN) x = lut.MIN;
    if (x >= lut.MAX - lut.STEP) x = lut.MAX - lut.STEP;

    float idx_f = (x - lut.MIN) / lut.STEP;
    int idx_lo = static_cast<int>(std::floor(idx_f));
    if (idx_lo < 0) idx_lo = 0;
    if (idx_lo >= NUM_ENTRIES) idx_lo = NUM_ENTRIES - 1;
    float frac = idx_f - static_cast<float>(idx_lo);

    float lo = lut.table[idx_lo];
    float hi = lut.table[idx_lo + 1];
    return lo + (hi - lo) * frac;
}

// ---------------------------------------------------------------------------
// softmax_tile — process SEQ_LEN elements, emit probabilities.
// ---------------------------------------------------------------------------
template<int SEQ_LEN>
void softmax_tile(
    hls::stream<float>& in_stream,   // SEQ_LEN logits in
    hls::stream<float>& out_stream,  // SEQ_LEN probabilities out
    const DefaultExpLUT& exp_lut
) {
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream
#pragma HLS INTERFACE bram      port=exp_lut
#pragma HLS INLINE off

    float buf[SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

    // Pass 1: find max
    float max_val = -1e30f;
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float v = in_stream.read();
        buf[i] = v;
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float shifted = buf[i] - max_val;
        float ev = exp_lut_apply<256>(shifted, exp_lut);
        buf[i] = ev;
        sum_exp += ev;
    }

    // Pass 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        out_stream.write(buf[i] * inv_sum);
    }
}

#endif  // FLLM_SOFTMAX_ENGINE_HPP
