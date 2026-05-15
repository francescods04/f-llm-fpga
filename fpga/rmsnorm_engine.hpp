// rmsnorm_engine.hpp — fixed-point streaming RMSNorm.
//
// Stages:
//   1. Read N elements, accumulate sum-of-squares (mean_sq = sum / N).
//   2. Compute rsqrt(mean_sq + eps) via log2-domain LUT.
//   3. Stream each element * weight * rsqrt_scale out.
//
// The rsqrt LUT lives in one BRAM (512 entries, float32).
//
// Oracolo Python: src/fllm/lut_activations.rsqrt_lut

#ifndef FLLM_RMSNORM_ENGINE_HPP
#define FLLM_RMSNORM_ENGINE_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// rsqrt LUT over [1.0, 2.0) using log2 decomposition.
//   x = 2^e * m,  m in [1, 2)
//   rsqrt(x) = rsqrt(m) * 2^(-e/2)
// ---------------------------------------------------------------------------
template<int NUM_ENTRIES>
struct RSqrtLUT {
    static constexpr int ENTRIES = NUM_ENTRIES;
    static constexpr float MIN = 1.0f;
    static constexpr float MAX = 2.0f;
    static constexpr float STEP = (MAX - MIN) / ENTRIES;
    float table[ENTRIES + 1];

    RSqrtLUT() {
        for (int i = 0; i <= ENTRIES; ++i) {
            float m = MIN + STEP * i;
            table[i] = 1.0f / std::sqrt(m);
        }
    }
};

template<int NUM_ENTRIES>
float rsqrt_lut_apply(float x, const RSqrtLUT<NUM_ENTRIES>& lut) {
    constexpr float EPS = 1e-6f;
    if (x <= 0.0f) x = EPS;
    float e_f = std::floor(std::log2(x));
    int e = static_cast<int>(e_f);
    float m = x / std::pow(2.0f, e_f);
    if (m < lut.MIN) m = lut.MIN;
    if (m >= lut.MAX - lut.STEP) m = lut.MAX - lut.STEP;

    float idx_f = (m - lut.MIN) / lut.STEP;
    int idx_lo = static_cast<int>(std::floor(idx_f));
    if (idx_lo < 0) idx_lo = 0;
    if (idx_lo >= NUM_ENTRIES) idx_lo = NUM_ENTRIES - 1;
    float frac = idx_f - static_cast<float>(idx_lo);

    float lo = lut.table[idx_lo];
    float hi = lut.table[idx_lo + 1];
    float rsqrt_m = lo + (hi - lo) * frac;
    return rsqrt_m * std::pow(2.0f, -e_f / 2.0f);
}

using DefaultRSqrtLUT = RSqrtLUT<512>;

// ---------------------------------------------------------------------------
// rmsnorm_tile — process HIDDEN elements, emit normalized elements.
// ---------------------------------------------------------------------------
template<int HIDDEN>
void rmsnorm_tile(
    hls::stream<float>& in_stream,      // HIDDEN elements in
    hls::stream<float>& out_stream,     // HIDDEN elements out
    const float         weight[HIDDEN], // per-element scale (BRAM)
    const DefaultRSqrtLUT& rsqrt_lut,
    float               eps = 1e-6f
) {
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream
#pragma HLS INTERFACE bram      port=weight
#pragma HLS INTERFACE bram      port=rsqrt_lut
#pragma HLS INLINE off

    float buf[HIDDEN];
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1

    // Pass 1: read and accumulate sum-of-squares
    float sum_sq = 0.0f;
    for (int i = 0; i < HIDDEN; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float v = in_stream.read();
        buf[i] = v;
        sum_sq += v * v;
    }

    float mean_sq = sum_sq / static_cast<float>(HIDDEN);
    float scale = rsqrt_lut_apply<512>(mean_sq + eps, rsqrt_lut);

    // Pass 2: write normalized values
    for (int i = 0; i < HIDDEN; ++i) {
#pragma HLS PIPELINE II=1 style=flp
        float v = buf[i] * scale * weight[i];
        out_stream.write(v);
    }
}

#endif  // FLLM_RMSNORM_ENGINE_HPP
