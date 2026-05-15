// rope_engine.hpp — Rotary Position Embedding (RoPE) for FPGA decode.
//
// At decode time (batch=1, seq=1) we need sin/cos for the *absolute*
// position = cache_length.  The table is precomputed at bitstream time
// and stored in BRAM: two tables of (max_ctx, head_dim/2) float entries.
//
// Oracolo Python: src/fllm/attention.apply_rope

#ifndef FLLM_ROPE_ENGINE_HPP
#define FLLM_ROPE_ENGINE_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Precomputed sin/cos tables (one pair per absolute position).
// ---------------------------------------------------------------------------
template<int MAX_CTX, int HEAD_DIM>
struct RoPETables {
    static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even");
    static constexpr int HALF = HEAD_DIM / 2;
    float cos_table[MAX_CTX][HALF];
    float sin_table[MAX_CTX][HALF];

    RoPETables(float theta = 10000.0f) {
        for (int pos = 0; pos < MAX_CTX; ++pos) {
            for (int i = 0; i < HALF; ++i) {
                float freq = 1.0f / std::pow(theta, static_cast<float>(2 * i) / HEAD_DIM);
                float angle = pos * freq;
                cos_table[pos][i] = std::cos(angle);
                sin_table[pos][i] = std::sin(angle);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// rope_tile — apply rotation to Q and K vectors for one decode step.
//
//   q_in / k_in:  (num_heads * head_dim) floats for Q,
//                 (num_kv_heads * head_dim) floats for K.
//   position:     absolute position (from KV cache length).
// ---------------------------------------------------------------------------
template<int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int MAX_CTX>
void rope_tile(
    hls::stream<float>& q_in,
    hls::stream<float>& k_in,
    hls::stream<float>& q_out,
    hls::stream<float>& k_out,
    const RoPETables<MAX_CTX, HEAD_DIM>& tables,
    int position
) {
#pragma HLS INTERFACE axis      port=q_in
#pragma HLS INTERFACE axis      port=k_in
#pragma HLS INTERFACE axis      port=q_out
#pragma HLS INTERFACE axis      port=k_out
#pragma HLS INTERFACE bram      port=tables
#pragma HLS INLINE off

    static_assert(HEAD_DIM % 2 == 0, "HEAD_DIM must be even");
    constexpr int HALF = HEAD_DIM / 2;

    // Clamp position to table bounds
    if (position < 0) position = 0;
    if (position >= MAX_CTX) position = MAX_CTX - 1;

    // Process Q (NUM_HEADS vectors)
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < HALF; ++i) {
#pragma HLS PIPELINE II=1 style=flp
            float x1 = q_in.read();
            float x2 = q_in.read();
            float c = tables.cos_table[position][i];
            float s = tables.sin_table[position][i];
            float rx1 = x1 * c - x2 * s;
            float rx2 = x1 * s + x2 * c;
            q_out.write(rx1);
            q_out.write(rx2);
        }
    }

    // Process K (NUM_KV_HEADS vectors)
    for (int h = 0; h < NUM_KV_HEADS; ++h) {
        for (int i = 0; i < HALF; ++i) {
#pragma HLS PIPELINE II=1 style=flp
            float x1 = k_in.read();
            float x2 = k_in.read();
            float c = tables.cos_table[position][i];
            float s = tables.sin_table[position][i];
            float rx1 = x1 * c - x2 * s;
            float rx2 = x1 * s + x2 * c;
            k_out.write(rx1);
            k_out.write(rx2);
        }
    }
}

#endif  // FLLM_ROPE_ENGINE_HPP
