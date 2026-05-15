// silu_lut.hpp — piecewise-linear SiLU via BRAM LUT.
//
// FPGA datapath:
//   1. Clamp input to [lut_min, lut_max).
//   2. Scale to table index: idx_f = (x - lut_min) / lut_step.
//   3. idx_lo = floor(idx_f), frac = idx_f - idx_lo.
//   4. Read table[idx_lo] and table[idx_lo+1] from BRAM.
//   5. Output = lo + (hi - lo) * frac.
//
// Table is precomputed at bitstream-build time and stored in one 18 Kb BRAM.
// For 1024 entries of float32: 4 KB = 1 BRAM.
//
// Oracolo Python: src/fllm/lut_activations.silu_lut

#ifndef FLLM_SILU_LUT_HPP
#define FLLM_SILU_LUT_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// SiLU = x * sigmoid(x)
// ---------------------------------------------------------------------------
static inline float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

// ---------------------------------------------------------------------------
// Build the table at run time (constructor) so no float template params.
// ---------------------------------------------------------------------------
struct SiLULUT {
    static constexpr int DEFAULT_ENTRIES = 1024;
    static constexpr float DEFAULT_MIN = -8.0f;
    static constexpr float DEFAULT_MAX = 8.0f;

    int num_entries;
    float lut_min;
    float lut_max;
    float step;
    float* table;  // owned, num_entries+1 floats

    SiLULUT(int entries = DEFAULT_ENTRIES, float minv = DEFAULT_MIN, float maxv = DEFAULT_MAX)
        : num_entries(entries), lut_min(minv), lut_max(maxv),
          step((maxv - minv) / entries), table(new float[entries + 1]) {
        for (int i = 0; i <= entries; ++i) {
            float x = lut_min + step * i;
            table[i] = silu_ref(x);
        }
    }

    ~SiLULUT() { delete[] table; }

    // disallow copy to avoid double-free; move is fine
    SiLULUT(const SiLULUT&) = delete;
    SiLULUT& operator=(const SiLULUT&) = delete;
};

// ---------------------------------------------------------------------------
// silu_lut_tile — stream one value per cycle (II=1).
// ---------------------------------------------------------------------------
inline void silu_lut_tile(
    hls::stream<float>& in_stream,
    hls::stream<float>& out_stream,
    const SiLULUT& lut
) {
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream
#pragma HLS INTERFACE bram      port=lut
#pragma HLS INLINE off

    while (!in_stream.empty()) {
#pragma HLS PIPELINE II=1 style=flp
        float x = in_stream.read();
        float xc = x;
        if (xc < lut.lut_min) xc = lut.lut_min;
        if (xc >= lut.lut_max - lut.step) xc = lut.lut_max - lut.step;

        float idx_f = (xc - lut.lut_min) / lut.step;
        int idx_lo = static_cast<int>(std::floor(idx_f));
        if (idx_lo < 0) idx_lo = 0;
        if (idx_lo >= lut.num_entries) idx_lo = lut.num_entries - 1;
        float frac = idx_f - static_cast<float>(idx_lo);

        float lo = lut.table[idx_lo];
        float hi = lut.table[idx_lo + 1];
        float y = lo + (hi - lo) * frac;
        out_stream.write(y);
    }
}

#endif  // FLLM_SILU_LUT_HPP
