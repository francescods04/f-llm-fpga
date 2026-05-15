// kv_bfp4.hpp — pack/unpack engine for 4-bit mantissa Block Floating Point.
//
// Each block of `BLOCK_SIZE` values shares one int8 exponent.
// Each value stores a 4-bit signed mantissa (range -8..+7).
// Packed size per block = 1 byte (exp) + BLOCK_SIZE/2 bytes (mantissas).
//
// Build:
//   g++ -std=c++17 -I./fpga -o /tmp/kv_bfp4_tb fpga/kv_bfp4_tb.cpp && /tmp/kv_bfp4_tb

#ifndef FLLM_KV_BFP4_HPP
#define FLLM_KV_BFP4_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cmath>
#include <cstdint>

// BFP4 block configuration
template<int BLOCK_SIZE>
struct BFP4Block {
    static_assert(BLOCK_SIZE % 2 == 0, "BLOCK_SIZE must be even");
    int8_t exponent;                    // shared exponent (power-of-2 scale)
    uint8_t mantissa[BLOCK_SIZE / 2];   // 2× 4-bit mantissas per byte
};

// Pack a float stream into BFP4 blocks.
// Input:  float values, one per cycle.
// Output: packed BFP4 blocks written to HBM-style stream.
template<int BLOCK_SIZE>
void kv_bfp4_pack(
    hls::stream<float>& float_in,
    hls::stream<BFP4Block<BLOCK_SIZE>>& bfp_out,
    int num_values
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE mode=axis port=float_in
#pragma HLS INTERFACE mode=axis port=bfp_out

    constexpr int HALF = BLOCK_SIZE / 2;
    for (int base = 0; base < num_values; base += BLOCK_SIZE) {
        float block_buf[BLOCK_SIZE];
        #pragma HLS BIND_STORAGE variable=block_buf type=RAM_1P

        // 1. Read block into local buffer and find abs max
        float abs_max = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            #pragma HLS PIPELINE II=1
            float v = (base + i < num_values) ? float_in.read() : 0.0f;
            block_buf[i] = v;
            float av = std::abs(v);
            if (av > abs_max) abs_max = av;
        }

        // 2. Compute exponent: scale = abs_max / 7  (qmax for 4-bit signed)
        float scale = abs_max / 7.0f;
        if (scale < 1e-30f) scale = 1e-30f;
        int exp_val = static_cast<int>(std::ceil(std::log2(scale)));
        float pow2 = std::pow(2.0f, static_cast<float>(exp_val));

        // 3. Quantize and pack
        BFP4Block<BLOCK_SIZE> out;
        out.exponent = static_cast<int8_t>(exp_val);
        for (int i = 0; i < HALF; ++i) {
            #pragma HLS PIPELINE II=1
            float v0 = block_buf[i * 2 + 0];
            float v1 = block_buf[i * 2 + 1];
            int8_t m0 = static_cast<int8_t>(std::round(v0 / pow2));
            int8_t m1 = static_cast<int8_t>(std::round(v1 / pow2));
            if (m0 < -8) m0 = -8; if (m0 > 7) m0 = 7;
            if (m1 < -8) m1 = -8; if (m1 > 7) m1 = 7;
            uint8_t packed = (static_cast<uint8_t>(m1 & 0x0F) << 4)
                           |  static_cast<uint8_t>(m0 & 0x0F);
            out.mantissa[i] = packed;
        }
        bfp_out.write(out);
    }
}

// Unpack BFP4 blocks back to float stream.
template<int BLOCK_SIZE>
void kv_bfp4_unpack(
    hls::stream<BFP4Block<BLOCK_SIZE>>& bfp_in,
    hls::stream<float>& float_out,
    int num_values
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE mode=axis port=bfp_in
#pragma HLS INTERFACE mode=axis port=float_out

    constexpr int HALF = BLOCK_SIZE / 2;
    for (int base = 0; base < num_values; base += BLOCK_SIZE) {
        BFP4Block<BLOCK_SIZE> blk = bfp_in.read();
        float scale = std::pow(2.0f, static_cast<float>(blk.exponent));
        for (int i = 0; i < HALF; ++i) {
            #pragma HLS PIPELINE II=1
            uint8_t packed = blk.mantissa[i];
            int8_t m0 = static_cast<int8_t>(packed & 0x0F);
            if (m0 & 0x08) m0 -= 16;  // sign-extend 4-bit to 8-bit
            int8_t m1 = static_cast<int8_t>(packed >> 4);
            if (m1 & 0x08) m1 -= 16;
            float_out.write(static_cast<float>(m0) * scale);
            if (base + i * 2 + 1 < num_values) {
                float_out.write(static_cast<float>(m1) * scale);
            }
        }
    }
}

#endif  // FLLM_KV_BFP4_HPP
