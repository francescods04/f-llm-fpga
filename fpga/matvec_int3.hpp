// matvec_int3.hpp — INT3 (range -4..+3) × INT8 matvec kernel.
//
// 3-bit weights are packed 8 per 3 bytes (or 10 per 32 bits with ap_uint).
// For simplicity this host-test version packs 2 weights per byte
// (wasting 2 bits) but exercises the exact INT3 datapath the FPGA will use.
//
// Build:
//   g++ -std=c++17 -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
//       -o /tmp/matvec_int3_tb fpga/matvec_int3_tb.cpp && /tmp/matvec_int3_tb

#ifndef FLLM_MATVEC_INT3_HPP
#define FLLM_MATVEC_INT3_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include "matvec_int4.hpp"   // for packed_a_beat_t, scale_t

#include <cstdint>

#ifndef LANES
#define LANES 32
#endif
#ifndef TILE_ROWS
#define TILE_ROWS 16
#endif

static_assert(LANES % 8 == 0, "LANES must be multiple of 8 for INT3");

// Packed weight beat: one byte per 2 weights (wasteful but host-simple).
static constexpr int W3_BEAT_BYTES = LANES / 2;
typedef ap_uint<W3_BEAT_BYTES * 8> packed_w3_beat_t;

inline int8_t sext3(uint8_t n) {
    int8_t v = static_cast<int8_t>(n & 0x07);
    if (v & 0x04) v -= 8;
    return v;
}

inline int8_t sext3_nibble(ap_uint<4> nibble) {
    uint8_t v = static_cast<uint8_t>(int(nibble) & 0x07);
    int8_t s = static_cast<int8_t>(v);
    if (s & 0x04) s -= 8;
    return s;
}

// Tile: OUT_FEATURES rows × IN_FEATURES columns, buffered activation
// Same contract as matvec_int4_tile_dense but with 3-bit weights.
template<int IN_FEATURES, int OUT_FEATURES>
void matvec_int3_tile_dense(
    hls::stream<packed_w3_beat_t>& s_w,
    hls::stream<packed_a_beat_t>& s_a,
    const scale_t scales[TILE_ROWS],
    hls::stream<ap_int<32>>& s_y
) {
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INTERFACE mode=axis port=s_w
#pragma HLS INTERFACE mode=axis port=s_a
#pragma HLS INTERFACE mode=axis port=s_y

    static_assert(IN_FEATURES % LANES == 0, "IN_FEATURES must be multiple of LANES");

    constexpr int W_BEAT_BYTES = LANES / 2;         // 2 weights per byte
    constexpr int BEATS_PER_ROW = IN_FEATURES / LANES;
    constexpr int ROWS = (OUT_FEATURES + TILE_ROWS - 1) / TILE_ROWS;

    // Activation URAM buffer (one copy)
    packed_a_beat_t a_uram[BEATS_PER_ROW];
    #pragma HLS BIND_STORAGE variable=a_uram type=RAM_1P

    for (int b = 0; b < BEATS_PER_ROW; ++b) {
        a_uram[b] = s_a.read();
    }

    for (int r = 0; r < ROWS; ++r) {
        for (int local_row = 0; local_row < TILE_ROWS; ++local_row) {
            #pragma HLS PIPELINE II=1
            ap_int<32> acc = 0;
            for (int b = 0; b < BEATS_PER_ROW; ++b) {
                #pragma HLS UNROLL
                packed_w3_beat_t wb = s_w.read();
                packed_a_beat_t  ab = a_uram[b];
                for (int byte_lane = 0; byte_lane < LANES / 2; ++byte_lane) {
                    #pragma HLS UNROLL
                    ap_uint<8> w_byte = ap_uint<8>(wb.range(8 * byte_lane + 7, 8 * byte_lane));
                    ap_int<4> w_lo = ap_int<4>(sext3_nibble(w_byte.range(3, 0)));
                    ap_int<4> w_hi = ap_int<4>(sext3_nibble(w_byte.range(7, 4)));
                    ap_int<8> a_lo = ap_int<8>(ap_uint<8>(ab.range(16 * byte_lane + 7,  16 * byte_lane + 0)));
                    ap_int<8> a_hi = ap_int<8>(ap_uint<8>(ab.range(16 * byte_lane + 15, 16 * byte_lane + 8)));
                    acc += ap_mul<32>(w_lo, a_lo) + ap_mul<32>(w_hi, a_hi);
                }
            }
            int global_row = r * TILE_ROWS + local_row;
            if (global_row < OUT_FEATURES) {
                float scale = scales[local_row];
                float val = static_cast<float>(static_cast<int>(acc)) * scale;
                ap_int<32> out = static_cast<ap_int<32>>(static_cast<int>(val));
                s_y.write(out);
            }
        }
    }
}

#endif  // FLLM_MATVEC_INT3_HPP
