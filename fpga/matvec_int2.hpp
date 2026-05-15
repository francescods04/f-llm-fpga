// matvec_int2.hpp — packed INT2 (ternary) weight x INT8 activation tile.
//
// Weights are packed 4 per byte:
//   00 = -1,  01 = 0,  10 = +1,  11 = reserved (treated as 0)
//
// Activation stream: INT8, one beat = LANES bytes.
// Weight stream:   one beat = LANES/4 bytes (4 int2 per byte).
//
// Targets the same FPGA tile bank as INT4, just with different unpacking.

#ifndef FLLM_MATVEC_INT2_HPP
#define FLLM_MATVEC_INT2_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cstdint>

// ---------------------------------------------------------------------------
// Tile geometry (must match matvec_int4.hpp build flags).
// ---------------------------------------------------------------------------
#ifndef FLLM_LANES
#define FLLM_LANES 32
#endif

#ifndef FLLM_TILE_ROWS
#define FLLM_TILE_ROWS 16
#endif

static constexpr int LANES        = FLLM_LANES;
static constexpr int TILE_ROWS    = FLLM_TILE_ROWS;
static constexpr int W_BEAT_BYTES = LANES / 4;       // 4 nibbles per byte
static constexpr int A_BEAT_BYTES = LANES;           // 1 int8 per MAC

static_assert(LANES % 4 == 0, "LANES must be multiple of 4 for INT2 packing");

// Packed types --------------------------------------------------------------
typedef ap_uint<A_BEAT_BYTES * 8> packed_a_beat_t;
typedef ap_uint<W_BEAT_BYTES * 8> packed_w_beat_t;
typedef float scale_t;

// ---------------------------------------------------------------------------
// Decode two bits into ternary value {-1, 0, +1} as signed int32.
// ---------------------------------------------------------------------------
inline ap_int<32> decode_int2(ap_uint<2> b) {
    // 00 -> -1, 01 -> 0, 10 -> +1, 11 -> 0 (reserved)
    int v = static_cast<int>(b);
    if (v == 0) return ap_int<32>(-1);
    if (v == 1) return ap_int<32>(0);
    if (v == 2) return ap_int<32>(1);
    return ap_int<32>(0);  // case 3
}

// ---------------------------------------------------------------------------
// matvec_int2_tile — one INT2×INT8 matvec tile.
//
// Parameters:
//   IN_FEATURES   : must be multiple of LANES
//   OUT_FEATURES  : total output rows (must be multiple of TILE_ROWS for simplicity)
//
// Ports:
//   hbm_w_in : packed INT2 weights, one beat = LANES/4 bytes.
//   hbm_a_in : INT8 activations, one beat = LANES bytes.
//   scales   : per-output-row float scale, TILE_ROWS entries.
//   y_out    : INT32 partial sums, one per output row.
// ---------------------------------------------------------------------------
template<int IN_FEATURES, int OUT_FEATURES>
void matvec_int2_tile(
    hls::stream<packed_w_beat_t>& hbm_w_in,
    hls::stream<packed_a_beat_t>& hbm_a_in,
    const scale_t               scales[TILE_ROWS],
    hls::stream<ap_int<32>>&      y_out
) {
#pragma HLS INTERFACE axis      port=hbm_w_in
#pragma HLS INTERFACE axis      port=hbm_a_in
#pragma HLS INTERFACE bram      port=scales
#pragma HLS INTERFACE axis      port=y_out
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INLINE off

    constexpr int BEATS_PER_ROW = IN_FEATURES / LANES;
    static_assert(IN_FEATURES % LANES == 0,
                  "IN_FEATURES must be a multiple of LANES");

    // Buffer activation vector in URAM (one copy reused for all rows).
    packed_a_beat_t a_buf[BEATS_PER_ROW];
#pragma HLS BIND_STORAGE variable=a_buf type=RAM_1P
    for (int b = 0; b < BEATS_PER_ROW; ++b) {
#pragma HLS PIPELINE II=1
        a_buf[b] = hbm_a_in.read();
    }

    ap_int<32> accum[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=accum complete dim=1

    constexpr int NUM_TILES = OUT_FEATURES / TILE_ROWS;

    for (int tile_iter = 0; tile_iter < NUM_TILES; ++tile_iter) {
        scale_t row_scale[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=row_scale complete dim=1
        for (int r = 0; r < TILE_ROWS; ++r) {
#pragma HLS UNROLL
            row_scale[r] = scales[r];
            accum[r] = 0;
        }

    ROWS:
        for (int r = 0; r < TILE_ROWS; ++r) {
        COLS:
            for (int b = 0; b < BEATS_PER_ROW; ++b) {
#pragma HLS PIPELINE II=1 style=flp
                packed_w_beat_t w_beat = hbm_w_in.read();
                packed_a_beat_t a_beat = a_buf[b];

                ap_int<32> partial = 0;

            LANES_LOOP:
                for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
#pragma HLS UNROLL
                    // One weight byte holds four int2 values.
                    ap_uint<8> w_byte = ap_uint<8>(w_beat.range(8 * lane + 7, 8 * lane));

                    // Four activation bytes per weight byte (32 bits total).
                    ap_int<8> a0 = ap_int<8>(ap_uint<8>(a_beat.range(32 * lane + 7,  32 * lane + 0)));
                    ap_int<8> a1 = ap_int<8>(ap_uint<8>(a_beat.range(32 * lane + 15, 32 * lane + 8)));
                    ap_int<8> a2 = ap_int<8>(ap_uint<8>(a_beat.range(32 * lane + 23, 32 * lane + 16)));
                    ap_int<8> a3 = ap_int<8>(ap_uint<8>(a_beat.range(32 * lane + 31, 32 * lane + 24)));

                    // Decode each 2-bit nibble.
                    ap_uint<2> b0 = ap_uint<2>(w_byte.range(1, 0));
                    ap_uint<2> b1 = ap_uint<2>(w_byte.range(3, 2));
                    ap_uint<2> b2 = ap_uint<2>(w_byte.range(5, 4));
                    ap_uint<2> b3 = ap_uint<2>(w_byte.range(7, 6));

                    ap_int<32> w0 = decode_int2(b0);
                    ap_int<32> w1 = decode_int2(b1);
                    ap_int<32> w2 = decode_int2(b2);
                    ap_int<32> w3 = decode_int2(b3);

                    partial += ap_mul<32>(w0, a0);
                    partial += ap_mul<32>(w1, a1);
                    partial += ap_mul<32>(w2, a2);
                    partial += ap_mul<32>(w3, a3);
                }
                accum[r] += partial;
            }
            y_out.write(accum[r]);
        }
    }
}

#endif  // FLLM_MATVEC_INT2_HPP
