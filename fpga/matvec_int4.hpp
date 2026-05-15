// matvec_int4.hpp — packed INT4 weight x INT8 activation matrix-vector engine.
//
// Targets AMD/Xilinx VU47P at 600 MHz.
//
// Time-multiplex strategy (lesson borrowed from Talos V2):
//   This same tile is dispatched for every matvec in the block:
//     QKV projection (3x)  -> tile run with stacked output dims
//     O projection         -> tile run, hidden -> hidden
//     MoE up + gate (each) -> tile run, hidden -> inner per active expert
//     MoE down             -> tile run, inner -> hidden per active expert
//     LM head              -> tile run, hidden -> vocab (chunked)
//   The matrix shape is parameterized at call site; the tile itself is one
//   placed instance. Replicate the tile K times to match HBM channel count;
//   each replica is fed by an independent channel/AXI port.
//
// Contract:
//   - Weights are packed 2 INT4 per byte, row-major.
//   - Activations are INT8 (per-tensor scale separate).
//   - Output is INT32 (caller requantizes).
//   - One row of W per pipelined iteration. SIMD = LANES inside the row.
//   - 2:4 structured sparsity: a companion bitmask per weight pair tells the
//     engine to skip zero pairs. This reduces HBM traffic and compute.
//
// Data layout per beat:
//   - Weight beat:  LANES/2 bytes  (2 nibbles/byte  => LANES int4 weights)
//   - Activation beat: LANES bytes  (1 int8 per weight => LANES int8 activations)
//   - Mask beat (optional): LANES/2 bits (2 bits/byte => LANES bits)
//     bit 2*lane+0 = valid for low nibble, bit 2*lane+1 = valid for high nibble.
//
// This file is the *kernel contract*, not synthesizable until placed inside a
// Vitis HLS project. It carries the pragmas, port widths, and II target so the
// Python reference (src/fllm/export.py) and the future RTL implementation share
// one source of truth.

#ifndef FLLM_MATVEC_INT4_HPP
#define FLLM_MATVEC_INT4_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cstdint>

// ---------------------------------------------------------------------------
// Tile geometry.  These can be overridden per-build via -D flags.
// ---------------------------------------------------------------------------
#ifndef FLLM_LANES
#define FLLM_LANES 32
#endif

#ifndef FLLM_TILE_ROWS
#define FLLM_TILE_ROWS 16
#endif

static constexpr int LANES        = FLLM_LANES;      // MACs per cycle (must be even)
static constexpr int TILE_ROWS    = FLLM_TILE_ROWS; // output rows processed in parallel
static constexpr int W_BEAT_BYTES = LANES / 2;       // 2 nibbles per byte
static constexpr int A_BEAT_BYTES = LANES;           // 1 int8 per MAC
static constexpr int M_BEAT_BITS  = LANES;           // 1 bit per MAC (2 per weight byte)

static_assert(LANES % 2 == 0, "LANES must be even for INT4 packing");

// Packed types --------------------------------------------------------------
typedef ap_uint<A_BEAT_BYTES * 8> packed_a_beat_t;   // LANES INT8 activations
typedef ap_uint<W_BEAT_BYTES * 8> packed_w_beat_t;    // LANES/2 bytes => LANES nibbles
typedef ap_uint<M_BEAT_BITS>       sparse_mask_beat_t; // 1 bit per MAC

// Scale per output row (loaded from BRAM at start of the tile).
typedef float scale_t;

// ---------------------------------------------------------------------------
// matvec_int4_tile — one INT4×INT8 matvec tile.
//
// Parameters (template so HLS can unroll / partition statically):
//   IN_FEATURES   : number of input columns (must be a multiple of LANES)
//   OUT_FEATURES  : number of output rows processed by this invocation
//   USE_SPARSITY  : 0 = dense, 1 = read sparse_mask stream and skip zeros
//
// Ports:
//   hbm_w_in      : packed INT4 weights, row-major, one beat = LANES/2 bytes.
//   hbm_mask_in   : (if USE_SPARSITY) one mask bit per MAC.
//   hbm_a_in      : INT8 activations, one beat = LANES bytes.
//   scales        : per-output-row float scale, TILE_ROWS entries.
//   y_out         : INT32 partial sums, one per output row.
// ---------------------------------------------------------------------------
template<int IN_FEATURES, int OUT_FEATURES, int USE_SPARSITY = 0>
void matvec_int4_tile(
    hls::stream<packed_w_beat_t>&   hbm_w_in,
    hls::stream<sparse_mask_beat_t>& hbm_mask_in,
    hls::stream<packed_a_beat_t>&   hbm_a_in,
    const scale_t                   scales[TILE_ROWS],
    hls::stream<ap_int<32>>&        y_out
) {
#pragma HLS INTERFACE axis      port=hbm_w_in
#pragma HLS INTERFACE axis      port=hbm_mask_in
#pragma HLS INTERFACE axis      port=hbm_a_in
#pragma HLS INTERFACE bram      port=scales
#pragma HLS INTERFACE axis      port=y_out
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INLINE off

    constexpr int BEATS_PER_ROW = IN_FEATURES / LANES;
    static_assert(IN_FEATURES % LANES == 0,
                  "IN_FEATURES must be a multiple of LANES");

    // Buffer the activation vector in URAM (one copy, reused for all rows).
    packed_a_beat_t a_buf[BEATS_PER_ROW];
#pragma HLS BIND_STORAGE variable=a_buf type=RAM_1P
    for (int b = 0; b < BEATS_PER_ROW; ++b) {
#pragma HLS PIPELINE II=1
        a_buf[b] = hbm_a_in.read();
    }

    ap_int<32> accum[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=accum complete dim=1

    const int tile_iterations = (OUT_FEATURES + TILE_ROWS - 1) / TILE_ROWS;

    for (int tile_iter = 0; tile_iter < tile_iterations; ++tile_iter) {
        const int rows_this_iter =
            (tile_iter == tile_iterations - 1)
                ? (OUT_FEATURES - tile_iter * TILE_ROWS)
                : TILE_ROWS;

        scale_t row_scale[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=row_scale complete dim=1
        for (int r = 0; r < TILE_ROWS; ++r) {
#pragma HLS UNROLL
            row_scale[r] = (r < rows_this_iter) ? scales[r] : scale_t(0);
            accum[r] = 0;
        }

    ROWS:
        for (int r = 0; r < rows_this_iter; ++r) {
        COLS:
            for (int b = 0; b < BEATS_PER_ROW; ++b) {
#pragma HLS PIPELINE II=1 style=flp
                packed_w_beat_t   w_beat = hbm_w_in.read();
                packed_a_beat_t   a_beat = a_buf[b];
                sparse_mask_beat_t mask_beat;
                if (USE_SPARSITY) {
                    mask_beat = hbm_mask_in.read();
                } else {
                    mask_beat = ~sparse_mask_beat_t(0);
                }

                ap_int<32> partial = 0;

            LANES_LOOP:
                for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
#pragma HLS UNROLL
                    // One weight byte holds two int4 nibbles.
                    ap_uint<8> w_byte = ap_uint<8>(w_beat.range(8 * lane + 7, 8 * lane));
                    ap_int<4> w_lo = ap_int<4>(ap_uint<4>(w_byte.range(3, 0)));
                    ap_int<4> w_hi = ap_int<4>(ap_uint<4>(w_byte.range(7, 4)));

                    // Two activation bytes per weight byte (one per nibble).
                    ap_int<8> a_lo = ap_int<8>(ap_uint<8>(a_beat.range(16 * lane + 7,  16 * lane + 0)));
                    ap_int<8> a_hi = ap_int<8>(ap_uint<8>(a_beat.range(16 * lane + 15, 16 * lane + 8)));

                    // Two mask bits per weight byte.
                    ap_uint<2> mask_pair = ap_uint<2>(mask_beat.range(2 * lane + 1, 2 * lane));

                    if (mask_pair[0]) {
                        partial += ap_mul<32>(w_lo, a_lo);
                    }
                    if (mask_pair[1]) {
                        partial += ap_mul<32>(w_hi, a_hi);
                    }
                }
                accum[r] += partial;
            }
            y_out.write(accum[r]);
        }
    }
}

// ---------------------------------------------------------------------------
// Dense convenience wrapper (no sparsity stream).
// ---------------------------------------------------------------------------
template<int IN_FEATURES, int OUT_FEATURES>
void matvec_int4_tile_dense(
    hls::stream<packed_w_beat_t>& hbm_w_in,
    hls::stream<packed_a_beat_t>& hbm_a_in,
    const scale_t               scales[TILE_ROWS],
    hls::stream<ap_int<32>>&      y_out
) {
#pragma HLS INLINE
    hls::stream<sparse_mask_beat_t> dummy_mask;
#pragma HLS STREAM variable=dummy_mask depth=1
    matvec_int4_tile<IN_FEATURES, OUT_FEATURES, 0>(
        hbm_w_in, dummy_mask, hbm_a_in, scales, y_out);
}

#endif  // FLLM_MATVEC_INT4_HPP
