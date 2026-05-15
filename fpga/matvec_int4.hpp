// matvec_int4.hpp — packed INT4 weight x INT8 activation matrix-vector engine.
//
// First FPGA kernel for the F-LLM pipeline. Targets AMD/Xilinx VU47P at 600 MHz.
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
//   - One row of W per pipelined iteration. SIMD = 8 lanes inside the row.
//
// This file is the *kernel contract*, not synthesizable until placed inside a
// Vitis HLS project. It carries the pragmas, port widths, and II target so the
// Python reference (src/fllm/export.py) and the future RTL implementation share
// one source of truth.

#ifndef FLLM_MATVEC_INT4_HPP
#define FLLM_MATVEC_INT4_HPP

#include <ap_int.h>
#include <hls_stream.h>

// Per-row scales live in BRAM. Output channels processed in PE tiles.
static constexpr int LANES        = 32;   // SIMD inside a row; matches DSP SIMD pack
static constexpr int TILE_ROWS    = 16;   // output rows processed in parallel
static constexpr int IN_FEATURES  = 4096; // hidden dim for Qwen3-A3B
static constexpr int OUT_FEATURES = 4096; // out dim per projection

typedef ap_int<4>  int4_t;
typedef ap_int<8>  int8_t_;
typedef ap_int<32> int32_t_;

// Packed weight type: 2x int4 = 1 byte; LANES lanes per beat.
typedef ap_uint<LANES * 8> packed_w_beat_t;
typedef ap_uint<LANES * 8> packed_a_beat_t; // INT8 activation, LANES per beat

// One MAC tile: process TILE_ROWS x LANES MACs per cycle.
// Target II = 1. Each row consumes IN_FEATURES/LANES beats.
//
// hbm_w_in: AXI burst from HBM, weight matrix in row-major packed-int4 layout.
// hbm_a_in: BRAM/URAM stream of INT8 activations for this matvec.
// scales:   one float per output row, loaded once at start.
// y_out:    INT32 partial-sums per output row.
void matvec_int4_tile(
    hls::stream<packed_w_beat_t>& hbm_w_in,
    hls::stream<packed_a_beat_t>& hbm_a_in,
    const float                   scales[TILE_ROWS],
    hls::stream<int32_t_>&        y_out
) {
#pragma HLS INTERFACE axis      port=hbm_w_in
#pragma HLS INTERFACE axis      port=hbm_a_in
#pragma HLS INTERFACE bram      port=scales
#pragma HLS INTERFACE axis      port=y_out

    int32_t_ accum[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=accum complete dim=1

INIT:
    for (int r = 0; r < TILE_ROWS; r++) {
#pragma HLS UNROLL
        accum[r] = 0;
    }

    constexpr int BEATS_PER_ROW = IN_FEATURES / LANES;

ROWS:
    for (int r = 0; r < TILE_ROWS; r++) {
COLS:
        for (int b = 0; b < BEATS_PER_ROW; b++) {
#pragma HLS PIPELINE II=1
            packed_w_beat_t w_beat = hbm_w_in.read();
            packed_a_beat_t a_beat = hbm_a_in.read();
            int32_t_ partial = 0;
        LANES_LOOP:
            for (int lane = 0; lane < LANES; lane++) {
#pragma HLS UNROLL
                // Each lane holds two int4 weights packed into one byte.
                ap_uint<8> w_byte = w_beat.range(8 * lane + 7, 8 * lane);
                int4_t  w_lo = w_byte.range(3, 0);
                int4_t  w_hi = w_byte.range(7, 4);
                int8_t_ a_lo = a_beat.range(8 * lane + 7, 8 * lane);
                // NOTE: lane pairs share the same activation byte in this draft.
                // Real layout will interleave activations to feed both halves.
                partial += w_lo * a_lo + w_hi * a_lo;
            }
            accum[r] += partial;
        }
        y_out.write(accum[r]);
    }
}

#endif  // FLLM_MATVEC_INT4_HPP
