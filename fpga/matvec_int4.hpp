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
//   - 2:4 structured sparsity: a companion bitmask per 4 weights (2 bytes)
//     tells the engine to skip zero pairs. This reduces HBM traffic and
//     compute for pruned layers.
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

static constexpr int LANES        = FLLM_LANES;      // SIMD inside a row
static constexpr int TILE_ROWS    = FLLM_TILE_ROWS; // output rows processed in parallel

// Packed types --------------------------------------------------------------
// One activation lane = 8 bits.  One packed beat = LANES activations.
typedef ap_uint<LANES * 8> packed_a_beat_t;

// One weight beat = LANES bytes, each holding two int4 weights (low nibble,
// high nibble).  Total weight pairs per beat = LANES * 2.
typedef ap_uint<LANES * 8> packed_w_beat_t;

// One sparsity mask bit per weight pair.  With 2:4 structured sparsity, half
// the pairs are zero and can be skipped.
typedef ap_uint<LANES * 2> sparse_mask_beat_t;

// Scale per output row (loaded from BRAM at start of the tile).
typedef float scale_t;

// ---------------------------------------------------------------------------
// matvec_int4_tile — one INT4×INT8 matvec tile.
//
// Parameters (template so HLS can unroll / partition statically):
//   IN_FEATURES   : number of input columns (must divide LANES*2 for sparsity)
//   OUT_FEATURES  : number of output rows processed by this invocation
//                   (may be > TILE_ROWS; caller loops externally)
//   USE_SPARSITY  : 0 = dense, 1 = read sparse_mask stream and skip zeros
//
// Ports:
//   hbm_w_in      : packed INT4 weights, row-major, one beat = LANES bytes.
//   hbm_mask_in   : (if USE_SPARSITY) one mask beat per weight beat.
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
    // Interface pragmas -----------------------------------------------------
#pragma HLS INTERFACE axis      port=hbm_w_in
#pragma HLS INTERFACE axis      port=hbm_mask_in
#pragma HLS INTERFACE axis      port=hbm_a_in
#pragma HLS INTERFACE bram      port=scales
#pragma HLS INTERFACE axis      port=y_out
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return

    // Ensure template parameters are visible to HLS
#pragma HLS INLINE off

    constexpr int PAIRS_PER_BEAT = LANES * 2;               // two int4 per byte
    constexpr int BEATS_PER_ROW  = IN_FEATURES / PAIRS_PER_BEAT;
    static_assert(IN_FEATURES % PAIRS_PER_BEAT == 0,
                  "IN_FEATURES must be a multiple of LANES*2");

    // Accumulators partitioned across PEs ------------------------------------
    ap_int<32> accum[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=accum complete dim=1

    // One tile processes TILE_ROWS output rows.
    // If OUT_FEATURES > TILE_ROWS the caller invokes this kernel multiple
    // times (time-multiplex) or replicates the tile spatially.
    const int tile_iterations = (OUT_FEATURES + TILE_ROWS - 1) / TILE_ROWS;

    for (int tile_iter = 0; tile_iter < tile_iterations; ++tile_iter) {
        const int rows_this_iter =
            (tile_iter == tile_iterations - 1)
                ? (OUT_FEATURES - tile_iter * TILE_ROWS)
                : TILE_ROWS;

        // Load per-row scales into local registers ---------------------------
        scale_t row_scale[TILE_ROWS];
#pragma HLS ARRAY_PARTITION variable=row_scale complete dim=1
        for (int r = 0; r < TILE_ROWS; ++r) {
#pragma HLS UNROLL
            row_scale[r] = (r < rows_this_iter) ? scales[r] : scale_t(0);
            accum[r] = 0;
        }

        // Process rows -------------------------------------------------------
    ROWS:
        for (int r = 0; r < rows_this_iter; ++r) {
        COLS:
            for (int b = 0; b < BEATS_PER_ROW; ++b) {
#pragma HLS PIPELINE II=1 style=flp
                packed_w_beat_t   w_beat = hbm_w_in.read();
                packed_a_beat_t   a_beat = hbm_a_in.read();
                sparse_mask_beat_t mask_beat;
                if (USE_SPARSITY) {
                    mask_beat = hbm_mask_in.read();
                } else {
                    mask_beat = ~sparse_mask_beat_t(0); // all ones
                }

                ap_int<32> partial = 0;

            LANES_LOOP:
                for (int lane = 0; lane < LANES; ++lane) {
#pragma HLS UNROLL
                    // Each byte contains two int4 nibbles.
                    ap_uint<8> w_byte = w_beat.range(8 * lane + 7, 8 * lane);
                    ap_int<4> w_lo = w_byte.range(3, 0).v;
                    ap_int<4> w_hi = w_byte.range(7, 4).v;

                    // Each activation byte feeds one lane (one int8 value).
                    // Activations are NOT interleaved; one a_beat holds
                    // exactly LANES consecutive INT8 values.
                    ap_int<8> a_val = a_beat.range(8 * lane + 7, 8 * lane).v;

                    // Two weight pairs per lane byte.
                    ap_uint<2> mask_pair = mask_beat.range(2 * lane + 1, 2 * lane).v;

                    // Pair 0 (low nibble)
                    if (mask_pair[0]) {
                        ap_int<32> prod0 = ap_int<32>(w_lo) * ap_int<32>(a_val);
                        partial += prod0;
                    }
                    // Pair 1 (high nibble)
                    if (mask_pair[1]) {
                        ap_int<32> prod1 = ap_int<32>(w_hi) * ap_int<32>(a_val);
                        partial += prod1;
                    }
                }
                accum[r] += partial;
            }
            // Apply per-row scale and emit
            // In real hardware the scale multiply happens here; in the stub
            // we keep INT32 out and let the caller handle requant.
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
