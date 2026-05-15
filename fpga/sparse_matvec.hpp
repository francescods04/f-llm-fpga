// sparse_matvec.hpp — row-skip controller on top of dense INT4 matvec.
//
// When an output row is all-zeros (due to N:M sparsity), the sparse controller
// skips reading its weight bytes entirely and writes a zero accumulator.
// This saves both HBM bandwidth and compute cycles.
//
// Build:
//   g++ -std=c++17 -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
//       -o /tmp/sparse_matvec_tb fpga/sparse_matvec_tb.cpp && /tmp/sparse_matvec_tb

#ifndef FLLM_SPARSE_MATVEC_HPP
#define FLLM_SPARSE_MATVEC_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include "matvec_int4.hpp"   // for dot_int4_int8, packed_w_beat_t, etc.

#include <cstdint>

// Row mask: 1 = row is dense (non-zero), 0 = row is sparse (skip).
// One bit per output row, packed into uint64_t words for host simplicity.
template<int OUT_FEATURES>
struct RowMask {
    static constexpr int WORDS = (OUT_FEATURES + 63) / 64;
    uint64_t word[WORDS];

    bool is_dense(int row) const {
        int w = row / 64;
        int b = row % 64;
        return (word[w] >> b) & 1ULL;
    }
};

// Sparse matvec wrapper: reads row mask, streams weights only for dense rows.
// The weight stream on the caller side must already be filtered (dense-only).
// The caller also provides the count of dense rows so the tile knows when
// to stop.
template<int IN_FEATURES, int OUT_FEATURES>
void sparse_matvec_tile(
    hls::stream<packed_w_beat_t>& s_w,      // dense rows only
    hls::stream<packed_a_beat_t>& s_a,
    const RowMask<OUT_FEATURES>& row_mask,
    int num_dense_rows,
    const scale_t scales[TILE_ROWS],
    hls::stream<ap_int<32>>& s_y
) {
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INTERFACE mode=axis port=s_w
#pragma HLS INTERFACE mode=axis port=s_a
#pragma HLS INTERFACE mode=axis port=s_y

    static_assert(IN_FEATURES % LANES == 0, "IN_FEATURES must be multiple of LANES");
    constexpr int BEATS_PER_ROW = IN_FEATURES / LANES;

    // Buffer activation once
    packed_a_beat_t a_uram[BEATS_PER_ROW];
    #pragma HLS BIND_STORAGE variable=a_uram type=RAM_1P
    for (int b = 0; b < BEATS_PER_ROW; ++b) {
        a_uram[b] = s_a.read();
    }

    int dense_row_idx = 0;
    for (int r = 0; r < OUT_FEATURES; ++r) {
        if (!row_mask.is_dense(r)) {
            // Sparse row: emit zero without reading weights
            ap_int<32> zero = 0;
            s_y.write(zero);
            continue;
        }
        // Dense row: consume one row of weight beats
        ap_int<32> acc = 0;
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            #pragma HLS PIPELINE II=1
            packed_w_beat_t wb = s_w.read();
            packed_a_beat_t ab = a_uram[b];
            for (int lane = 0; lane < LANES / 2; ++lane) {
                ap_uint<8> w_byte = ap_uint<8>(wb.range(8 * lane + 7, 8 * lane));
                ap_int<4> w_lo = ap_int<4>(ap_uint<4>(w_byte.range(3, 0)));
                ap_int<4> w_hi = ap_int<4>(ap_uint<4>(w_byte.range(7, 4)));
                ap_int<8> a_lo = ap_int<8>(ap_uint<8>(ab.range(16 * lane + 7,  16 * lane + 0)));
                ap_int<8> a_hi = ap_int<8>(ap_uint<8>(ab.range(16 * lane + 15, 16 * lane + 8)));
                acc += ap_mul<32>(w_lo, a_lo) + ap_mul<32>(w_hi, a_hi);
            }
        }
        float scale = scales[dense_row_idx % TILE_ROWS];
        float val = static_cast<float>(static_cast<int>(acc)) * scale;
        s_y.write(static_cast<ap_int<32>>(static_cast<int>(val)));
        dense_row_idx++;
    }
}

#endif  // FLLM_SPARSE_MATVEC_HPP
