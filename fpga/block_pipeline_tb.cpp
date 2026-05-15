// block_pipeline_tb.cpp — topology compile-check for block_pipeline.hpp.
//
// This file does NOT run the pipeline (stages are stubbed and would deadlock
// without real data).  It only proves that block_pipeline.hpp compiles with
// the Qwen3-A3B shape when linked against hls_stubs.hpp.
//
// Build:
//   g++ -O2 -std=c++17 -Wno-unknown-pragmas -I. \
//       fpga/block_pipeline_tb.cpp -o /tmp/block_pipeline_tb
// Run:
//   /tmp/block_pipeline_tb

#include <cstdio>
#include "fpga/hls_stubs.hpp"
#include "fpga/block_pipeline.hpp"

using QwenBlock = BlockShape<
    4096,   // HIDDEN
    32,     // NUM_HEADS
    8,      // NUM_KV_HEADS
    128,    // HEAD_DIM
    128,    // NUM_EXPERTS
    8,      // ACTIVE_EXPERTS
    1408,   // EXPERT_INNER
    512,    // LOCAL_WINDOW
    4096    // MAX_CTX
>;

// We only need to instantiate the template to verify type correctness.
// The function pointer is never dereferenced.
void (*pipeline_inst)(
    hls::stream<HiddenVec<4096>>&, hls::stream<HiddenVec<4096>>&,
    hls::stream<packed_w_beat_t>&, hls::stream<packed_a_beat_t>&,
    hls::stream<packed_w_beat_t>&, hls::stream<packed_a_beat_t>&,
    hls::stream<packed_w_beat_t>&, hls::stream<packed_a_beat_t>&,
    hls::stream<packed_w_beat_t>&, hls::stream<packed_a_beat_t>&,
    hls::stream<packed_w_beat_t>&, hls::stream<packed_a_beat_t>&,
    hls::stream<packed_w_beat_t>&, hls::stream<packed_a_beat_t>&,
    const float_t*, const float_t*, const float_t*,
    const float_t*, const float_t*, const float_t*,
    hls::stream<ap_uint<8>>&, hls::stream<ap_uint<8>>&, hls::stream<ap_uint<8>>&,
    int, int
) = &block_pipeline<QwenBlock>;

int main() {
    (void)pipeline_inst;
    std::printf("block_pipeline<QwenBlock> template instantiates and compiles\n");
    return 0;
}
