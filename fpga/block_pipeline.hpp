// block_pipeline.hpp — one Transformer block as a spatial dataflow pipeline.
//
// This is the architectural heart of the F-LLM FPGA: a single composite kernel
// that chains every operation inside one Transformer block using hls::stream
// and #pragma HLS DATAFLOW.  The hidden state (4096-d INT8) never leaves
// on-chip URAM between stages; only weights, KV cache, and the final output
// touch HBM.
//
// Stages (left-to-right, all connected by FIFO streams):
//
//   in_hidden  -> rmsnorm_stage -> qkv_proj_stage -> rope_stage -> kv_stage
//                                                                     |
//   kv_cache_hbm <-> gqa_attention_stage -> o_proj_stage -> residual_add1
//                                                                     |
//   rmsnorm_mid_stage -> router_stage -> topk_stage -> dispatch_stage
//                                                                     |
//   expert_cache_hbm <-> expert_stage -> combine_stage -> residual_add2
//                                                                     |
//                                                               out_hidden
//
// Every matvec uses the same matvec_int4_tile primitive, time-multiplexed
// across the block.  The FPGA build instantiates K replicas of the tile bank
// to match HBM channel parallelism.
//
// This file is a structural skeleton.  Individual stages are stubbed with
// minimal logic so that host compilation via hls_stubs.hpp succeeds and the
// topology can be validated before any Vitis HLS synthesis.

#ifndef FLLM_BLOCK_PIPELINE_HPP
#define FLLM_BLOCK_PIPELINE_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include "matvec_int4.hpp"

#include <cstdint>

// ---------------------------------------------------------------------------
// Block configuration (template so one .xo can be rebuilt per model)
// ---------------------------------------------------------------------------
template<int HIDDEN, int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM,
         int NUM_EXPERTS, int ACTIVE_EXPERTS, int EXPERT_INNER,
         int LOCAL_WINDOW, int MAX_CTX>
struct BlockShape {
    static constexpr int hidden        = HIDDEN;
    static constexpr int num_heads       = NUM_HEADS;
    static constexpr int num_kv_heads    = NUM_KV_HEADS;
    static constexpr int head_dim        = HEAD_DIM;
    static constexpr int num_experts     = NUM_EXPERTS;
    static constexpr int active_experts  = ACTIVE_EXPERTS;
    static constexpr int expert_inner    = EXPERT_INNER;
    static constexpr int local_window    = LOCAL_WINDOW;
    static constexpr int max_ctx         = MAX_CTX;
    static constexpr int qkv_out         = (num_heads + 2 * num_kv_heads) * head_dim;
};

// ---------------------------------------------------------------------------
// On-chip token-level types
// ---------------------------------------------------------------------------
typedef ap_int<8>  int8_token_t;   // one activation element
typedef ap_int<32> int32_acc_t;   // accumulator / logits
typedef float      float_t;

// One hidden vector = HIDDEN int8 values.  We stream it beat-by-beat.
template<int HIDDEN>
struct HiddenVec {
    static constexpr int BYTES = HIDDEN;
    ap_uint<BYTES * 8> data;
    // Convenience accessors would be added here for real HLS.
};

// ---------------------------------------------------------------------------
// Stage 1: RMSNorm (streaming)
// ---------------------------------------------------------------------------
template<int HIDDEN>
void rmsnorm_stage(
    hls::stream<HiddenVec<HIDDEN>>& in,
    hls::stream<HiddenVec<HIDDEN>>& out,
    const float_t rms_weight[HIDDEN]
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE bram      port=rms_weight

    // TODO: streaming mean-square + LUT rsqrt + per-element multiply.
    // For skeleton: passthrough.
    HiddenVec<HIDDEN> v = in.read();
    out.write(v);
}

// ---------------------------------------------------------------------------
// Stage 2: QKV projection (matvec_int4_tile)
// ---------------------------------------------------------------------------
template<int IN_FEATURES, int OUT_FEATURES>
void qkv_proj_stage(
    hls::stream<HiddenVec<IN_FEATURES>>& in,
    hls::stream<HiddenVec<OUT_FEATURES>>& out,
    hls::stream<packed_w_beat_t>&   w_qkv,
    hls::stream<packed_a_beat_t>&   a_qkv,
    const float_t                   scales_qkv[TILE_ROWS]
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE axis      port=w_qkv
#pragma HLS INTERFACE axis      port=a_qkv
#pragma HLS INTERFACE bram      port=scales_qkv

    // TODO: unpack hidden vector into a_qkv stream, call matvec_int4_tile_dense.
    // For skeleton: passthrough with shape conversion.
    HiddenVec<IN_FEATURES> v = in.read();
    HiddenVec<OUT_FEATURES> r;
    out.write(r);
}

// ---------------------------------------------------------------------------
// Stage 3: RoPE apply (sin/cos table lookup)
// ---------------------------------------------------------------------------
template<int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM>
void rope_stage(
    hls::stream<HiddenVec<(NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM>>& in,
    hls::stream<HiddenVec<(NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM>>& out,
    int position
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out

    // TODO: rotate Q and K using precomputed sin/cos tables in BRAM.
    auto v = in.read();
    out.write(v);
}

// ---------------------------------------------------------------------------
// Stage 4: KV cache write (BFP8 pack -> HBM)
// ---------------------------------------------------------------------------
template<int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM>
void kv_write_stage(
    hls::stream<HiddenVec<(NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM>>& in,
    // HBM port for KV append
    hls::stream<ap_uint<8>>& kv_hbm_out,
    int write_position
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=kv_hbm_out

    // TODO: extract K+V bytes from the Q+K+V vector, then BFP8 pack.
    auto v = in.read();
    // write to kv_hbm_out...
}

// ---------------------------------------------------------------------------
// Stage 5: GQA Attention (Q*K^T + softmax + P*V)
// ---------------------------------------------------------------------------
template<int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM, int LOCAL_WINDOW, int MAX_CTX>
void gqa_attention_stage(
    hls::stream<HiddenVec<(NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM>>& in,
    hls::stream<HiddenVec<NUM_HEADS * HEAD_DIM>>& out,
    // HBM port for KV read
    hls::stream<ap_uint<8>>& kv_hbm_in,
    int current_len
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE axis      port=kv_hbm_in

    // TODO: dot-product attention over cached KV, local window masking,
    // shifted-softmax with LUT exp, scaled by 1/sqrt(head_dim).
    auto v = in.read();
    HiddenVec<NUM_HEADS * HEAD_DIM> r;
    out.write(r);
}

// ---------------------------------------------------------------------------
// Stage 6: O projection (matvec_int4_tile)
// ---------------------------------------------------------------------------
template<int IN_FEATURES, int OUT_FEATURES>
void o_proj_stage(
    hls::stream<HiddenVec<IN_FEATURES>>& in,
    hls::stream<HiddenVec<OUT_FEATURES>>& out,
    hls::stream<packed_w_beat_t>&   w_o,
    hls::stream<packed_a_beat_t>&   a_o,
    const float_t                   scales_o[TILE_ROWS]
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE axis      port=w_o
#pragma HLS INTERFACE axis      port=a_o
#pragma HLS INTERFACE bram      port=scales_o

    // TODO: matvec_int4_tile_dense<IN_FEATURES, OUT_FEATURES>
    auto v = in.read();
    HiddenVec<OUT_FEATURES> r;
    out.write(r);
}

// ---------------------------------------------------------------------------
// Stage 7: Residual add
// ---------------------------------------------------------------------------
template<int HIDDEN>
void residual_add_stage(
    hls::stream<HiddenVec<HIDDEN>>& branch,
    hls::stream<HiddenVec<HIDDEN>>& main,
    hls::stream<HiddenVec<HIDDEN>>& out
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=branch
#pragma HLS INTERFACE axis      port=main
#pragma HLS INTERFACE axis      port=out

    // TODO: element-wise INT8 add with saturation.
    auto b = branch.read();
    auto m = main.read();
    out.write(m);  // skeleton: passthrough main
}

// ---------------------------------------------------------------------------
// Stage 8: Router projection + top-k bitonic sort
// ---------------------------------------------------------------------------
template<int HIDDEN, int NUM_EXPERTS, int ACTIVE_EXPERTS>
void router_topk_stage(
    hls::stream<HiddenVec<HIDDEN>>& in,
    hls::stream<HiddenVec<HIDDEN>>& out_hidden,
    hls::stream<ap_uint<ACTIVE_EXPERTS * 8>>& out_expert_ids,
    hls::stream<ap_uint<ACTIVE_EXPERTS * 16>>& out_gates,
    hls::stream<packed_w_beat_t>&   w_router,
    hls::stream<packed_a_beat_t>&   a_router,
    const float_t                   scales_router[TILE_ROWS]
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out_hidden
#pragma HLS INTERFACE axis      port=out_expert_ids
#pragma HLS INTERFACE axis      port=out_gates
#pragma HLS INTERFACE axis      port=w_router
#pragma HLS INTERFACE axis      port=a_router
#pragma HLS INTERFACE bram      port=scales_router

    // TODO: matvec router, bitonic top-k, softmax gates.
    auto v = in.read();
    out_hidden.write(v);
    ap_uint<ACTIVE_EXPERTS * 8> ids;
    ap_uint<ACTIVE_EXPERTS * 16> gates;
    out_expert_ids.write(ids);
    out_gates.write(gates);
}

// ---------------------------------------------------------------------------
// Stage 9: Expert dispatch / cache controller
// ---------------------------------------------------------------------------
template<int HIDDEN, int NUM_EXPERTS, int ACTIVE_EXPERTS, int EXPERT_INNER>
void expert_dispatch_stage(
    hls::stream<HiddenVec<HIDDEN>>& in,
    hls::stream<ap_uint<ACTIVE_EXPERTS * 8>>& expert_ids,
    hls::stream<ap_uint<ACTIVE_EXPERTS * 16>>& gates,
    // One stream per active expert slot
    hls::stream<HiddenVec<HIDDEN>> expert_in[ACTIVE_EXPERTS],
    hls::stream<float_t>         expert_gate[ACTIVE_EXPERTS],
    // HBM port for cold expert weights
    hls::stream<ap_uint<8>>& expert_hbm_in
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=expert_ids
#pragma HLS INTERFACE axis      port=gates
#pragma HLS INTERFACE axis      port=expert_hbm_in

    // TODO: read expert_ids, check URAM directory, stream tokens to
    // resident experts or issue HBM prefetch for cold experts.
    auto v = in.read();
    for (int i = 0; i < ACTIVE_EXPERTS; ++i) {
        expert_in[i].write(v);
        expert_gate[i].write(float_t(1.0f));
    }
}

// ---------------------------------------------------------------------------
// Stage 10: Expert matvec (up + gate + down per expert)
// ---------------------------------------------------------------------------
template<int HIDDEN, int EXPERT_INNER>
void expert_stage(
    hls::stream<HiddenVec<HIDDEN>>& in,
    hls::stream<HiddenVec<HIDDEN>>& out,
    hls::stream<packed_w_beat_t>&   w_up,
    hls::stream<packed_a_beat_t>&   a_up,
    hls::stream<packed_w_beat_t>&   w_gate,
    hls::stream<packed_a_beat_t>&   a_gate,
    hls::stream<packed_w_beat_t>&   w_down,
    hls::stream<packed_a_beat_t>&   a_down,
    const float_t                   scales_up[TILE_ROWS],
    const float_t                   scales_gate[TILE_ROWS],
    const float_t                   scales_down[TILE_ROWS]
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=in
#pragma HLS INTERFACE axis      port=out
#pragma HLS INTERFACE axis      port=w_up
#pragma HLS INTERFACE axis      port=a_up
#pragma HLS INTERFACE axis      port=w_gate
#pragma HLS INTERFACE axis      port=a_gate
#pragma HLS INTERFACE axis      port=w_down
#pragma HLS INTERFACE axis      port=a_down
#pragma HLS INTERFACE bram      port=scales_up
#pragma HLS INTERFACE bram      port=scales_gate
#pragma HLS INTERFACE bram      port=scales_down

    // TODO: up matvec -> gate matvec -> SiLU LUT -> mul -> down matvec.
    auto v = in.read();
    HiddenVec<HIDDEN> r;
    out.write(r);
}

// ---------------------------------------------------------------------------
// Stage 11: MoE combine (weighted sum of expert outputs)
// ---------------------------------------------------------------------------
template<int HIDDEN, int ACTIVE_EXPERTS>
void combine_stage(
    hls::stream<HiddenVec<HIDDEN>> expert_out[ACTIVE_EXPERTS],
    hls::stream<float_t>         expert_gate[ACTIVE_EXPERTS],
    hls::stream<HiddenVec<HIDDEN>>& out
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis      port=out

    // TODO: accumulate gate[i] * expert_out[i], requantize to INT8.
    HiddenVec<HIDDEN> r;
    out.write(r);
}

// ---------------------------------------------------------------------------
// Top-level block pipeline
// ---------------------------------------------------------------------------
template<typename SHAPE>
void block_pipeline(
    hls::stream<HiddenVec<SHAPE::hidden>>& in_hidden,
    hls::stream<HiddenVec<SHAPE::hidden>>& out_hidden,
    // Weight streams (one per matvec inside the block)
    hls::stream<packed_w_beat_t>& w_qkv,
    hls::stream<packed_a_beat_t>& a_qkv,
    hls::stream<packed_w_beat_t>& w_o,
    hls::stream<packed_a_beat_t>& a_o,
    hls::stream<packed_w_beat_t>& w_router,
    hls::stream<packed_a_beat_t>& a_router,
    hls::stream<packed_w_beat_t>& w_expert_up,
    hls::stream<packed_a_beat_t>& a_expert_up,
    hls::stream<packed_w_beat_t>& w_expert_gate,
    hls::stream<packed_a_beat_t>& a_expert_gate,
    hls::stream<packed_w_beat_t>& w_expert_down,
    hls::stream<packed_a_beat_t>& a_expert_down,
    // Scale tables (BRAM)
    const float_t scales_qkv[TILE_ROWS],
    const float_t scales_o[TILE_ROWS],
    const float_t scales_router[TILE_ROWS],
    const float_t scales_expert_up[TILE_ROWS],
    const float_t scales_expert_gate[TILE_ROWS],
    const float_t scales_expert_down[TILE_ROWS],
    // HBM ports for KV and expert cold storage
    hls::stream<ap_uint<8>>& kv_hbm_in,
    hls::stream<ap_uint<8>>& kv_hbm_out,
    hls::stream<ap_uint<8>>& expert_hbm_in,
    // Control
    int position,
    int current_len
) {
#pragma HLS DATAFLOW
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INTERFACE axis      port=in_hidden
#pragma HLS INTERFACE axis      port=out_hidden
#pragma HLS INTERFACE axis      port=kv_hbm_in
#pragma HLS INTERFACE axis      port=kv_hbm_out
#pragma HLS INTERFACE axis      port=expert_hbm_in

    // Internal FIFOs (depth sized for one token)
    hls::stream<HiddenVec<SHAPE::hidden>> s_rms_in;
    hls::stream<HiddenVec<SHAPE::hidden>> s_rms_out;
    hls::stream<HiddenVec<SHAPE::qkv_out>> s_qkv;
    hls::stream<HiddenVec<SHAPE::qkv_out>> s_rope;
    hls::stream<HiddenVec<SHAPE::num_heads * SHAPE::head_dim>> s_attn;
    hls::stream<HiddenVec<SHAPE::hidden>> s_o_proj;
    hls::stream<HiddenVec<SHAPE::hidden>> s_res1;
    hls::stream<HiddenVec<SHAPE::hidden>> s_rms_mid;
    hls::stream<HiddenVec<SHAPE::hidden>> s_router_hidden;
    hls::stream<ap_uint<SHAPE::active_experts * 8>> s_expert_ids;
    hls::stream<ap_uint<SHAPE::active_experts * 16>> s_gates;
    hls::stream<HiddenVec<SHAPE::hidden>> s_expert_in[SHAPE::active_experts];
    hls::stream<float_t> s_expert_gate[SHAPE::active_experts];
    hls::stream<HiddenVec<SHAPE::hidden>> s_expert_out[SHAPE::active_experts];
    hls::stream<HiddenVec<SHAPE::hidden>> s_combine;

#pragma HLS STREAM variable=s_rms_in       depth=2
#pragma HLS STREAM variable=s_rms_out      depth=2
#pragma HLS STREAM variable=s_qkv         depth=2
#pragma HLS STREAM variable=s_rope        depth=2
#pragma HLS STREAM variable=s_attn        depth=2
#pragma HLS STREAM variable=s_o_proj      depth=2
#pragma HLS STREAM variable=s_res1        depth=2
#pragma HLS STREAM variable=s_rms_mid     depth=2
#pragma HLS STREAM variable=s_router_hidden depth=2
#pragma HLS STREAM variable=s_expert_ids  depth=2
#pragma HLS STREAM variable=s_gates       depth=2
#pragma HLS STREAM variable=s_combine     depth=2

    // Read one token and buffer it for the two residual adds.
    HiddenVec<SHAPE::hidden> token = in_hidden.read();
    s_rms_in.write(token);

    rmsnorm_stage<SHAPE::hidden>(s_rms_in, s_rms_out, nullptr);  // TODO: pass real scale
    qkv_proj_stage<SHAPE::hidden, SHAPE::qkv_out>(
        s_rms_out, s_qkv, w_qkv, a_qkv, scales_qkv);
    rope_stage<SHAPE::num_heads, SHAPE::num_kv_heads, SHAPE::head_dim>(
        s_qkv, s_rope, position);

    kv_write_stage<SHAPE::num_heads, SHAPE::num_kv_heads, SHAPE::head_dim>(
        s_rope, kv_hbm_out, current_len);

    gqa_attention_stage<SHAPE::num_heads, SHAPE::num_kv_heads, SHAPE::head_dim,
                         SHAPE::local_window, SHAPE::max_ctx>(
        s_rope, s_attn, kv_hbm_in, current_len);

    o_proj_stage<SHAPE::num_heads * SHAPE::head_dim, SHAPE::hidden>(
        s_attn, s_o_proj, w_o, a_o, scales_o);

    // First residual: token + attention output
    hls::stream<HiddenVec<SHAPE::hidden>> s_res1_branch;
    s_res1_branch.write(token);
    residual_add_stage<SHAPE::hidden>(s_o_proj, s_res1_branch, s_res1);

    rmsnorm_stage<SHAPE::hidden>(s_res1, s_rms_mid, nullptr);  // TODO: real scale

    router_topk_stage<SHAPE::hidden, SHAPE::num_experts, SHAPE::active_experts>(
        s_rms_mid, s_router_hidden, s_expert_ids, s_gates,
        w_router, a_router, scales_router);

    expert_dispatch_stage<SHAPE::hidden, SHAPE::num_experts, SHAPE::active_experts,
                          SHAPE::expert_inner>(
        s_router_hidden, s_expert_ids, s_gates,
        s_expert_in, s_expert_gate, expert_hbm_in);

    // ACTIVE_EXPERTS parallel expert stages
    for (int e = 0; e < SHAPE::active_experts; ++e) {
#pragma HLS UNROLL
        expert_stage<SHAPE::hidden, SHAPE::expert_inner>(
            s_expert_in[e], s_expert_out[e],
            w_expert_up, a_expert_up,
            w_expert_gate, a_expert_gate,
            w_expert_down, a_expert_down,
            scales_expert_up, scales_expert_gate, scales_expert_down);
    }

    combine_stage<SHAPE::hidden, SHAPE::active_experts>(
        s_expert_out, s_expert_gate, s_combine);

    // Final residual: token + MoE output
    hls::stream<HiddenVec<SHAPE::hidden>> s_res2_branch;
    s_res2_branch.write(token);
    residual_add_stage<SHAPE::hidden>(s_combine, s_res2_branch, out_hidden);
}

#endif  // FLLM_BLOCK_PIPELINE_HPP
