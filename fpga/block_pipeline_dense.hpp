// block_pipeline_dense.hpp — one DENSE Transformer block with real FPGA kernels.
//
// This is the first fully-linked block pipeline.  It replaces the MoE path
// with a dense SwiGLU MLP so the datapath is simpler while still exercising
// every real kernel:
//   matvec_int4_tile_dense, rmsnorm_engine, silu_lut, softmax_engine, rope_engine.
//
// Stages:
//   in_hidden -> rmsnorm -> qkv_proj -> rope -> kv_write -> attention ->
//   o_proj -> res_add -> rmsnorm -> mlp_up -> silu -> mlp_gate -> mul ->
//   mlp_down -> res_add -> out_hidden
//
// All matvecs use the same INT4 tile primitive.  The hidden state stays in
// URAM FIFOs between stages.  Only weights and KV cache touch HBM.

#ifndef FLLM_BLOCK_PIPELINE_DENSE_HPP
#define FLLM_BLOCK_PIPELINE_DENSE_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include "matvec_int4.hpp"
#include "rmsnorm_engine.hpp"
#include "silu_lut.hpp"
#include "softmax_engine.hpp"
#include "rope_engine.hpp"

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// Dense block shape (no MoE, no expert dispatch)
// ---------------------------------------------------------------------------
template<int HIDDEN, int NUM_HEADS, int NUM_KV_HEADS, int HEAD_DIM,
         int MLP_RATIO, int LOCAL_WINDOW, int MAX_CTX>
struct DenseBlockShape {
    static constexpr int hidden       = HIDDEN;
    static constexpr int num_heads      = NUM_HEADS;
    static constexpr int num_kv_heads   = NUM_KV_HEADS;
    static constexpr int head_dim       = HEAD_DIM;
    static constexpr int mlp_ratio      = MLP_RATIO;
    static constexpr int mlp_inner      = HIDDEN * MLP_RATIO;
    static constexpr int local_window   = LOCAL_WINDOW;
    static constexpr int max_ctx        = MAX_CTX;
    static constexpr int qkv_out        = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
    static constexpr int o_proj_in      = NUM_HEADS * HEAD_DIM;
};

// ---------------------------------------------------------------------------
// Helper: pack float activation vector into INT8 A-buffer (URAM)
// ---------------------------------------------------------------------------
template<int COUNT>
void pack_float_to_int8_uram(
    hls::stream<float>& float_in,
    int8_t int8_buf[COUNT]
) {
#pragma HLS INLINE
    for (int i = 0; i < COUNT; ++i) {
#pragma HLS PIPELINE II=1
        float f = float_in.read();
        // Simple rounding; in real HW this is part of the quant unit
        int8_buf[i] = static_cast<int8_t>(f);
    }
}

// ---------------------------------------------------------------------------
// Helper: stream one INT8 buffer (multiple times) as packed A beats
// ---------------------------------------------------------------------------
template<int COUNT>
void stream_int8_buf_as_a_beats(
    const int8_t int8_buf[COUNT],
    hls::stream<packed_a_beat_t>& a_out
) {
#pragma HLS INLINE
    static_assert(COUNT % LANES == 0, "COUNT must be multiple of LANES");
    constexpr int BEATS = COUNT / LANES;
    for (int b = 0; b < BEATS; ++b) {
#pragma HLS PIPELINE II=1
        packed_a_beat_t ab;
        for (int j = 0; j < ab.LIMBS; ++j) ab.limb[j] = 0;
        for (int lane = 0; lane < LANES / 2; ++lane) {
            int idx = b * LANES + lane * 2;
            int8_t i_lo = int8_buf[idx];
            int8_t i_hi = int8_buf[idx + 1];
            ab.set_range(16 * lane + 7,  16 * lane + 0,
                         ap_uint<A_BEAT_BYTES * 8>(static_cast<uint8_t>(i_lo)));
            ab.set_range(16 * lane + 15, 16 * lane + 8,
                         ap_uint<A_BEAT_BYTES * 8>(static_cast<uint8_t>(i_hi)));
        }
        a_out.write(ab);
    }
}

// ---------------------------------------------------------------------------
// Helper: consume INT32 y-stream and emit float output stream
// ---------------------------------------------------------------------------
template<int COUNT>
void unpack_int32_to_float_stream(
    hls::stream<ap_int<32>>& y_in,
    hls::stream<float>& float_out,
    float out_scale
) {
#pragma HLS INLINE
    for (int i = 0; i < COUNT; ++i) {
#pragma HLS PIPELINE II=1
        ap_int<32> v = y_in.read();
        float f = static_cast<float>(static_cast<int>(v)) * out_scale;
        float_out.write(f);
    }
}

// ---------------------------------------------------------------------------
// Generic dense matvec: float_in -> (pack) -> matvec tiles -> (unpack) -> float_out
//
// Assumes:
//   - IN_FEATURES is a multiple of LANES
//   - OUT_FEATURES is a multiple of TILE_ROWS
// ---------------------------------------------------------------------------
template<int IN_FEATURES, int OUT_FEATURES>
void matvec_dense(
    hls::stream<float>& float_in,
    hls::stream<float>& float_out,
    hls::stream<packed_w_beat_t>& w_in,
    const scale_t scales[TILE_ROWS],
    float out_scale
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE axis port=float_in
#pragma HLS INTERFACE axis port=float_out
#pragma HLS INTERFACE axis port=w_in

    static_assert(IN_FEATURES % LANES == 0, "IN_FEATURES must be multiple of LANES");

    // 1. Buffer activation in URAM as INT8
    int8_t a_uram[IN_FEATURES];
    #pragma HLS BIND_STORAGE variable=a_uram type=RAM_1P
    pack_float_to_int8_uram<IN_FEATURES>(float_in, a_uram);

    // 2. Stream activations as packed beats
    hls::stream<packed_a_beat_t> a_stream(256);
    stream_int8_buf_as_a_beats<IN_FEATURES>(a_uram, a_stream);

    // 3. Call the dense tile (consumes full weight stream internally)
    hls::stream<ap_int<32>> y_stream(OUT_FEATURES + 16);
    matvec_int4_tile_dense<IN_FEATURES, OUT_FEATURES>(
        w_in, a_stream, scales, y_stream);

    // 4. Convert INT32 -> float
    unpack_int32_to_float_stream<OUT_FEATURES>(y_stream, float_out, out_scale);
}

// ---------------------------------------------------------------------------
// Stage: RMSNorm
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_rmsnorm(
    hls::stream<float>& in_stream,
    hls::stream<float>& out_stream,
    const float rms_weight[SHAPE::hidden],
    const DefaultRSqrtLUT& rsqrt_lut
) {
    rmsnorm_tile<SHAPE::hidden>(in_stream, out_stream, rms_weight, rsqrt_lut);
}

// ---------------------------------------------------------------------------
// Stage: QKV projection
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_qkv_proj(
    hls::stream<float>& in_stream,
    hls::stream<float>& out_stream,
    hls::stream<packed_w_beat_t>& w_qkv,
    const scale_t scales_qkv[TILE_ROWS],
    float out_scale
) {
    matvec_dense<SHAPE::hidden, SHAPE::qkv_out>(
        in_stream, out_stream, w_qkv, scales_qkv, out_scale);
}

// ---------------------------------------------------------------------------
// Stage: RoPE (position-aware)
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_rope(
    hls::stream<float>& qkv_in,
    hls::stream<float>& q_out,
    hls::stream<float>& k_out,
    hls::stream<float>& v_out,
    const RoPETables<SHAPE::max_ctx, SHAPE::head_dim>& rope_tables,
    int position
) {
#pragma HLS INLINE off
    constexpr int num_q_heads = SHAPE::num_heads;
    constexpr int num_kv_heads = SHAPE::num_kv_heads;
    constexpr int head_dim = SHAPE::head_dim;

    hls::stream<float> q_in(4096);
    hls::stream<float> k_in(4096);

    // Split QKV into three streams
    for (int i = 0; i < num_q_heads * head_dim; ++i) {
        q_in.write(qkv_in.read());
    }
    for (int i = 0; i < num_kv_heads * head_dim; ++i) {
        k_in.write(qkv_in.read());
    }
    for (int i = 0; i < num_kv_heads * head_dim; ++i) {
        v_out.write(qkv_in.read());
    }

    // Apply RoPE via the engine tile
    rope_tile<num_q_heads, num_kv_heads, head_dim, SHAPE::max_ctx>(
        q_in, k_in, q_out, k_out, rope_tables, position);
}

// ---------------------------------------------------------------------------
// Stage: KV cache write (current token -> HBM)
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_kv_write(
    hls::stream<float>& k_in,
    hls::stream<float>& v_in,
    // For the testbench/skeleton we store KV in a local URAM buffer
    float kv_cache_k[SHAPE::max_ctx * SHAPE::num_kv_heads * SHAPE::head_dim],
    float kv_cache_v[SHAPE::max_ctx * SHAPE::num_kv_heads * SHAPE::head_dim],
    int position
) {
#pragma HLS INLINE off
    constexpr int stride = SHAPE::num_kv_heads * SHAPE::head_dim;
    int base = position * stride;
    for (int i = 0; i < stride; ++i) {
        kv_cache_k[base + i] = k_in.read();
    }
    for (int i = 0; i < stride; ++i) {
        kv_cache_v[base + i] = v_in.read();
    }
}

// ---------------------------------------------------------------------------
// Stage: GQA Attention
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_attention(
    hls::stream<float>& q_in,
    hls::stream<float>& out_stream,
    const float kv_cache_k[SHAPE::max_ctx * SHAPE::num_kv_heads * SHAPE::head_dim],
    const float kv_cache_v[SHAPE::max_ctx * SHAPE::num_kv_heads * SHAPE::head_dim],
    int current_len
) {
#pragma HLS INLINE off
    constexpr int num_heads = SHAPE::num_heads;
    constexpr int num_kv_heads = SHAPE::num_kv_heads;
    constexpr int head_dim = SHAPE::head_dim;
    constexpr int q_heads_per_kv = num_heads / num_kv_heads;
    constexpr int kv_stride = num_kv_heads * head_dim;

    // Read all Q heads
    float q_buf[num_heads * head_dim];
    for (int i = 0; i < num_heads * head_dim; ++i) {
        q_buf[i] = q_in.read();
    }

    // Per-head attention
    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / q_heads_per_kv;
        float scores[SHAPE::max_ctx];
        #pragma HLS BIND_STORAGE variable=scores type=RAM_1P

        // 1. Q · K^T scores for all past positions
        float max_score = -1e30f;
        for (int pos = 0; pos < current_len; ++pos) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float qv = q_buf[h * head_dim + d];
                float kv = kv_cache_k[pos * kv_stride + kv_h * head_dim + d];
                dot += qv * kv;
            }
            // Scale by 1/sqrt(head_dim)
            dot *= 1.0f / std::sqrt(static_cast<float>(head_dim));
            scores[pos] = dot;
            if (dot > max_score) max_score = dot;
        }

        // 2. Softmax over scores (manual 3-pass, variable length)
        float sum_exp = 0.0f;
        for (int pos = 0; pos < current_len; ++pos) {
            scores[pos] = std::exp(scores[pos] - max_score);
            sum_exp += scores[pos];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int pos = 0; pos < current_len; ++pos) {
            scores[pos] *= inv_sum;
        }

        // 3. Weighted sum over V
        float out_head[head_dim];
        for (int d = 0; d < head_dim; ++d) out_head[d] = 0.0f;

        for (int pos = 0; pos < current_len; ++pos) {
            float p = scores[pos];
            for (int d = 0; d < head_dim; ++d) {
                float vv = kv_cache_v[pos * kv_stride + kv_h * head_dim + d];
                out_head[d] += p * vv;
            }
        }

        for (int d = 0; d < head_dim; ++d) {
            out_stream.write(out_head[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage: O projection
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_o_proj(
    hls::stream<float>& in_stream,
    hls::stream<float>& out_stream,
    hls::stream<packed_w_beat_t>& w_o,
    const scale_t scales_o[TILE_ROWS],
    float out_scale
) {
    matvec_dense<SHAPE::o_proj_in, SHAPE::hidden>(
        in_stream, out_stream, w_o, scales_o, out_scale);
}

// ---------------------------------------------------------------------------
// Stage: residual add
// ---------------------------------------------------------------------------
template<int COUNT>
void stage_residual_add_stream(
    hls::stream<float>& main_stream,
    hls::stream<float>& res_stream,
    hls::stream<float>& out_stream
) {
    for (int i = 0; i < COUNT; ++i) {
        #pragma HLS PIPELINE II=1
        float a = main_stream.read();
        float b = res_stream.read();
        out_stream.write(a + b);
    }
}

template<int COUNT>
void stage_residual_add_buf(
    hls::stream<float>& main_stream,
    const float res_buf[COUNT],
    hls::stream<float>& out_stream
) {
    for (int i = 0; i < COUNT; ++i) {
        #pragma HLS PIPELINE II=1
        float a = main_stream.read();
        out_stream.write(a + res_buf[i]);
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP proper with forked input
// ---------------------------------------------------------------------------
template<typename SHAPE>
void stage_mlp_swiglu(
    hls::stream<float>& in_stream,
    hls::stream<float>& out_stream,
    hls::stream<packed_w_beat_t>& w_up,
    hls::stream<packed_w_beat_t>& w_gate,
    hls::stream<packed_w_beat_t>& w_down,
    const scale_t scales_up[TILE_ROWS],
    const scale_t scales_gate[TILE_ROWS],
    const scale_t scales_down[TILE_ROWS],
    const SiLULUT& silu_lut,
    float scale_up, float scale_gate, float scale_down
) {
#pragma HLS INLINE off

    // Buffer input so we can fork to both up and gate
    float in_buf[SHAPE::hidden];
    #pragma HLS BIND_STORAGE variable=in_buf type=RAM_1P
    for (int i = 0; i < SHAPE::hidden; ++i) {
        in_buf[i] = in_stream.read();
    }

    // --- up projection ---
    hls::stream<float> up_in(4096);
    for (int i = 0; i < SHAPE::hidden; ++i) up_in.write(in_buf[i]);
    hls::stream<float> up_out(8192);
    matvec_dense<SHAPE::hidden, SHAPE::mlp_inner>(
        up_in, up_out, w_up, scales_up, scale_up);

    // --- gate projection ---
    hls::stream<float> gate_in(4096);
    for (int i = 0; i < SHAPE::hidden; ++i) gate_in.write(in_buf[i]);
    hls::stream<float> gate_out(8192);
    matvec_dense<SHAPE::hidden, SHAPE::mlp_inner>(
        gate_in, gate_out, w_gate, scales_gate, scale_gate);

    // --- SiLU(gate) * up -> down projection ---
    hls::stream<float> silu_gate(8192);
    for (int i = 0; i < SHAPE::mlp_inner; ++i) {
        #pragma HLS PIPELINE II=1
        float x = gate_out.read();
        float xc = x;
        if (xc < silu_lut.lut_min) xc = silu_lut.lut_min;
        if (xc >= silu_lut.lut_max - silu_lut.step) xc = silu_lut.lut_max - silu_lut.step;
        float idx_f = (xc - silu_lut.lut_min) / silu_lut.step;
        int idx_lo = static_cast<int>(std::floor(idx_f));
        if (idx_lo < 0) idx_lo = 0;
        if (idx_lo >= silu_lut.num_entries) idx_lo = silu_lut.num_entries - 1;
        float frac = idx_f - static_cast<float>(idx_lo);
        float lo = silu_lut.table[idx_lo];
        float hi = silu_lut.table[idx_lo + 1];
        silu_gate.write(lo + (hi - lo) * frac);
    }

    hls::stream<float> mul_out(8192);
    for (int i = 0; i < SHAPE::mlp_inner; ++i) {
        #pragma HLS PIPELINE II=1
        float u = up_out.read();
        float g = silu_gate.read();
        mul_out.write(u * g);
    }

    // --- down projection ---
    matvec_dense<SHAPE::mlp_inner, SHAPE::hidden>(
        mul_out, out_stream, w_down, scales_down, scale_down);
}

// ---------------------------------------------------------------------------
// Top-level dense block
// ---------------------------------------------------------------------------
template<typename SHAPE>
void block_pipeline_dense(
    hls::stream<float>& in_hidden,
    hls::stream<float>& out_hidden,
    // Weight streams (packed INT4)
    hls::stream<packed_w_beat_t>& w_qkv,
    hls::stream<packed_w_beat_t>& w_o,
    hls::stream<packed_w_beat_t>& w_up,
    hls::stream<packed_w_beat_t>& w_gate,
    hls::stream<packed_w_beat_t>& w_down,
    // Scales (BRAM)
    const scale_t scales_qkv[TILE_ROWS],
    const scale_t scales_o[TILE_ROWS],
    const scale_t scales_up[TILE_ROWS],
    const scale_t scales_gate[TILE_ROWS],
    const scale_t scales_down[TILE_ROWS],
    // RMSNorm weights
    const float rms_weight_1[SHAPE::hidden],
    const float rms_weight_2[SHAPE::hidden],
    // RoPE tables
    const RoPETables<SHAPE::max_ctx, SHAPE::head_dim>& rope_tables,
    // SiLU LUT
    const SiLULUT& silu_lut,
    // KV cache URAM (to be replaced by HBM in real build)
    float kv_cache_k[SHAPE::max_ctx * SHAPE::num_kv_heads * SHAPE::head_dim],
    float kv_cache_v[SHAPE::max_ctx * SHAPE::num_kv_heads * SHAPE::head_dim],
    // Control
    int position,
    int current_len
) {
#pragma HLS DATAFLOW
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INTERFACE axis      port=in_hidden
#pragma HLS INTERFACE axis      port=out_hidden
#pragma HLS INTERFACE axis      port=w_qkv
#pragma HLS INTERFACE axis      port=w_o
#pragma HLS INTERFACE axis      port=w_up
#pragma HLS INTERFACE axis      port=w_gate
#pragma HLS INTERFACE axis      port=w_down

    // Internal FIFOs between stages (depth must cover largest vector = qkv_out)
    hls::stream<float> s_rms1_out(4096);
    hls::stream<float> s_qkv_out(4096);
    hls::stream<float> s_rope_q(4096);
    hls::stream<float> s_rope_k(4096);
    hls::stream<float> s_rope_v(4096);
    hls::stream<float> s_attn_out(4096);
    hls::stream<float> s_o_proj_out(4096);
    hls::stream<float> s_res1_out(4096);
    hls::stream<float> s_rms2_out(4096);

    // Buffer the residual token before it forks
    float token_buf[SHAPE::hidden];
    #pragma HLS BIND_STORAGE variable=token_buf type=RAM_1P

    hls::stream<float> token_stream1(4096);
    for (int i = 0; i < SHAPE::hidden; ++i) {
        token_buf[i] = in_hidden.read();
        token_stream1.write(token_buf[i]);
    }

    // Stage 1: RMSNorm
    DefaultRSqrtLUT rsqrt_lut;
    stage_rmsnorm<SHAPE>(token_stream1, s_rms1_out, rms_weight_1, rsqrt_lut);

    // Stage 2: QKV projection
    stage_qkv_proj<SHAPE>(s_rms1_out, s_qkv_out, w_qkv, scales_qkv, 1.0f);

    // Stage 3: RoPE
    stage_rope<SHAPE>(s_qkv_out, s_rope_q, s_rope_k, s_rope_v,
                      rope_tables, position);

    // Stage 4: KV write
    stage_kv_write<SHAPE>(s_rope_k, s_rope_v, kv_cache_k, kv_cache_v, position);

    // Stage 5: Attention
    stage_attention<SHAPE>(s_rope_q, s_attn_out, kv_cache_k, kv_cache_v, current_len);

    // Stage 6: O projection
    stage_o_proj<SHAPE>(s_attn_out, s_o_proj_out, w_o, scales_o, 1.0f);

    // Stage 7: Residual add 1 (attention branch)
    stage_residual_add_buf<SHAPE::hidden>(s_o_proj_out, token_buf, s_res1_out);

    // Stage 8: RMSNorm 2
    stage_rmsnorm<SHAPE>(s_res1_out, s_rms2_out, rms_weight_2, rsqrt_lut);

    // Stage 9: MLP SwiGLU
    hls::stream<float> s_mlp_out(4096);
    stage_mlp_swiglu<SHAPE>(
        s_rms2_out, s_mlp_out,
        w_up, w_gate, w_down,
        scales_up, scales_gate, scales_down,
        silu_lut, 1.0f, 1.0f, 1.0f);

    // Stage 10: Residual add 2 (MLP branch)
    stage_residual_add_buf<SHAPE::hidden>(s_mlp_out, token_buf, out_hidden);
}

#endif  // FLLM_BLOCK_PIPELINE_DENSE_HPP
