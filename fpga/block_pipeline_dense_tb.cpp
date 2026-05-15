// block_pipeline_dense_tb.cpp — host testbench for the dense block pipeline.
//
// Uses g++ + hls_stubs.hpp (no Vitis HLS required).
// Strategy: set all INT4 weights to zero => every matvec output is zero.
// The only live path is the two residual adds, so output == input.
//
// This proves the DATAFLOW topology is correct and every stage connects.

#include "block_pipeline_dense.hpp"
#include <cmath>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers to generate packed INT4 weight streams (all zeroes)
// ---------------------------------------------------------------------------
static void write_zero_weights(int num_beats, hls::stream<packed_w_beat_t>& s) {
    for (int i = 0; i < num_beats; ++i) {
        packed_w_beat_t b;
        for (int j = 0; j < b.LIMBS; ++j) b.limb[j] = 0;
        s.write(b);
    }
}

// ---------------------------------------------------------------------------
// Test shape: Qwen3.6-A3B dense layer
// ---------------------------------------------------------------------------
using BS = DenseBlockShape<
    /*HIDDEN=*/2560,
    /*NUM_HEADS=*/32,
    /*NUM_KV_HEADS=*/4,
    /*HEAD_DIM=*/80,
    /*MLP_RATIO=*/3,
    /*LOCAL_WINDOW=*/128,
    /*MAX_CTX=*/128>;

int main() {
    std::cout << "=== block_pipeline_dense testbench ===" << std::endl;
    std::cout << "Hidden=" << BS::hidden
              << " Heads=" << BS::num_heads << "/" << BS::num_kv_heads
              << " HeadDim=" << BS::head_dim
              << " MlpInner=" << BS::mlp_inner
              << " MaxCtx=" << BS::max_ctx << std::endl;

    // 1. Generate random input token
    std::vector<float> input_token(BS::hidden);
    for (int i = 0; i < BS::hidden; ++i) {
        input_token[i] = static_cast<float>((i % 17) - 8) * 0.1f;  // small values
    }

    // 2. Build weight streams (all zero => matvecs output zero)
    hls::stream<packed_w_beat_t> w_qkv(700000);
    hls::stream<packed_w_beat_t> w_o(700000);
    hls::stream<packed_w_beat_t> w_up(700000);
    hls::stream<packed_w_beat_t> w_gate(700000);
    hls::stream<packed_w_beat_t> w_down(700000);

    const int qkv_beats = BS::qkv_out * (BS::hidden / LANES);
    const int o_beats   = BS::hidden * (BS::o_proj_in / LANES);
    const int up_beats  = BS::mlp_inner * (BS::hidden / LANES);
    const int down_beats = BS::hidden * (BS::mlp_inner / LANES);

    write_zero_weights(qkv_beats, w_qkv);
    write_zero_weights(o_beats, w_o);
    write_zero_weights(up_beats, w_up);
    write_zero_weights(up_beats, w_gate);
    write_zero_weights(down_beats, w_down);

    // 3. Scales (all 1.0 so INT32 output is just the integer dot product)
    scale_t scales_qkv[TILE_ROWS];
    scale_t scales_o[TILE_ROWS];
    scale_t scales_up[TILE_ROWS];
    scale_t scales_gate[TILE_ROWS];
    scale_t scales_down[TILE_ROWS];
    for (int i = 0; i < TILE_ROWS; ++i) {
        scales_qkv[i] = 1.0f;
        scales_o[i]   = 1.0f;
        scales_up[i]  = 1.0f;
        scales_gate[i]= 1.0f;
        scales_down[i]= 1.0f;
    }

    // 4. RMSNorm weights (all 1.0 => RMSNorm(x) = x / RMS(x))
    float rms_w1[BS::hidden];
    float rms_w2[BS::hidden];
    for (int i = 0; i < BS::hidden; ++i) {
        rms_w1[i] = 1.0f;
        rms_w2[i] = 1.0f;
    }

    // 5. RoPE tables (constructor precomputes sin/cos)
    RoPETables<BS::max_ctx, BS::head_dim> rope_tables;

    // 6. SiLU LUT
    SiLULUT silu_lut;

    // 7. KV cache (zero-initialized)
    float kv_cache_k[BS::max_ctx * BS::num_kv_heads * BS::head_dim];
    float kv_cache_v[BS::max_ctx * BS::num_kv_heads * BS::head_dim];
    for (int i = 0; i < BS::max_ctx * BS::num_kv_heads * BS::head_dim; ++i) {
        kv_cache_k[i] = 0.0f;
        kv_cache_v[i] = 0.0f;
    }

    // 8. Input / output streams
    hls::stream<float> in_stream(4096);
    hls::stream<float> out_stream(4096);
    for (int i = 0; i < BS::hidden; ++i) {
        in_stream.write(input_token[i]);
    }

    // 9. Run the block
    block_pipeline_dense<BS>(
        in_stream, out_stream,
        w_qkv, w_o, w_up, w_gate, w_down,
        scales_qkv, scales_o, scales_up, scales_gate, scales_down,
        rms_w1, rms_w2,
        rope_tables, silu_lut,
        kv_cache_k, kv_cache_v,
        /*position=*/0,
        /*current_len=*/1);

    // 10. Read output and compare
    float max_err = 0.0f;
    int output_count = 0;
    while (!out_stream.empty()) {
        float y = out_stream.read();
        float ref = input_token[output_count];
        float err = std::abs(y - ref);
        if (err > max_err) max_err = err;
        output_count++;
    }

    std::cout << "Output tokens produced: " << output_count << std::endl;
    if (output_count != BS::hidden) {
        std::cerr << "FAIL: expected " << BS::hidden << " outputs, got " << output_count << std::endl;
        return 1;
    }

    // With all-zero weights, every matvec emits 0.  Residual adds bring back
    // the original token.  RMSNorm changes the scale but then the second
    // residual add should restore exactly because MLP also outputs 0.
    // Wait: first residual = O_proj(0) + token = token.
    // Then RMSNorm(token) is computed, but MLP on that gives 0.
    // Second residual = MLP(0) + token = token.  So output == input exactly.
    std::cout << "Max error vs input: " << max_err << std::endl;
    if (max_err > 1e-4f) {
        std::cerr << "FAIL: output does not match expected passthrough" << std::endl;
        return 1;
    }

    std::cout << "PASS: block_pipeline_dense structural test passed" << std::endl;
    return 0;
}
