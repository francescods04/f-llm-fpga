// token_loop_ctrl.hpp — hardware token-loop FSM for end-to-end decode.
//
// This is the top-level controller referenced in docs/H100_BEAT_PLAN.md D3.
// It orchestrates N transformer blocks + LM head + sampler in a loop,
// driving the DATAFLOW pipeline and managing KV cache addressing.
//
// For the host-testbench we simulate the FSM state machine in C++.
//
// Build:
//   g++ -std=c++17 -I./fpga -o /tmp/token_loop_ctrl_tb fpga/token_loop_ctrl_tb.cpp \
//       && /tmp/token_loop_ctrl_tb

#ifndef FLLM_TOKEN_LOOP_CTRL_HPP
#define FLLM_TOKEN_LOOP_CTRL_HPP

#ifdef __SYNTHESIS__
#include <ap_int.h>
#include <hls_stream.h>
#else
#include "hls_stubs.hpp"
#endif

#include <cstdint>

// ---------------------------------------------------------------------------
// Token loop state machine (synthesizable on FPGA)
// ---------------------------------------------------------------------------
enum class TokenState : uint8_t {
    IDLE = 0,
    BLOCK_RUN = 1,      // running one transformer block
    LM_HEAD = 2,        // final projection to vocab
    SAMPLE = 3,         // argmax / temperature sampling
    KV_UPDATE = 4,      // append new K/V to cache
    DONE = 5,
};

// Control packet passed between blocks
template<int MAX_CTX>
struct TokenCtrl {
    TokenState state;
    int position;           // current sequence position (0 .. MAX_CTX-1)
    int current_len;        // number of tokens already in context
    int token_id;           // sampled token for this step
    bool eos_reached;
};

// ---------------------------------------------------------------------------
// Sampler: argmax over vocab logits stream
// ---------------------------------------------------------------------------
template<int VOCAB>
void sampler_argmax(
    hls::stream<float>& logits_in,   // VOCAB floats
    hls::stream<int>& token_id_out   // 1 int
) {
#pragma HLS INLINE off
#pragma HLS INTERFACE mode=axis port=logits_in
#pragma HLS INTERFACE mode=axis port=token_id_out

    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = 0; i < VOCAB; ++i) {
        #pragma HLS PIPELINE II=1
        float v = logits_in.read();
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }
    token_id_out.write(best_idx);
}

// ---------------------------------------------------------------------------
// Token loop FSM (skeleton)
// ---------------------------------------------------------------------------
template<int NUM_LAYERS, int MAX_CTX, int VOCAB>
void token_loop_fsm(
    hls::stream<float>& prompt_embedding,   // initial hidden state stream
    hls::stream<int>& output_tokens,        // generated token IDs
    // Block interfaces (array of streams, one per layer)
    hls::stream<float>* block_in[NUM_LAYERS],
    hls::stream<float>* block_out[NUM_LAYERS],
    // LM head
    hls::stream<float>& lm_head_in,
    hls::stream<float>& lm_head_out,
    // Control
    int max_new_tokens,
    int eos_token_id
) {
#pragma HLS INTERFACE mode=ap_ctrl_chain port=return
#pragma HLS INTERFACE mode=axis port=prompt_embedding
#pragma HLS INTERFACE mode=axis port=output_tokens
#pragma HLS INTERFACE mode=axis port=lm_head_in
#pragma HLS INTERFACE mode=axis port=lm_head_out

    TokenCtrl<MAX_CTX> ctrl;
    ctrl.state = TokenState::BLOCK_RUN;
    ctrl.position = 0;
    ctrl.current_len = 1;  // assume prompt already embedded
    ctrl.eos_reached = false;

    for (int gen = 0; gen < max_new_tokens && !ctrl.eos_reached; ++gen) {
        // ------------------------------------------------------------------
        // Run all layers
        // ------------------------------------------------------------------
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            // Handshake: forward hidden state through block
            // In real build this is a DATAFLOW call to block_pipeline_dense
            // Here we just copy stream tokens for structural test.
            int token_count = 0;  // placeholder; real shape is hidden_size
            // (Actual implementation would call block_pipeline_dense here)
        }

        // ------------------------------------------------------------------
        // LM head
        // ------------------------------------------------------------------
        // (Actual implementation would call matvec into vocab)

        // ------------------------------------------------------------------
        // Sample
        // ------------------------------------------------------------------
        int sampled = 0;  // placeholder
        if (sampled == eos_token_id) {
            ctrl.eos_reached = true;
        }
        output_tokens.write(sampled);

        // ------------------------------------------------------------------
        // KV update and position advance
        // ------------------------------------------------------------------
        ctrl.position++;
        ctrl.current_len++;
    }

    ctrl.state = TokenState::DONE;
}

#endif  // FLLM_TOKEN_LOOP_CTRL_HPP
