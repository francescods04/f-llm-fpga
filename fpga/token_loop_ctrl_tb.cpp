// token_loop_ctrl_tb.cpp — structural test for token-loop FSM.
//
// Build:
//   g++ -std=c++17 -I./fpga -o /tmp/token_loop_ctrl_tb fpga/token_loop_ctrl_tb.cpp && /tmp/token_loop_ctrl_tb

#include <cstdio>

#include "hls_stubs.hpp"
#include "token_loop_ctrl.hpp"

int main() {
    std::printf("=== token_loop_ctrl structural test ===\n");

    // Sampler test: argmax over a small vocab
    constexpr int VOCAB = 8;
    hls::stream<float> logits(16);
    float test_logits[VOCAB] = {0.1f, 0.5f, 0.2f, 0.05f, 0.8f, 0.0f, 0.3f, 0.1f};
    for (int i = 0; i < VOCAB; ++i) logits.write(test_logits[i]);

    hls::stream<int> token_out(4);
    sampler_argmax<VOCAB>(logits, token_out);

    int best = token_out.read();
    std::printf("Argmax index: %d (expected 4) %s\n", best, (best == 4) ? "PASS" : "FAIL");

    return (best == 4) ? 0 : 1;
}
