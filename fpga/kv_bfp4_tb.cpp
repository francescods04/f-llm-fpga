// kv_bfp4_tb.cpp — host testbench for BFP4 pack/unpack.
//
// Build:
//   g++ -std=c++17 -I./fpga -o /tmp/kv_bfp4_tb fpga/kv_bfp4_tb.cpp && /tmp/kv_bfp4_tb

#include <cmath>
#include <cstdio>
#include <vector>

#include "hls_stubs.hpp"
#include "kv_bfp4.hpp"

int main() {
    std::printf("=== kv_bfp4 pack/unpack testbench ===\n");

    constexpr int BLOCK_SIZE = 32;
    constexpr int NUM_VALUES = 128;

    std::vector<float> values(NUM_VALUES);
    for (int i = 0; i < NUM_VALUES; ++i) values[i] = static_cast<float>(i) * 0.1f - 6.0f;

    hls::stream<float> f_in(256);
    for (float v : values) f_in.write(v);

    hls::stream<BFP4Block<BLOCK_SIZE>> bfp_stream(16);
    kv_bfp4_pack<BLOCK_SIZE>(f_in, bfp_stream, NUM_VALUES);

    hls::stream<float> f_out(256);
    kv_bfp4_unpack<BLOCK_SIZE>(bfp_stream, f_out, NUM_VALUES);

    bool ok = true;
    double max_err = 0.0;
    for (int i = 0; i < NUM_VALUES; ++i) {
        float ref = values[i];
        float hw = f_out.read();
        double err = std::abs(ref - hw);
        if (err > max_err) max_err = err;
        if (err > 0.5f) {
            std::printf("MISMATCH idx %d: ref=%.4f hw=%.4f err=%.4f\n", i, ref, hw, err);
            ok = false;
        }
    }

    std::printf("Max error: %.4f %s\n", max_err, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
