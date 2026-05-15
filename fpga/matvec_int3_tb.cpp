// matvec_int3_tb.cpp — host testbench for INT3×INT8 matvec.
//
// Build:
//   g++ -std=c++17 -I. -I./fpga -DFLLM_LANES=32 -DFLLM_TILE_ROWS=16 \
//       -o /tmp/matvec_int3_tb fpga/matvec_int3_tb.cpp && /tmp/matvec_int3_tb

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "fpga/hls_stubs.hpp"
#include "fpga/matvec_int3.hpp"

int main() {
    std::printf("=== matvec_int3_tile_dense testbench ===\n");

    constexpr int IN_F = 128;
    constexpr int OUT_F = 64;
    constexpr int W_BEAT_BYTES = LANES / 2;
    constexpr int BEATS_PER_ROW = IN_F / LANES;
    constexpr int ROWS = (OUT_F + TILE_ROWS - 1) / TILE_ROWS;

    // Random INT3 weights and INT8 activations
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> wdist(-4, 3);
    std::uniform_int_distribution<int> adist(-127, 127);

    std::vector<int8_t> w(OUT_F * IN_F);
    std::vector<int8_t> a(IN_F);
    for (auto& v : w) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : a) v = static_cast<int8_t>(adist(rng));

    // Reference
    std::vector<int32_t> y_ref(OUT_F, 0);
    for (int r = 0; r < OUT_F; ++r) {
        int32_t acc = 0;
        for (int c = 0; c < IN_F; ++c) {
            acc += static_cast<int32_t>(w[r * IN_F + c]) * static_cast<int32_t>(a[c]);
        }
        y_ref[r] = acc;
    }

    // Pack weights: 2 weights per byte, low nibble first
    std::vector<uint8_t> packed(OUT_F * BEATS_PER_ROW * W_BEAT_BYTES);
    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                int col = b * LANES + lane * 2;
                uint8_t lo = static_cast<uint8_t>(w[r * IN_F + col] & 0x07);
                uint8_t hi = static_cast<uint8_t>(w[r * IN_F + col + 1] & 0x07);
                packed[r * BEATS_PER_ROW * W_BEAT_BYTES + b * W_BEAT_BYTES + lane] = (hi << 4) | lo;
            }
        }
    }

    // Streams
    hls::stream<packed_w3_beat_t> s_w(10000);
    hls::stream<packed_a_beat_t> s_a(1000);
    hls::stream<ap_int<32>> s_y(500);

    for (int b = 0; b < BEATS_PER_ROW; ++b) {
        packed_a_beat_t ab;
        for (int j = 0; j < ab.LIMBS; ++j) ab.limb[j] = 0;
        for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
            int idx = b * LANES + lane * 2;
            ab.set_range(16 * lane + 7,  16 * lane + 0,  ap_uint<16>(static_cast<uint8_t>(a[idx + 0])));
            ab.set_range(16 * lane + 15, 16 * lane + 8,  ap_uint<16>(static_cast<uint8_t>(a[idx + 1])));
        }
        s_a.write(ab);
    }

    for (int r = 0; r < OUT_F; ++r) {
        for (int b = 0; b < BEATS_PER_ROW; ++b) {
            packed_w3_beat_t wb = 0;
            for (int lane = 0; lane < W_BEAT_BYTES; ++lane) {
                uint8_t byte = packed[r * BEATS_PER_ROW * W_BEAT_BYTES + b * W_BEAT_BYTES + lane];
                wb.set_range(8 * lane + 7, 8 * lane, ap_uint<8>(byte));
            }
            s_w.write(wb);
        }
    }

    float scales[TILE_ROWS];
    for (int i = 0; i < TILE_ROWS; ++i) scales[i] = 1.0f;
    matvec_int3_tile_dense<IN_F, OUT_F>(s_w, s_a, scales, s_y);

    bool ok = true;
    int64_t max_err = 0;
    for (int r = 0; r < OUT_F; ++r) {
        ap_int<32> v = s_y.read();
        int32_t hw = static_cast<int32_t>(v);
        int64_t err = std::llabs(static_cast<int64_t>(hw) - static_cast<int64_t>(y_ref[r]));
        if (err > max_err) max_err = err;
        if (hw != y_ref[r]) {
            std::printf("MISMATCH row %d: hw=%d ref=%d\n", r, hw, y_ref[r]);
            ok = false;
        }
    }

    std::printf("Max error: %lld %s\n", static_cast<long long>(max_err), ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
